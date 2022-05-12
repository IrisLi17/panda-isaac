import math, os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, tf_combine, tensor_clamp
from panda_isaac.base_config import BaseConfig
from panda_isaac.base_task import BaseTask
import torch
from panda_isaac.utils.ik_utils import orientation_error, control_ik, control_osc, control_cartesian_pd
from collections import deque
import torchvision


class PandaPushEnv(BaseTask):
    def __init__(self, cfg, physics_engine=gymapi.SIM_PHYSX, sim_device="cuda:0", headless=False):
        self._parse_cfg(cfg)
        super().__init__(cfg, physics_engine, sim_device, headless)
        # create tensor buffers for simulation
        self._init_buffer()
        # for image observation
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.goal_in_air = 0.0

    def create_sim(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_envs()
    
    def _create_envs(self):
        asset_root = os.path.join(os.path.dirname(__file__), "asset")

        # create table asset
        table_dims = gymapi.Vec3(0.8, 1.0, 0.4)
        self.table_dims = [0.8, 1.0, 0.4]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # create box asset
        self.box_size = box_size = 0.05
        asset_options = gymapi.AssetOptions()
        box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, asset_options)

        # # create goal asset for visualization only
        # asset_options = gymapi.AssetOptions()
        # asset_options.density = 0.0
        # asset_options.fix_base_link = True
        # goal_asset = self.gym.create_sphere(self.sim, 0.025, asset_options)

        # load franka asset
        franka_asset_file = self.cfg.asset.robot_urdf
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        # asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # configure franka dofs
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_lower_limits = franka_dof_props["lower"]
        franka_upper_limits = franka_dof_props["upper"]
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        controller = self.cfg.control.controller
        if controller == "ik":
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            # franka_dof_props["stiffness"][:7].fill(400.0)
            # franka_dof_props["damping"][:7].fill(40.0)
            franka_dof_props["stiffness"][:7] = np.array([80.0, 120.0, 100.0, 100.0, 70.0, 50.0, 20.0])
            franka_dof_props["damping"][:7] = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
            # franka_dof_props["damping"][:7].fill(80.0)
        else:       # osc and cartesian_impedance
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(0.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)

        # default dof states and position targets
        self.franka_num_dofs = franka_num_dofs = self.gym.get_asset_dof_count(franka_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
        # "ready" pos
        # default_dof_pos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        # [0.4, 0.0, 0.7] pos
        default_dof_pos[:7] = np.array([0.0, -0.22146, 0.0, -2.7018, 0.0, 2.4804, 0.78533])
        # grippers open
        default_dof_pos[7:] = franka_upper_limits[7:]

        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device).unsqueeze(dim=0).repeat((self.num_envs, 1))
        self._default_dof_initialized = False

        self.franka_dof_effort = to_torch(franka_dof_props["effort"][:7])

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]
        
        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0, 0, 0.4)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.3, 0.0, 0.5 * table_dims.z)
        self.table_position = [0.3, 0.0, 0.5 * table_dims.z]

        box_pose = gymapi.Transform()
        # goal_pose = gymapi.Transform()

        self.envs = []
        self.box_idxs = []
        self.hand_idxs = []
        self.lfinger_idxs = []
        self.rfinger_idxs = []
        init_pos_list = []
        init_rot_list = []
        # self.goal_idxs = []
        self.cams = []
        self.cam_tensors = []

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)
            color = gymapi.Vec3(116 / 255, 142 / 256, 138 / 256)
            self.gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            # table_rs_props = self.gym.get_actor_rigid_shape_properties(env, table_handle)
            # if i == 0:
            #     print(len(table_rs_props))
            #     print(table_rs_props[0].compliance, table_rs_props[0].filter, table_rs_props[0].friction, 
            #           table_rs_props[0].restitution, table_rs_props[0].rolling_friction, 
            #           table_rs_props[0].thickness, table_rs_props[0].torsion_friction)
            if i == 0:
                self.table_handle = table_handle
            
            # add box
            box_pose.p.x = table_pose.p.x + np.random.uniform(-0.1, 0.3)
            box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            box_pose.p.z = table_dims.z + 0.5 * box_size
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            box_handle = self.gym.create_actor(env, box_asset, box_pose, "box", i, 0)
            if i == 0:
                self.box_handle = box_handle
                self.box_position = [box_pose.p.x, box_pose.p.y, box_pose.p.z]
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)

            # add franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)
            if i == 0:
                self.franka_handle = franka_handle

            # set dof properties
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

            # get inital hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # get global index of left and right fingers in rigid body state tensor
            lfinger_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_leftfinger", gymapi.DOMAIN_SIM)
            self.lfinger_idxs.append(lfinger_idx)
            rfinger_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_rightfinger", gymapi.DOMAIN_SIM)
            self.rfinger_idxs.append(rfinger_idx)

            # goal_handle = self.gym.create_actor(env, goal_asset, goal_pose, "goal", -1, 0)
            # if i == 0:
            #     self.goal_handle = goal_handle
            # goal_idx = self.gym.get_actor_rigid_body_index(env, goal_handle, 0, gymapi.DOMAIN_SIM)
            # self.goal_idxs.append(goal_idx)

            if self.cfg.obs.type == "pixel":
                # add camera on wrist
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cfg.cam.w
                cam_props.height = self.cfg.cam.h
                cam_props.horizontal_fov = self.cfg.cam.fov
                cam_props.supersampling_horizontal = self.cfg.cam.ss
                cam_props.supersampling_vertical = self.cfg.cam.ss
                cam_props.enable_tensors = True
                cam_handle = self.gym.create_camera_sensor(env, cam_props)
                if self.cfg.cam.view == "ego":
                    # add camera on wrist
                    rigid_body_hand_ind = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
                    local_t = gymapi.Transform()
                    local_t.p = gymapi.Vec3(*self.cfg.cam.loc_p)
                    xyz_angle_rad = [np.radians(a) for a in self.cfg.cam.loc_r]
                    local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                    self.gym.attach_camera_to_body(
                        cam_handle, env, rigid_body_hand_ind,
                        local_t, gymapi.FOLLOW_TRANSFORM
                    )
                elif self.cfg.cam.view == "third":
                    # add third-person view camera
                    rigid_body_table_ind = self.gym.get_actor_rigid_body_handle(env, table_handle, 0)
                    local_t = gymapi.Transform()
                    local_t.p = gymapi.Vec3(0.0, 0.0, 0.9)
                    local_t.r = gymapi.Quat.from_euler_zyx(np.radians(180.0), np.radians(90.0), 0.0)
                    self.gym.attach_camera_to_body(
                        cam_handle, env, rigid_body_table_ind, local_t, gymapi.FOLLOW_TRANSFORM
                    )
                    # self.gym.set_camera_location(
                    #     cam_handle, env, 
                    #     gymapi.Vec3(0.5, 0.0, 1.0), 
                    #     gymapi.Vec3(0.5, 0.0001, 0.5),
                    # )
                else:
                    raise NotImplementedError
                self.cams.append(cam_handle)
                # Camera tensor
                cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                self.cam_tensors.append(cam_tensor_th)

    def _parse_cfg(self, cfg: BaseConfig):
        self.max_episode_length = cfg.env.max_episode_length
        self.damping = None
        self.kp, self.kd, self.kp_null, self.kd_null = None, None, None, None
        if cfg.control.controller == "cartesian_impedance":
            self.kp = torch.diag(torch.cat([torch.ones(3) * cfg.control.kp_translation, torch.ones(3) * cfg.control.kp_rotation]).float()).unsqueeze(dim=0)
            self.kd = torch.diag(torch.cat([torch.ones(3) * cfg.control.kd_translation, torch.ones(3) * cfg.control.kd_rotation]).float()).unsqueeze(dim=0)
            self.kp_null = cfg.control.kp_null
            self.kd_null = cfg.control.kd_null
        elif cfg.control.controller == "osc":
            self.kp = cfg.control.kp
            self.kd = cfg.control.kd
            self.kp_null = cfg.control.kp_null
            self.kd_null = cfg.control.kd_null
        elif cfg.control.controller == "ik":
            self.damping = cfg.control.damping

    def _init_buffer(self):
        # get base state tensor
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_actor_root_state)
        self.actor_ids_int32 = torch.arange(0, self.root_state.shape[0], dtype=torch.int32, device=self.device, requires_grad=False).reshape((self.num_envs, -1))
        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.gym.refresh_jacobian_tensors(self.sim)
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.franka_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.gym.refresh_mass_matrix_tensors(self.sim)
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7]          # only need elements corresponding to the franka arm

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        # self.box_pos = self.rb_states[self.box_idxs, :3].view(self.num_envs, 3)
        # self.eef_pos = self.rb_states[self.hand_idxs, :3].view(self.num_envs, 3)
        # self.eef_orn = self.rb_states[self.hand_idxs, 3:7].view(self.num_envs, 4)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 9, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 9, 1)

        # get contact force tensor
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf)

        self.local_grasp_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.local_grasp_pos[:, 2] = 0.1034
        self.local_grasp_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.local_grasp_rot[:, 3] = 1

        # prepare image history if any
        if self.cfg.obs.type == "pixel":
            self.image_history = deque(maxlen=self.cfg.obs.history_length)
            for _ in range(self.cfg.obs.history_length):
                self.image_history.append(torch.zeros((self.num_envs, 3 * self.cfg.obs.im_size * self.cfg.obs.im_size), dtype=torch.float, device=self.device))
            self.image_transform = torchvision.transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.1)
        self.state_history = deque(maxlen=self.cfg.obs.state_history_length)
        # num_state = self.num_obs // self.cfg.obs.state_history_length if self.cfg.obs.type == "state" else (self.num_obs - 3 * self.cfg.obs.im_size * self.cfg.obs.im_size * self.cfg.obs.history_length) // self.cfg.obs.state_history_length
        num_state = self.num_state_obs // self.cfg.obs.state_history_length
        for _ in range(self.cfg.obs.state_history_length):
            self.state_history.append(torch.zeros((self.num_envs, num_state), dtype=torch.float, device=self.device))
        
        # prepare obs
        self.target_eef_pos_obs = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.noise_vec = torch.zeros((1, num_state), dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.obs.noise:
            assert num_state == 18
            self.noise_vec[:, :3] = 0.01
            self.noise_vec[:, 3:6] = 1e-3
            self.noise_vec[:, 6:10] = 0.0
            self.noise_vec[:, 10:12] = 0.005
            self.noise_vec[:, 12:18] = 0.0
        # prepare buffers for actions
        self.motor_pos_target = torch.zeros(self.num_envs, self.franka_num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.effort_action = torch.zeros_like(self.motor_pos_target)
        self.last_torques = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device)
        self.target_eef_orn = torch.tensor([[1.0, 0., 0., 0.]], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.target_eef_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.target_finger = torch.ones((self.num_envs, 1), dtype=torch.float, device=self.device, requires_grad=False)
        if isinstance(self.kp, torch.Tensor):
            self.kp = self.kp.to(self.device)
        if isinstance(self.kd, torch.Tensor):
            self.kd = self.kd.to(self.device)
        # prepare buffers for goal
        self.box_goals = torch.zeros_like(self.rb_states[self.box_idxs, :3], requires_grad=False)

        self.last_distance = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)

        # safety
        self.is_brake = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # prepare buffers for episode info
        self.episode_sums = {
            "r": torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False),
            "l": torch.zeros((self.num_envs,), dtype=torch.int, device=self.device, requires_grad=False),
            "is_success": torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False),
        }
        self.extras["episode"] = {k: None for k in self.episode_sums}

        self.common_step_counter = 0

    def _init_default_dof(self):
        desired_x = self.table_position[0] + torch.rand((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False) * 0.4 - 0.1
        desired_y = self.table_position[1] + torch.rand((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False) * 0.6 - 0.3
        desired_z = 0.6 + torch.rand((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False) * 0.2
        desired_pos = torch.stack([desired_x, desired_y, desired_z], dim=-1)
        for _ in range(10):
            pos_error = desired_pos - self.rb_states[self.hand_idxs, :3]
            rot_error = orientation_error(self.target_eef_orn, self.rb_states[self.hand_idxs, 3:7])
            dpose = torch.cat([pos_error, rot_error], dim=-1).unsqueeze(dim=-1)
            dof_pos = self.dof_pos.squeeze(-1)[:, :7] + control_ik(dpose, self.j_eef, 0.05)
            self.dof_pos[: ,:7, 0] = dof_pos
            self.motor_pos_target[:, :7] = dof_pos
            self.effort_action[:] = 0
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.motor_pos_target))
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.effort_action))
       
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.default_dof_pos_tensor[:, :7] = self.dof_pos[:, :7, 0]
        self._default_dof_initialized = True
    
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not self._default_dof_initialized:
            self._init_default_dof()
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_goals(env_ids)
        self.is_brake[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        for k in self.episode_sums:
            self.episode_sums[k][env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_torques[env_ids] = 0
        # reset image history if any
        if self.cfg.obs.type == "pixel":
            for i in range(len(self.image_history)):
                self.image_history[i][env_ids] = 0
        for i in range(len(self.state_history)):
            self.state_history[i][env_ids] = 0
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # Set last distance
        self.last_distance[env_ids] = torch.norm(self.rb_states[self.box_idxs, :3][env_ids] - self.box_goals[env_ids], dim=-1)
        if self.cfg.reward.type == "dense":
            hand_rot = self.rb_states[self.hand_idxs, 3:7][env_ids]
            hand_pos = self.rb_states[self.hand_idxs, :3][env_ids]
            tcp_rot, tcp_pos = tf_combine(
                hand_rot, hand_pos,
                self.local_grasp_rot[env_ids], self.local_grasp_pos[env_ids]
            )
            tcp2obj = torch.norm(tcp_pos - self.rb_states[self.box_idxs, :3][env_ids], dim=-1)
            self.last_distance[env_ids] += 0.1 * tcp2obj
    
    def _reset_dofs(self, env_ids):
        # Important! should feed actor id, not env id
        actor_ids_int32 = self.actor_ids_int32[env_ids, self.franka_handle].flatten()
        self.dof_pos[env_ids, :, 0] = self.default_dof_pos_tensor[env_ids]
        
        self.dof_vel[env_ids, :, 0] = 0
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        self.motor_pos_target[env_ids, :] = self.default_dof_pos_tensor[env_ids]
        self.effort_action[env_ids, :] = 0
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.motor_pos_target), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.effort_action), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))

    def _reset_root_states(self, env_ids):
        # Randomize box position
        actor_ids_int32 = self.actor_ids_int32[env_ids, self.box_handle].flatten()
        actor_ids_long = actor_ids_int32.long()
        self.root_state[actor_ids_long, 0] = self.table_position[0] + torch.rand(size=actor_ids_int32.shape, dtype=torch.float, device=self.device) * 0.4 - 0.1
        self.root_state[actor_ids_long, 1] = self.table_position[1] + torch.rand(size=actor_ids_int32.shape, dtype=torch.float, device=self.device) * 0.6 - 0.3
        self.root_state[actor_ids_long, 2] = self.box_position[2]
        self.root_state[actor_ids_long, 7:13] = 0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32)
        )
    
    def _resample_goals(self, env_ids):
        self.box_goals[env_ids, 0] = self.table_position[0] + torch.rand(size=env_ids.shape, dtype=torch.float, device=self.device) * 0.4 - 0.1
        self.box_goals[env_ids, 1] = self.table_position[1] + torch.rand(size=env_ids.shape, dtype=torch.float, device=self.device) * 0.6 - 0.3
        self.box_goals[env_ids, 2] = self.table_dims[2] + self.box_size / 2
        if np.random.uniform() < self.goal_in_air:
            self.box_goals[env_ids, 2] += torch.rand(size=env_ids.shape, dtype=torch.float, device=self.device) * 0.4
        # actor_ids_int32 = self.actor_ids_int32[env_ids, self.goal_handle].flatten()
        # actor_ids_long = actor_ids_int32.long()
        # self.root_state[actor_ids_long, 0] = self.box_goals[env_ids, 0]
        # self.root_state[actor_ids_long, 1] = self.box_goals[env_ids, 1]
        # self.root_state[actor_ids_long, 2] = self.box_goals[env_ids, 2]
        # self.gym.set_actor_root_state_tensor_indexed(
        #     self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32)
        # )
    
    def _set_dof_position(self, target_dof):
        actor_ids_int32 = self.actor_ids_int32[:, self.franka_handle].flatten().contiguous()
        self.dof_pos[:, :, 0] = target_dof[:]
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
    def step(self, actions):
        actions = torch.clone(actions)
        if actions.shape[1] == 5:
            action_type = actions[:, 0:1]
            pos_action = torch.where(action_type == 0, torch.clamp(actions[:, 1:4], -1.0, 1.0), torch.zeros_like(actions[:, 1:4]))
            gripper_action = torch.where(action_type == 1, torch.clamp(actions[:, 4:5], -1.0, 1.0), self.target_finger)
        elif actions.shape[1] == 4:
            pos_action = torch.clamp(actions[:, :3], -1.0, 1.0)
            gripper_action = torch.clamp(actions[:, 3:], -1.0, 1.0)
        eef_target = self.rb_states[self.hand_idxs, :3] + 0.05 * pos_action
        filtered_pos_target = self.rb_states[self.hand_idxs, :3]

        _, self.target_eef_pos_obs = tf_combine(
            self.rb_states[self.hand_idxs, 3:7], eef_target,
            self.local_grasp_rot, self.local_grasp_pos
        )
        # TODO: do safe clip, find out where is "hand"
        # eef_target[:, 0] = torch.clamp(eef_target[:, 0], min=self.table_position[0] - 0.15, max=self.table_position[0] + 0.35)
        # eef_target[:, 1] = torch.clamp(eef_target[:, 1], min=self.table_position[1] - 0.35, max=self.table_position[1] + 0.35)
        # eef_target[:, 2] = torch.clamp(eef_target[:, 2], min=0.51)
        self.target_eef_pos[:] = eef_target
        self.target_finger[:] = gripper_action
        initial_error = eef_target[0] - self.rb_states[self.hand_idxs, :3][0]
        for i in range(self.cfg.control.decimal):
            filtered_pos_target = self.cfg.control.filter_param * eef_target + (1 - self.cfg.control.filter_param) * filtered_pos_target
            pos_error = filtered_pos_target - self.rb_states[self.hand_idxs, :3]
            orn_error = orientation_error(self.target_eef_orn, self.rb_states[self.hand_idxs, 3:7])
            dpose = torch.cat([pos_error, orn_error], dim=-1).unsqueeze(dim=-1)
            if self.cfg.control.controller == "ik":
                self.motor_pos_target[:, :7] = self.dof_pos[:, :7, 0] + control_ik(dpose, self.j_eef, self.damping)
            elif self.cfg.control.controller == "osc":
                calculate_torque = control_osc(
                    dpose, self.kp, self.kd, self.kp_null, self.kd_null,
                    self.default_dof_pos_tensor, self.mm, self.j_eef, self.dof_pos, self.dof_vel, self.j_eef @ self.dof_vel[:, :7]
                )
                diff = calculate_torque - self.last_torques
                self.effort_action[:, :7] = tensor_clamp(
                    self.last_torques + torch.clamp(diff, -1000 * self.sim_params.dt, 1000 * self.sim_params.dt),
                    -self.franka_dof_effort, self.franka_dof_effort
                )
                self.last_torques[:] = self.effort_action[:, :7]
            elif self.cfg.control.controller == "cartesian_impedance":
                # self.effort_action[:, :7] = control_cartesian_impedance(
                #     dpose, self.kp, self.kd, self.kp_null, self.kd_null,
                #     self.default_dof_pos_tensor[:, :7], self.j_eef, self.dof_pos[:, :7], self.dof_vel[:, :7]
                # )
                hand_rot = self.rb_states[self.hand_idxs, 3:7]
                hand_pos = self.rb_states[self.hand_idxs, :3]
                tcp_rot, tcp_pos = tf_combine(
                    hand_rot, hand_pos,
                    self.local_grasp_rot, self.local_grasp_pos
                )
                tcp_desired_rot, tcp_desired_pos = tf_combine(
                    self.target_eef_orn, self.target_eef_pos, 
                    self.local_grasp_rot, self.local_grasp_pos
                )
                ee_vel_desired = torch.zeros_like(tcp_pos)
                ee_rvel_desired = torch.zeros_like(tcp_pos)
                self.effort_action[:, :7] = control_cartesian_pd(
                    self.dof_pos[:, :7], self.dof_vel[:, :7], tcp_pos, tcp_rot, self.j_eef, 
                    tcp_desired_pos, tcp_desired_rot,
                    ee_vel_desired, ee_rvel_desired, self.kp, self.kd
                )
            self.motor_pos_target[:, 7:9] = 0.02 + 0.02 * gripper_action.repeat((1, 2))
            
            # print(self.dof_pos[0, :, 0])
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.motor_pos_target))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

        self.gym.refresh_net_contact_force_tensor(self.sim)
        end_error = self.target_eef_pos[0] - self.rb_states[self.hand_idxs, :3][0]
        if self.cfg.obs.type == "pixel" or self.viewer is not None:
            self.gym.step_graphics(self.sim)
        # update viewer
        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        self.common_step_counter += 1
        self.post_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_step(self):
        self.episode_length_buf += 1
        self.episode_sums["l"] += 1
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids):
            self.extras["episode"] = {}
            for k in self.episode_sums:
                self.extras["episode"][k] = self.episode_sums[k][env_ids].clone()
        else:
            self.extras = {}
        self.extras["time_out"] = self.time_out_buf.clone()
        self.reset_idx(env_ids)
        self.light_randomization()
        self.compute_observations()
    
    def check_termination(self):
        box_pos = self.rb_states[self.box_idxs, :3]
        # hand_pos = self.rb_states[self.hand_idxs, :3]
        # hand_goal = box_pos + torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float, device=self.device)
        self.reset_buf = torch.norm(box_pos - self.box_goals, dim=-1) < self.box_size
        # self.reset_buf = torch.norm(hand_pos - hand_goal, dim=-1) < self.box_size
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.time_out_buf
        if self.cfg.safety.brake_on_contact:
            self.is_brake = torch.logical_or(self.net_cf[self.lfinger_idxs, 2] > self.cfg.safety.contact_force_th, 
                                             self.net_cf[self.rfinger_idxs, 2] > self.cfg.safety.contact_force_th)
            self.reset_buf |= self.is_brake
    
    def compute_reward(self):
        box_pos = self.rb_states[self.box_idxs, :3]
        # hand_pos = self.rb_states[self.hand_idxs, :3]
        # hand_goal = box_pos + torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float, device=self.device)
        distance = torch.norm(box_pos - self.box_goals, dim=-1)
        # distance = torch.norm(hand_pos - hand_goal, dim=-1)
        if self.cfg.reward.type == "sparse":
            rew = torch.logical_and(distance < self.box_size, self.episode_length_buf > 1).float()
        elif self.cfg.reward.type == "dense":
            hand_rot = self.rb_states[self.hand_idxs, 3:7]
            hand_pos = self.rb_states[self.hand_idxs, :3]
            tcp_rot, tcp_pos = tf_combine(
                hand_rot, hand_pos,
                self.local_grasp_rot, self.local_grasp_pos
            )
            tcp2obj = torch.norm(tcp_pos - box_pos, dim=-1)
            total_distance = distance + 0.1 * tcp2obj
            # rew = torch.clamp(self.last_distance - distance, min=0)
            bonus = torch.logical_and(distance < self.box_size, self.episode_length_buf > 1)
            finger_contact = self.is_brake
            rew = torch.clamp(self.last_distance - total_distance, min=0) + bonus.float() + self.cfg.reward.contact_coef * finger_contact.float()
            self.last_distance = total_distance
        else:
            raise NotImplementedError
        self.rew_buf = rew
        self.episode_sums["r"] += rew
        self.episode_sums["is_success"] += (distance < self.box_size).float()
    
    def light_randomization(self):
        if self.common_step_counter % 1 == 0:
            # weaker randomization
            l_color = gymapi.Vec3(np.random.uniform(1, 1), np.random.uniform(1, 1), np.random.uniform(1, 1))
            # l_ambient = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            _ambient = np.random.uniform(0, 1)
            l_ambient = gymapi.Vec3(_ambient, _ambient, _ambient)
            l_direction = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
    
    def compute_observations(self):
        self.state_history.append(torch.zeros_like(self.state_history[0]))
        if self.cfg.obs.type == "pixel":
            # Image based
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.image_history.append(torch.zeros_like(self.image_history[0]))
            for i in range(self.num_envs):
                crop_l = (self.cfg.cam.w - self.cfg.obs.im_size) // 2 if self.cfg.cam.crop == "center" else 0
                crop_r = crop_l + self.cfg.obs.im_size
                # TODO: apply random shift
                _rgb_obs = self.cam_tensors[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
                # _rgb_obs = ((_rgb_obs - self.im_mean) / self.im_std).flatten()
                _rgb_obs = _rgb_obs.flatten()
                self.image_history[-1][i, :] = _rgb_obs
                # self.obs_buf[i, :3 * self.cfg.obs.im_size ** 2] = _rgb_obs
            for i in range(len(self.image_history)):
                _image = ((self.image_transform(self.image_history[i].reshape(
                    (self.num_envs, 3, self.cfg.obs.im_size, self.cfg.obs.im_size)
                )) - self.im_mean.unsqueeze(dim=0)) / self.im_std.unsqueeze(dim=0)).reshape((self.num_envs, -1))
                self.obs_buf[:, 3 * self.cfg.obs.im_size * self.cfg.obs.im_size * i: 
                                3 * self.cfg.obs.im_size * self.cfg.obs.im_size * (i + 1)] = _image
            self.gym.end_access_image_tensors(self.sim)
            self.state_history[-1][:, :3] = self.rb_states[self.box_idxs, :3]
            start_idx = 3 * self.cfg.obs.im_size ** 2 * len(self.image_history)
            state_start_idx = 3
        elif self.cfg.obs.type == "state":
            self.state_history[-1][:, :3] = self.rb_states[self.box_idxs, :3]
            start_idx = 0
            state_start_idx = 0
        else:
            raise NotImplementedError
        # Low dimensional states
        hand_rot = self.rb_states[self.hand_idxs, 3:7]
        hand_pos = self.rb_states[self.hand_idxs, :3]
        tcp_rot, tcp_pos = tf_combine(
            hand_rot, hand_pos,
            self.local_grasp_rot, self.local_grasp_pos
        )
        self.state_history[-1][:, 3:] = torch.cat([
            tcp_pos, tcp_rot, self.dof_pos[:, 7:9, 0], self.target_eef_pos_obs, self.box_goals
        ], dim=-1)
        state_dim = self.num_state_obs // self.cfg.obs.state_history_length - state_start_idx
        for i in range(len(self.state_history)):
            self.obs_buf[:, start_idx + i * state_dim : start_idx + (i + 1) * state_dim] = self.state_history[i][:, state_start_idx:] + torch.randn_like(self.noise_vec[:, state_start_idx:]) * self.noise_vec[:, state_start_idx]
    
    def get_state_obs(self):
        return torch.cat([self.state_history[i] for i in range(len(self.state_history))], dim=-1)
    
    def set_goal_in_air_ratio(self, goal_in_air):
        self.goal_in_air = goal_in_air
    
    def set_contact_force_th(self, contact_force_th):
        self.cfg.safety.contact_force_th = contact_force_th

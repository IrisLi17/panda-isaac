import math
import random
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_apply, tf_combine
from panda_isaac.base_config import BaseConfig
from panda_isaac.base_task import BaseTask
import torch
from panda_isaac.utils.ik_utils import orientation_error, control_ik, control_osc, control_cartesian_impedance


class PandaPushEnv(BaseTask):
    def __init__(self, cfg, physics_engine=gymapi.SIM_PHYSX, sim_device="cuda:0", headless=False):
        self._parse_cfg(cfg)
        super().__init__(cfg, physics_engine, sim_device, headless)
        # create tensor buffers for simulation
        self._init_buffer()
        # for image observation
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.goal_in_air = 0

    def create_sim(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_envs()
    
    def _create_envs(self):
        asset_root = "/home/yunfei/projects/isaac_projects/isaacgym/assets"

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

        # load franka asset
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
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
        self.franka_lower_limits = to_torch(franka_lower_limits)
        franka_upper_limits = franka_dof_props["upper"]
        self.franka_upper_limits = to_torch(franka_upper_limits)
        franka_ranges = franka_upper_limits - franka_lower_limits
        franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

        # use position drive for all dofs
        controller = self.cfg.control.controller
        assert controller == "joint"
        # franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
        # franka_dof_props["stiffness"][:7].fill(400.0)
        # franka_dof_props["damping"][:7].fill(80.0)
        franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        franka_dof_props["stiffness"][:7].fill(0.0)
        franka_dof_props["stiffness"][:7].fill(0.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(1.0e6)
        franka_dof_props["damping"][7:].fill(1.0e2)
        franka_dof_props["effort"][7:] = 200
        # franka_dof_props["stiffness"][7:].fill(800.0)
        # franka_dof_props["damping"][7:].fill(40.0)
        
        self.franka_dof_stiffness = to_torch([400.0] * 7 + [1.0e6] * 2)
        self.franka_dof_damping = to_torch([80.0] * 7 + [1.0e2] * 2)
        # new2
        # self.franka_dof_stiffness = to_torch([600.0] * 4 + [250.0, 150.0, 50.0] + [1.0e6] * 2)
        # self.franka_dof_damping = to_torch([50.0] * 3 + [20.0] * 3 + [10.0] + [1.0e2] * 2)
        self.franka_dof_effort = to_torch(franka_dof_props["effort"])
        
        self.franka_dof_speed_scales = 0.8 * to_torch(franka_dof_props["velocity"])
        self.franka_dof_speed_scales[7:] = 0.1


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

        self.envs = []
        self.box_idxs = []
        self.hand_idxs = []
        self.lfinger_idxs, self.rfinger_idxs = [], []
        init_pos_list = []
        init_rot_list = []
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

            lfinger_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_leftfinger", gymapi.DOMAIN_SIM)
            rfinger_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_rightfinger", gymapi.DOMAIN_SIM)
            self.lfinger_idxs.append(lfinger_idx)
            self.rfinger_idxs.append(rfinger_idx)

            if self.cfg.obs.type == "pixel":
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
                    self.gym.set_camera_location(
                        cam_handle, env, 
                        gymapi.Vec3(0.8, 0.0, 0.8), 
                        gymapi.Vec3(0.3, 0.0, 0.5),
                    )
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

    def _init_buffer(self):
        # get base state tensor
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_actor_root_state)
        self.actor_ids_int32 = torch.arange(0, self.root_state.shape[0], dtype=torch.int32, device=self.device, requires_grad=False).reshape((self.num_envs, -1))
        
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, 9, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, 9, 1)

        self.init_hand_vector = torch.tensor([[0., 0., 1.]], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.downward_vector = torch.tensor([[0., 0., -1.0]], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.local_grasp_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.local_grasp_pos[:, 2] = 0.045
        self.local_grasp_rot = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.local_grasp_rot[:, 3] = 1

        # prepare buffers for actions
        self.motor_pos_target = torch.zeros(self.num_envs, self.franka_num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.franka_num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # prepare buffers for goal
        self.box_goals = torch.zeros_like(self.rb_states[self.box_idxs, :3], requires_grad=False)

        # prepare buffers for episode info
        self.episode_sums = {
            "r": torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False),
            "l": torch.zeros((self.num_envs,), dtype=torch.int, device=self.device, requires_grad=False),
            "is_success": torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False),
        }
        self.extras["episode"] = {k: None for k in self.episode_sums}
        
    def _init_default_dof(self):
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        j_eef = jacobian[:, self.franka_hand_index - 1, :, :7]

        # desired_x = self.table_position[0] + torch.rand((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False) * 0.4 - 0.1
        desired_x = 0.4 * torch.ones((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)
        # desired_y = self.table_position[1] + torch.rand((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False) * 0.6 - 0.3
        desired_y = 0.0 * torch.ones((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)
        # desired_z = 0.52 + torch.rand((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False) * 0.2
        desired_z = 0.7 * torch.ones((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)
        desired_pos = torch.stack([desired_x, desired_y, desired_z], dim=-1)
        for _ in range(10):
            pos_error = desired_pos - self.rb_states[self.hand_idxs, :3]
            desired_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(dim=0).repeat((self.num_envs, 1))
            rot_error = orientation_error(desired_rot, self.rb_states[self.hand_idxs, 3:7])
            dpose = torch.cat([pos_error, rot_error], dim=-1).unsqueeze(dim=-1)
            dof_pos = self.dof_pos.squeeze(-1)[:, :7] + control_ik(dpose, j_eef, 0.05)
            self.dof_pos[: ,:7, 0] = dof_pos
            self.motor_pos_target[:, :7] = dof_pos
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.motor_pos_target))
            torques = torch.zeros_like(self.last_torques)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torques))
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
        # if not self._default_dof_initialized:
        #     self._init_default_dof()
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_goals(env_ids)
        self.episode_length_buf[env_ids] = 0
        for k in self.episode_sums:
            self.episode_sums[k][env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_torques[env_ids] = 0
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        # print("achieved", self.rb_states[self.hand_idxs, :3], self.dof_pos[0, :, 0])
        # assert abs(self.rb_states[self.box_idxs[env_ids[0]], 2] - 0.425) < 1e-4, self.rb_states[self.box_idxs[env_ids[0]]]
    
    def _reset_dofs(self, env_ids):
        # Important! should feed actor id, not env id
        actor_ids_int32 = self.actor_ids_int32[env_ids, self.franka_handle].flatten()
        dof_pos_noise = torch.rand((len(env_ids), self.franka_num_dofs), device=self.device)
        if False:
            dof_pos = tensor_clamp(
                # new3 0.5, new4 revert
                # self.default_dof_pos_tensor.unsqueeze(0) + 0.25 * (dof_pos_noise - 0.5),
                self.dof_pos[env_ids, :, 0] + 0.25 * (dof_pos_noise - 0.5),
                self.franka_lower_limits, self.franka_upper_limits
            )
        else:
            dof_pos = tensor_clamp(
                # new3 0.5, new4 revert
                self.default_dof_pos_tensor[env_ids] + 0.25 * (dof_pos_noise - 0.5),
                self.franka_lower_limits, self.franka_upper_limits
            )
        # new4
        dof_pos[:, 8] = dof_pos[:, 7]
        self.dof_pos[env_ids, :, 0] = dof_pos
        # print("desired", self.dof_pos[0, :, 0])
        
        self.dof_vel[env_ids, :, 0] = 0
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        self.motor_pos_target[env_ids, :] = dof_pos
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.motor_pos_target), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        torques = torch.zeros_like(self.last_torques)
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(torques), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        
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
        # for evaluation only
        # self.box_goals[env_ids, 0] = 0.5
        # self.box_goals[env_ids, 1] = 0.0
    
    def step(self, actions):
        # actions = torch.clone(actions)
        # force closed finger
        # actions[:, -1] = -1.0
        actions = torch.cat([actions, actions[:, -1:]], dim=-1)
        assert actions.shape[-1] == self.franka_num_dofs
        actions = torch.clamp(actions, -1.0, 1.0)
        # new4 revert
        # joint_target = self.motor_pos_target + self.franka_dof_speed_scales * self.sim_params.dt * actions * self.cfg.control.decimal
        # new3
        joint_target = self.dof_pos.squeeze(dim=-1) + self.franka_dof_speed_scales * self.sim_params.dt * actions * self.cfg.control.decimal
        self.motor_pos_target[:, :] = tensor_clamp(joint_target, self.franka_lower_limits, self.franka_upper_limits)

        for i in range(self.cfg.control.decimal):
            cal_torques = self.franka_dof_stiffness * (self.motor_pos_target - self.dof_pos[:, :, 0]) + self.franka_dof_damping * (-self.dof_vel[:, :, 0])
            diff = cal_torques - self.last_torques
            torques = self.last_torques + torch.clamp(diff, -1000 * self.sim_params.dt, 1000 * self.sim_params.dt)
            torques = tensor_clamp(torques, -self.franka_dof_effort, self.franka_dof_effort)
            # print(i, torques[0])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.motor_pos_target))
            self.last_torques[:] = torques
            self.render()
            # step the physics
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # print(self.motor_pos_target[0] - self.dof_pos[0, :, 0])
        if self.cfg.obs.type == "pixel" or self.viewer is not None:
            self.gym.step_graphics(self.sim)
        # update viewer
        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
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
        self.compute_observations()
    
    def check_termination(self):
        box_pos = self.rb_states[self.box_idxs, :3]
        # hand_pos = self.rb_states[self.hand_idxs, :3]
        # hand_goal = box_pos + torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float, device=self.device)
        self.reset_buf = torch.norm(box_pos - self.box_goals, dim=-1) < self.box_size
        # self.reset_buf = torch.norm(hand_pos - hand_goal, dim=-1) < self.box_size
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        self.reset_buf |= self.time_out_buf
    
    def compute_reward(self):
        box_pos = self.rb_states[self.box_idxs, :3]
        # hand_pos = self.rb_states[self.hand_idxs, :3]
        # hand_goal = box_pos + torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float, device=self.device)
        distance = torch.norm(box_pos - self.box_goals, dim=-1)
        # distance = torch.norm(hand_pos - hand_goal, dim=-1)
        hand_direction = quat_apply(self.rb_states[self.hand_idxs, 3:7], self.init_hand_vector)
        is_downward = torch.sum(hand_direction * self.downward_vector, dim=-1) > 0.8
        if self.cfg.reward.type == "sparse":
            rew = (distance < self.box_size).float() - 0.1 * (~is_downward).float()
        elif self.cfg.reward.type == "dense":
            lfinger_pos = self.rb_states[self.lfinger_idxs, :3]
            lfinger_rot = self.rb_states[self.lfinger_idxs, 3:7]
            rfinger_pos = self.rb_states[self.rfinger_idxs, :3]
            rfinger_rot = self.rb_states[self.rfinger_idxs, 3:7]
            lfinger_grasp_rot, lfinger_grasp_pos = tf_combine(
                lfinger_rot, lfinger_pos,
                self.local_grasp_rot, self.local_grasp_pos
            )
            rfinger_grasp_rot, rfinger_grasp_pos = tf_combine(
                rfinger_rot, rfinger_pos,
                self.local_grasp_rot, self.local_grasp_pos
            )
            lfinger2obj = torch.clamp(torch.norm(box_pos - lfinger_grasp_pos, dim=-1), min=0.025)
            rfinger2obj = torch.clamp(torch.norm(box_pos - rfinger_grasp_pos, dim=-1), min=0.025)
            rew = 0.1 * (-(~is_downward).float() - (lfinger2obj - 0.025) - (rfinger2obj - 0.025)) + (distance < self.box_size).float()
        else:
            raise NotImplementedError
        self.rew_buf = rew
        self.episode_sums["r"] += rew
        self.episode_sums["is_success"] += (distance < self.box_size).float()
    
    def compute_observations(self):
        if self.cfg.obs.type == "pixel":
            # Image based
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for i in range(self.num_envs):
                crop_l = (self.cfg.cam.w - self.cfg.obs.im_size) // 2 if self.cfg.cam.crop == "center" else 0
                crop_r = crop_l + self.cfg.obs.im_size
                _rgb_obs = self.cam_tensors[i][:, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
                _rgb_obs = ((_rgb_obs - self.im_mean) / self.im_std).flatten()
                self.obs_buf[i, :3 * self.cfg.obs.im_size ** 2] = _rgb_obs
            self.gym.end_access_image_tensors(self.sim)
            start_idx = 3 * self.cfg.obs.im_size ** 2
        elif self.cfg.obs.type == "state":
            self.obs_buf[:, :3] = self.rb_states[self.box_idxs, :3]
            start_idx = 3
        else:
            raise NotImplementedError
        # Low dimensional states
        self.obs_buf[:, start_idx:] = torch.cat([
            2 * (self.dof_pos[:, :8, 0] - self.franka_lower_limits[:8]) / (self.franka_upper_limits[:8] - self.franka_lower_limits[:8]) - 1, 
            self.dof_vel[:, :8, 0] * self.sim_params.dt, 
            2 * (self.motor_pos_target[:, :8] - self.franka_lower_limits[:8]) / (self.franka_upper_limits[:8] - self.franka_lower_limits[:8]) - 1,
            self.rb_states[self.hand_idxs, :3],
            self.box_goals
        ], dim=-1)
    
    def set_goal_in_air_ratio(self, goal_in_air):
        self.goal_in_air = goal_in_air

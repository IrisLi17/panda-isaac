# Adapted from ETH legged gym
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch
from panda_isaac.base_config import BaseConfig
from panda_isaac.utils.helper import set_seed


# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg: BaseConfig, physics_engine=gymapi.SIM_PHYSX, sim_device="cuda:0", headless=False):
        self.gym = gymapi.acquire_gym()

        self.cfg = cfg
        
        set_seed(self.cfg.env.seed)
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True
        if physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = 0
            sim_params.physx.use_gpu = True
        else:
            raise Exception("This example can only be used with PhysX")
        self.sim_params = sim_params

        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        # if self.headless == True:
        #     self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_actions = cfg.env.num_actions
        self.num_state_obs = cfg.env.num_state_obs

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            cam_pos = gymapi.Vec3(4, 3, 2)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + int(np.sqrt(self.num_envs)) // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)
        else:
            camera_properties = gymapi.CameraProperties()
            camera_properties.width = 320
            camera_properties.height = 320
            camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_properties)
            self.gym.set_camera_location(camera_handle, self.envs[0], gymapi.Vec3(1.0, 0.0, 0.5), gymapi.Vec3(0.0, 0.0, 0.5))
            self.camera_handle = camera_handle

    def create_sim(self):
        raise NotImplementedError

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
    
    def get_camera_image(self):
        # camera_position = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1] + 1.0, self.root_states[0, 2])
        # camera_target = gymapi.Vec3(self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2])
        # self.gym.set_camera_location(self.camera_handle, self.envs[0], camera_position, camera_target)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        # self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR, "test.png")
        image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR)
        image = image.reshape((image.shape[0], -1, 4))
        return image
    
    def close(self):
        # cleanup
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

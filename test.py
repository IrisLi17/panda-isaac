from panda_isaac.panda_push import PandaPushEnv
# from panda_isaac.panda_push_joint import PandaPushEnv
from panda_isaac.base_config import BaseConfig
import torch
from PIL import Image
import numpy as np


class TestConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 1
        num_observations = 1 * 3 * 84 * 84 + 15
        num_state_obs = 18
        # num_observations = 3 + 15
        num_actions = 4
        max_episode_length = 100
    
    class asset(BaseConfig.asset):
        robot_urdf = "urdf/franka_description/robots/franka_panda.urdf"
    
    class cam(BaseConfig.cam):
        # view = "ego"
        view = "third"
        fov = 86
        w = 149
        h = 84
        loc_p = [0.11104 - 0.0592106 - 0.01, -0.0156, 0.015]
    
    class obs(BaseConfig.obs):
        # type = "state"
        type = "pixel"
        im_size = 84
        history_length = 1
        state_history_length = 1
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "ik"
        # controller = "cartesian_impedance"
        # controller = "osc"
        damping = 0.05
        kp_translation = 300 # 0-400
        kp_rotation = 10  # 0-30
        kd_translation = 2.0 * np.sqrt(kp_translation)
        kd_rotation = 2.0 * np.sqrt(kp_rotation)
        kp_null = 0.5
        kd_null = 2.0 * np.sqrt(kp_null)
    
    class reward(BaseConfig.reward):
        type = "dense"

    class safety(BaseConfig.safety):
        brake_on_contact = True
        contact_force_th = 1.0
    
    class domain_randomization(BaseConfig.domain_randomization):
        friction_range = [0.5, 3.0]

class ManualController():
    def __init__(self, env):
        self.env = env
        assert self.env.num_envs == 1
        self.phase = 0
        self.device = env.device
    
    def reset(self):
        self.phase = 0
    
    def act(self):
        hand_pos = self.env.rb_states[self.env.hand_idxs, :3]
        box_pos = self.env.rb_states[self.env.box_idxs, :3]
        # box_pos = torch.tensor([[0.5, 0.5, 0.425]], device=self.device)
        action = torch.zeros((1, 4), dtype=torch.float, device=self.device)
        if self.phase == 0:
            dpos = box_pos + torch.tensor([[0, 0, 0.2034]], dtype=torch.float, device=self.device) - hand_pos
            action[:, :3] = 20 * dpos
            action[:, 3] = 1.0
            if torch.norm(dpos) < 1e-2:
                self.phase = 1
        elif self.phase == 1:
            dpos = box_pos + torch.tensor([0, 0, 0.1034], dtype=torch.float, device=self.device) - hand_pos
            action[:, :3] = 20 * dpos
            action[:, 3] = 1.0
            if torch.norm(dpos) < 1.5e-2:
                self.phase = 2
        elif self.phase == 2:
            action[:, :3] = 0
            action[:, 3] = -1.0
            if torch.all(self.env.dof_pos[0, 7:9, 0] < 0.028):
                self.phase = 3
        elif self.phase == 3:
            dpos = self.env.box_goals - box_pos
            action[:, :3] = 20 * dpos
            action[:, 2] = 0.5
            action[:, 3] = -1.0
        return action

# cfg = TestJointConfig()
cfg = TestConfig()
env = PandaPushEnv(cfg, headless=True)
controller = ManualController(env)
obs = env.reset()
controller.reset()
for i in range(100):
    action = 2 * torch.rand(size=(env.num_envs, env.num_actions), dtype=torch.float, device=env.device) - 1
    action = controller.act()
    # action = 20 * (env.rb_states[env.box_idxs, :3] + torch.tensor([[0., 0., 0.15]], device=env.device) - env.rb_states[env.hand_idxs, :3])
    # action = 10 * (torch.from_numpy(np.array([[0.4, 0.0, 0.7]])).float().to(env.device).repeat(env.num_envs, 1) - env.rb_states[env.hand_idxs, :3])
    # action = torch.cat([action, torch.zeros(env.num_envs, 1, dtype=torch.float, device=env.device)], dim=-1)
    # print(i, action[0])
    if hasattr(env, "camera_handle"):
        image = env.get_camera_image()
        image = Image.fromarray(image.astype(np.uint8))
        filename = "tmp/tmp%d.png" % i
        image.save(filename)
    if env.cfg.obs.type == "pixel":
        obs_image = obs[0, :3 * env.cfg.obs.im_size * env.cfg.obs.im_size].reshape((3, env.cfg.obs.im_size, env.cfg.obs.im_size))
        obs_image = (obs_image * env.im_std + env.im_mean).permute(1, 2, 0) * 255
        obs_image = Image.fromarray(obs_image.cpu().numpy().astype(np.uint8))
        filename = "tmp/tmpobs%d.png" % i
        obs_image.save(filename)
    else:
        pass
        # print(obs[0])
    print(i)
    obs, reward, done, info = env.step(action)
    print(obs[0][-15:])
    # print(reward[0])
    if done[0]:
        print("reset", obs[0])
        controller.reset()
env.close()

'''
ik, decimal 12
0 tensor([-0.3472,  0.9935, -0.8587, -0.4214], device='cuda:0')
eef error tensor(0.0097, device='cuda:0')
1 tensor([-0.3703, -0.3859,  0.2242, -0.5491], device='cuda:0')
eef error tensor(0.0043, device='cuda:0')
2 tensor([-0.5490,  0.1671, -0.1273,  0.1354], device='cuda:0')
eef error tensor(0.0047, device='cuda:0')
3 tensor([-0.2833, -0.4026, -0.3414, -0.2061], device='cuda:0')
eef error tensor(0.0049, device='cuda:0')
4 tensor([ 0.0960,  0.3570, -0.2139,  0.1990], device='cuda:0')
eef error tensor(0.0032, device='cuda:0')
5 tensor([-0.4673, -0.8489, -0.4131,  0.0050], device='cuda:0')
eef error tensor(0.0084, device='cuda:0')
6 tensor([ 0.0493, -0.2970, -0.7032,  0.2535], device='cuda:0')
eef error tensor(0.0055, device='cuda:0')
7 tensor([-0.3454,  0.1291, -0.5394, -0.4445], device='cuda:0')
eef error tensor(0.0048, device='cuda:0')
8 tensor([-0.2387,  0.5743,  0.5964,  0.7755], device='cuda:0')
eef error tensor(0.0063, device='cuda:0')
9 tensor([-0.5607, -0.8752,  0.4987,  0.0632], device='cuda:0')
eef error tensor(0.0086, device='cuda:0')

ik, decimal 6
0 tensor([-0.3472,  0.9935, -0.8587, -0.4214], device='cuda:0')
eef error tensor(0.0348, device='cuda:0')
1 tensor([-0.3703, -0.3859,  0.2242, -0.5491], device='cuda:0')
eef error tensor(0.0194, device='cuda:0')
2 tensor([-0.5490,  0.1671, -0.1273,  0.1354], device='cuda:0')
eef error tensor(0.0148, device='cuda:0')
3 tensor([-0.2833, -0.4026, -0.3414, -0.2061], device='cuda:0')
eef error tensor(0.0156, device='cuda:0')
4 tensor([ 0.0960,  0.3570, -0.2139,  0.1990], device='cuda:0')
eef error tensor(0.0119, device='cuda:0')
5 tensor([-0.4673, -0.8489, -0.4131,  0.0050], device='cuda:0')
eef error tensor(0.0260, device='cuda:0')
6 tensor([ 0.0493, -0.2970, -0.7032,  0.2535], device='cuda:0')
eef error tensor(0.0201, device='cuda:0')
7 tensor([-0.3454,  0.1291, -0.5394, -0.4445], device='cuda:0')
eef error tensor(0.0132, device='cuda:0')
8 tensor([-0.2387,  0.5743,  0.5964,  0.7755], device='cuda:0')
eef error tensor(0.0225, device='cuda:0')
9 tensor([-0.5607, -0.8752,  0.4987,  0.0632], device='cuda:0')
eef error tensor(0.0278, device='cuda:0')
'''

'''
osc
0 tensor([-0.3472,  0.9935, -0.8587, -0.4214], device='cuda:0')
eef error tensor(0.0210, device='cuda:0')
1 tensor([-0.3703, -0.3859,  0.2242, -0.5491], device='cuda:0')
eef error tensor(0.0102, device='cuda:0')
2 tensor([-0.5490,  0.1671, -0.1273,  0.1354], device='cuda:0')
eef error tensor(0.0090, device='cuda:0')
3 tensor([-0.2833, -0.4026, -0.3414, -0.2061], device='cuda:0')
eef error tensor(0.0090, device='cuda:0')
4 tensor([ 0.0960,  0.3570, -0.2139,  0.1990], device='cuda:0')
eef error tensor(0.0070, device='cuda:0')
5 tensor([-0.4673, -0.8489, -0.4131,  0.0050], device='cuda:0')
eef error tensor(0.0167, device='cuda:0')
6 tensor([ 0.0493, -0.2970, -0.7032,  0.2535], device='cuda:0')
eef error tensor(0.0110, device='cuda:0')
7 tensor([-0.3454,  0.1291, -0.5394, -0.4445], device='cuda:0')
eef error tensor(0.0095, device='cuda:0')
8 tensor([-0.2387,  0.5743,  0.5964,  0.7755], device='cuda:0')
eef error tensor(0.0137, device='cuda:0')
9 tensor([-0.5607, -0.8752,  0.4987,  0.0632], device='cuda:0')
eef error tensor(0.0179, device='cuda:0')
'''
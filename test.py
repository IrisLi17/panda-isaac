# from panda_isaac.panda_push import PandaPushEnv
from panda_isaac.panda_push_joint import PandaPushEnv
from panda_isaac.base_config import BaseConfig
import torch
from PIL import Image
import numpy as np


class TestConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 1
        # num_observations = 3 * 224 * 224 + 12
        num_observations = 7 + 15
        num_actions = 4
        max_episode_length = 50
    
    class obs(BaseConfig.obs):
        type = "state"
        # type = "pixel"
        im_size = 224
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "ik"
        damping = 0.05
    
    class reward(BaseConfig.reward):
        type = "dense"


class TestJointConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 1
        num_observations = 7 + 27
        # num_observations = 3 * 224 * 224 + 27
        num_actions = 8
        max_episode_length = 50
    
    class obs(BaseConfig.obs):
        type = "state"
        # type = "pixel"
        im_size = 224
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "joint"
        # common_speed = 2.17 * decimal
    
    class reward(BaseConfig.reward):
        type = "sparse"

cfg = TestJointConfig()
# cfg = TestConfig()
env = PandaPushEnv(cfg, headless=True)
obs = env.reset()
for i in range(100):
    action = 2 * torch.rand(size=(env.num_envs, env.num_actions), dtype=torch.float, device=env.device) - 1
    # action = 20 * (env.rb_states[env.box_idxs, :3] + torch.tensor([[0., 0., 0.15]], device=env.device) - env.rb_states[env.hand_idxs, :3])
    # action = 10 * (torch.from_numpy(np.array([[0.4, 0.0, 0.7]])).float().to(env.device).repeat(env.num_envs, 1) - env.rb_states[env.hand_idxs, :3])
    # action = torch.cat([action, torch.zeros(env.num_envs, 1, dtype=torch.float, device=env.device)], dim=-1)
    # print(i, action[0])
    image = env.get_camera_image()
    image = Image.fromarray(image.astype(np.uint8))
    filename = "tmp/tmp%d.png" % i
    image.save(filename)
    if env.cfg.obs.type == "pixel":
        obs_image = obs[0, :3 * 224 * 224].reshape((3, 224, 224))
        obs_image = (obs_image * env.im_std + env.im_mean).permute(1, 2, 0) * 255
        obs_image = Image.fromarray(obs_image.cpu().numpy().astype(np.uint8))
        filename = "tmp/tmpobs%d.png" % i
        obs_image.save(filename)
    else:
        pass
        # print(obs[0])
    obs, reward, done, info = env.step(action)
    # print(reward[0])
    if done[0]:
        print("reset", obs[0])
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
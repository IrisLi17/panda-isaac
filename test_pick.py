from panda_isaac.panda_pick import PandaPickEnv
from panda_isaac.base_config import BaseConfig
import torch
from PIL import Image
import numpy as np


class TestConfig(BaseConfig):
  class env(BaseConfig.env):
    seed = 42
    num_envs = 1
    num_observations = 1 * 3 * 84 * 84 + 30
    num_state_obs = 27+3
    # num_observations = 3 + 15
    num_actions = 8
    max_episode_length = 100

  class asset(BaseConfig.asset):
    robot_urdf = "urdf/franka_description/robots/franka_panda_cam.urdf"

  class cam(BaseConfig.cam):
    view = "ego"
    fov = 86
    w = 149
    h = 84

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
    kp_translation = 300  # 0-400
    kp_rotation = 10  # 0-30
    kd_translation = 2.0 * np.sqrt(kp_translation)
    kd_rotation = 2.0 * np.sqrt(kp_rotation)
    kp_null = 0.5
    kd_null = 2.0 * np.sqrt(kp_null)

  class reward(BaseConfig.reward):
    type = "dense"

  class safety(BaseConfig.safety):
    brake_on_contact = False
    contact_force_th = 1.0

  class domain_randomization(BaseConfig.domain_randomization):
    friction_range = [0.5, 3.0]


class ManualController():
  def __init__(self, env):
    self.env = env
    assert self.env.num_envs == 1
    self.phase = 0
    self.steps = 0
    self.ep_steps = np.cumsum([5,30,5,8,70,5,5,30])
    self.device = env.device

  def reset(self):
    self.phase = 0
    self.steps = 0

  def act(self):
    hand_pos = self.env.rb_states[self.env.hand_idxs, :3]
    box_pos = self.env.rb_states[self.env.box_idxs, :3]
    action = torch.zeros((2, 4), dtype=torch.float, device=self.device)*2-1
    if self.steps < self.ep_steps[0]:
      action[:, 2] = 1
    elif self.steps < self.ep_steps[1]:
      dpos = box_pos + \
        torch.tensor([[0, -0.06, 0.08]], dtype=torch.float,
                     device=self.device) - hand_pos
      action[:, :3] = dpos/torch.norm(dpos, dim=-1, keepdim=True)
      action[:, 3] = 1.0
    elif self.steps < self.ep_steps[2]:
      action[:, :3] = 0
      action[0, 3] = -1.0
    elif self.steps < self.ep_steps[3]:
      action[:, 2] = 1
    elif self.steps < self.ep_steps[4]:
      dpos0 = -(box_pos - self.env.box_goals) 
      action[0, :3] = dpos0/torch.norm(dpos0)
      action[0,3]=-1
      dpos1 = (box_pos + torch.tensor([[0, 0.06, 0.08]], dtype=torch.float,
                     device=self.device) - hand_pos[1]) 
      action[1, :3] = dpos1/torch.norm(dpos1)
      action[1,3]=1
    self.steps += 1
    return action.view(self.env.num_envs, -1)


# cfg = TestJointConfig()
cfg = TestConfig()
env = PandaPickEnv(cfg, headless=False)
controller = ManualController(env)
obs = env.reset()
controller.reset()
for i in range(10000):
  action = 2 * torch.rand(size=(env.num_envs, env.num_actions),
                          dtype=torch.float, device=env.device) - 1
  obs, reward, done, info = env.step(action)
env.close()
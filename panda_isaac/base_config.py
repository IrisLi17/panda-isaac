import numpy as np


class BaseConfig(object):
    class env:
        seed: int
        num_envs: int
        num_observations: int
        num_actions: int
        num_state_obs: int
        max_episode_length: int
    
    class asset:
        robot_urdf = "urdf/franka_description/robots/franka_panda.urdf"
    
    class cam:
        view = "ego"
        crop = "center"
        w = 398
        h = 224
        # fov = 120
        fov = 86
        ss = 2
        loc_p = [0.11104 - 0.0592106, -0.0156, 0.0595]
        loc_r = [180, -90.0, 180.0]

    class obs:
        type: str
        im_size: int
        history_length: int
        state_history_length: int
        noise = False

    class control:
        decimal: int
        controller: str  # "ik" or "osc"
        damping = 0.05

        filter_param = 0.2
        # OSC params
        kp = 150.
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.
        kd_null = 2.0 * np.sqrt(kp_null)

        # Cartesian impedance params
        kp_translation = 400 # 0-400
        kp_rotation = 30  # 0-30
        kd_translation = 2.0 * np.sqrt(kp_translation)
        kd_rotation = 2.0 * np.sqrt(kp_rotation)
        kp_null = 0.5
        kd_null = 2.0 * np.sqrt(kp_null)

        # Joint
        common_speed: float
    
    class reward:
        type: str
        contact_coef = 0.0
    
    class safety:
        brake_on_contact = False
        contact_force_th = 1
    
    class domain_randomization:
        friction_range = [1.0, 1.0]

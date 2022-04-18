import numpy as np


class BaseConfig(object):
    class env:
        seed: int
        num_envs: int
        num_observations: int
        num_actions: int
        max_episode_length: int
    
    class cam:
        view = "ego"
        crop = "center"
        w = 298
        h = 224
        fov = 120
        # fov = 86
        ss = 2
        loc_p = [0.04, 0.0, 0.045]
        loc_r = [180, -90.0, 0.0]

    class obs:
        type: str
        im_size: int

    class control:
        decimal: int
        controller: str  # "ik" or "osc"
        damping = 0.05

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
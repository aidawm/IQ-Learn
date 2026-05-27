"""CartPoleContinuous-v0

Sign-based action adapter on top of stock CartPole-v1. Continuous action
in [-1, 1] is mapped to discrete {0, 1} via int(a > 0). The underlying
physics, reward (+1 per step), and termination (pole angle > 12 deg,
|x| > 2.4, or 500 steps) are unchanged from CartPole-v1.

Needed because the IQ-Learn SAC backbone requires a continuous action
space; the same adapter is used in several SAC-on-discrete-control
papers (e.g. SAC-discrete baselines).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CartPoleContinuousWrapper(gym.ActionWrapper):
    """Wrap CartPole-v1 so it accepts a continuous Box([-1], [1]) action.

    The wrapper exposes a single continuous action a in [-1, 1] and
    internally maps it to the discrete action {0, 1} via int(a > 0).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        a = np.asarray(action).flatten()
        return int(a[0] > 0)


def make_cartpole_continuous(**kwargs):
    base = gym.make("CartPole-v1", **kwargs)
    return CartPoleContinuousWrapper(base)

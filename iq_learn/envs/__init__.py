from gymnasium.envs.registration import register
import gymnasium as gym


def _register_if_absent(env_id, **kwargs):
    if env_id not in gym.envs.registry:
        register(id=env_id, **kwargs)


def register_custom_envs():
    _register_if_absent(
        'PointMazeRight-v0',
        entry_point='envs.point_maze_env:PointMazeEnv',
        kwargs={'sparse_reward': False, 'direction': 1, 'discrete': True},
        max_episode_steps=100,
    )
    _register_if_absent(
        'PointMazeLeft-v0',
        entry_point='envs.point_maze_env:PointMazeEnv',
        kwargs={'sparse_reward': False, 'direction': 0, 'discrete': True},
        max_episode_steps=100,
    )
    _register_if_absent(
        'PointMazeRightCont-v0',
        entry_point='envs.point_maze_env:PointMazeEnv',
        kwargs={'sparse_reward': False, 'direction': 1, 'discrete': False},
        max_episode_steps=100,
    )
    _register_if_absent(
        'PointMazeLeftCont-v0',
        entry_point='envs.point_maze_env:PointMazeEnv',
        kwargs={'sparse_reward': False, 'direction': 0, 'discrete': False},
        max_episode_steps=100,
    )
    # Continuous-action adapter on top of stock CartPole-v1.
    # See envs/cartpole_continuous.py for the sign-based action mapping.
    _register_if_absent(
        'CartPoleContinuous-v0',
        entry_point='envs.cartpole_continuous:make_cartpole_continuous',
        max_episode_steps=500,
    )

"""Train a SAC expert on CartPoleContinuous-v0 using Stable-Baselines3.

Saves the trained SAC model to:
    iq_learn/trained_policies/sac_CartPoleContinuous-v0.zip

Usage (from iq_learn/):
    python expert_train_sac.py --total-steps 50000 --seed 1

We use SB3 SAC instead of the in-tree agent.SAC because the in-tree
train_rl.py loop uses the old gym (4-tuple) API while train_iq.py uses
gymnasium (5-tuple). SB3 SAC also gives reliable CartPole results in
a couple of minutes.
"""

import argparse
import os

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

import envs  # registers CartPoleContinuous-v0
envs.register_custom_envs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="CartPoleContinuous-v0")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-dir", default="trained_policies")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"sac_{args.env_name}")

    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    model.learn(total_timesteps=args.total_steps, log_interval=10)
    model.save(save_path)
    print(f"--> Saved expert model to: {save_path}.zip")

    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Eval over {args.eval_episodes} episodes: "
          f"{mean_r:.2f} +/- {std_r:.2f}")


if __name__ == "__main__":
    main()

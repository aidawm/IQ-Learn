"""Train a SAC expert on CartPoleContinuous-v0 / Pendulum-v1 using Stable-Baselines3.
Saves the trained SAC model to:
    iq_learn/trained_policies/sac_<env-name>.zip
Usage (from iq_learn/):
    python expert_train_sac.py --env-name Pendulum-v1 --total-steps 100000 --seed 42
"""
import argparse
import os
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import envs  # registers CartPoleContinuous-v0
envs.register_custom_envs()


# ── optional progress callback ────────────────────────────────────────────────
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5_000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_r, std_r = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True, warn=False,
            )
            if self.verbose:
                print(f"  step {self.num_timesteps:>8,} | return {mean_r:8.2f} ± {std_r:.2f}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name",       default="CartPoleContinuous-v0")
    parser.add_argument("--total-steps",    type=int,   default=100_000)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--save-dir",       default="trained_policies")
    parser.add_argument("--eval-episodes",  type=int,   default=10)
    parser.add_argument("--device",         default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"sac_{args.env_name}")

    env      = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,        # was 50_000  → more experience retained
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",            # automatic entropy tuning
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),  # was [64,64] → larger network
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    cb = EvalCallback(eval_env, eval_freq=5_000, n_eval_episodes=10)
    model.learn(total_timesteps=args.total_steps, callback=cb, log_interval=10)
    model.save(save_path)
    print(f"--> Saved expert model to: {save_path}.zip")

    mean_r, std_r = evaluate_policy(
        model, eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Eval over {args.eval_episodes} episodes: {mean_r:.2f} +/- {std_r:.2f}")


if __name__ == "__main__":
    main()
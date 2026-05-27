"""Collect expert demonstrations on CartPoleContinuous-v0 from a trained
Stable-Baselines3 SAC policy and save them in the .pkl format expected by
IQ-Learn's dataset/expert_dataset.py.

Saved file: iq_learn/experts/CartPoleContinuous-v0_K{K}.pkl
Pickled dict with keys: states, next_states, actions, rewards, dones, lengths.

Usage (from iq_learn/):
    python expert_collect_sac.py --K 20 --seed 100
    # then point env.demo at it via env.demo=CartPoleContinuous-v0_K20

This script also writes multiple K sizes in one go if --K-list is passed:
    python expert_collect_sac.py --K-list 1 5 10 20
The largest K is rolled out and then truncated to produce the smaller K files,
so each smaller dataset is exactly a prefix of the larger one (clean
demo-size sweep).
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

import envs  # registers CartPoleContinuous-v0
envs.register_custom_envs()


def collect(model, env, n_episodes, seed, max_steps, threshold):
    """Roll out `n_episodes` accepted trajectories.

    An episode is accepted only if its total reward >= threshold (None
    means accept everything).
    """
    trajs = defaultdict(list)
    accepted = 0
    attempted = 0
    while accepted < n_episodes:
        obs, _ = env.reset(seed=seed + attempted)
        attempted += 1
        traj = []
        ep_ret = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).flatten()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            traj.append((
                np.asarray(obs, dtype=np.float32),
                np.asarray(next_obs, dtype=np.float32),
                action,
                float(reward),
                done,
            ))
            ep_ret += reward
            obs = next_obs
            if done:
                break

        if threshold is None or ep_ret >= threshold:
            states, next_states, actions, rewards, dones = zip(*traj)
            trajs["states"].append(list(states))
            trajs["next_states"].append(list(next_states))
            trajs["actions"].append(list(actions))
            trajs["rewards"].append(list(rewards))
            trajs["dones"].append(list(dones))
            trajs["lengths"].append(len(traj))
            accepted += 1
            print(f"  accepted ep {accepted}/{n_episodes} "
                  f"len={len(traj)} return={ep_ret:.1f}")
        else:
            print(f"  skipped ep (return={ep_ret:.1f} < {threshold})")

    rewards = np.array([sum(r) for r in trajs["rewards"]])
    lengths = np.array(trajs["lengths"])
    print(f"\nCollected {accepted} eps  (attempts={attempted})")
    print(f"  return: {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"  length: {lengths.mean():.2f} +/- {lengths.std():.2f}")
    return trajs


def truncate(trajs, K):
    out = {}
    for k, v in trajs.items():
        out[k] = list(v[:K])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="CartPoleContinuous-v0")
    parser.add_argument("--model-path",
                        default="trained_policies/sac_CartPoleContinuous-v0")
    parser.add_argument("--out-dir", default="experts")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=475.0,
                        help="minimum episode return to accept "
                             "(None to disable)")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--K", type=int, default=None,
                        help="single K (collect exactly K episodes)")
    parser.add_argument("--K-list", type=int, nargs="+", default=None,
                        help="multiple K sizes; the largest K is rolled "
                             "out and the smaller files are prefixes")
    args = parser.parse_args()

    if args.K is None and args.K_list is None:
        parser.error("specify either --K or --K-list")

    os.makedirs(args.out_dir, exist_ok=True)

    env = gym.make(args.env_name)
    model = SAC.load(args.model_path, env=env)

    if args.K_list is not None:
        Ks = sorted(set(args.K_list))
        K_max = max(Ks)
        print(f"\nRolling out {K_max} episodes (threshold={args.threshold})")
        trajs = collect(model, env, K_max, args.seed,
                        args.max_steps, args.threshold)
        for K in Ks:
            sub = truncate(trajs, K)
            out_path = os.path.join(args.out_dir,
                                    f"{args.env_name}_K{K}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(sub, f)
            print(f"--> wrote {out_path}  (K={K})")
    else:
        print(f"\nRolling out {args.K} episodes (threshold={args.threshold})")
        trajs = collect(model, env, args.K, args.seed,
                        args.max_steps, args.threshold)
        out_path = os.path.join(args.out_dir,
                                f"{args.env_name}_K{args.K}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(trajs, f)
        print(f"--> wrote {out_path}")


if __name__ == "__main__":
    main()

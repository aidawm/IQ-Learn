"""
Parse IQ-Learn stdout logs and write progress.csv files in the same format
as SOAR-IL, so both algorithms can be compared with the same plotting code.

Output structure mirrors SOAR-IL:
  logs/<env>/exp-{K}/iq_learn/seed{seed}/progress.csv

Progress.csv columns (matching SOAR-IL):
  Real Det Return, Running Env Steps, Running Reverse KL, Running Forward KL,
  Real Sto Return, Itration, Reward Loss, Running Update Time

Usage:
  python generate_progress_csv.py                  # Pendulum (default)
  python generate_progress_csv.py cartpole
  python generate_progress_csv.py pendulum
"""

import os
import re
import csv
import sys

CONFIGS = {
    "pendulum": {
        "logs_dir": "outputs/pendulum_iq",
        "env_name": "Pendulum-v1",
    },
    "cartpole": {
        "logs_dir": "outputs/cartpole_iq",
        "env_name": "CartPoleContinuous-v0",
    },
}

env_key = sys.argv[1].lower() if len(sys.argv) > 1 else "pendulum"
if env_key not in CONFIGS:
    print(f"Unknown env '{env_key}'. Choose from: {list(CONFIGS)}")
    sys.exit(1)

IQ_LOGS_DIR = CONFIGS[env_key]["logs_dir"]
OUT_BASE = "logs"
ENV_NAME = CONFIGS[env_key]["env_name"]
ALG_NAME = "iq_learn"

EVAL_RE = re.compile(
    r"\| eval\s+\| E:\s*(\d+)\s+\| S:\s*(\d+)\s+\| R:\s*([-\d.]+)"
)
LOG_RE = re.compile(r"demos_(\d+)_seed_(\d+)_gpu_\d+\.log")

FIELDNAMES = [
    "Real Det Return",
    "Running Env Steps",
    "Running Reverse KL",
    "Running Forward KL",
    "Real Sto Return",
    "Itration",
    "Reward Loss",
    "Running Update Time",
]


def parse_log(log_path):
    rows = []
    with open(log_path) as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                episode, step, reward = int(m.group(1)), int(m.group(2)), float(m.group(3))
                rows.append((episode, step, reward))
    return rows


def write_progress_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for itration, (episode, step, reward) in enumerate(rows):
            writer.writerow({
                "Real Det Return": reward,
                "Running Env Steps": step,
                "Running Reverse KL": 0.0,
                "Running Forward KL": 0.0,
                "Real Sto Return": reward,
                "Itration": itration,
                "Reward Loss": 0.0,
                "Running Update Time": 0,
            })


def main():
    log_dir = IQ_LOGS_DIR
    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    for fname in sorted(os.listdir(log_dir)):
        m = LOG_RE.match(fname)
        if not m:
            continue
        k, seed = int(m.group(1)), int(m.group(2))
        log_path = os.path.join(log_dir, fname)

        rows = parse_log(log_path)
        if not rows:
            print(f"No eval rows found in {fname}, skipping.")
            continue

        out_path = os.path.join(
            OUT_BASE, ENV_NAME, f"exp-{k}", ALG_NAME, f"seed{seed}", "progress.csv"
        )
        write_progress_csv(rows, out_path)
        print(f"K={k:2d}  seed={seed}  rows={len(rows):3d}  -> {out_path}")


if __name__ == "__main__":
    main()

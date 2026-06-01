# Inverse Q-Learning (IQ-Learn)

## SOTA framework for non-adversarial Imitation Learning

IQ-Learn enables very fast, scalable and stable imitation learning.
Our IQ-Learn algorithm is present in `iq.py`. This file can be used standalone to add **IQ** to your IL & RL projects.

IQ-Learn can be implemented on top of most existing RL methods (off-policy & on-policy) by changing the critic update loss to our proposed `iq_loss`. <br>
(IQ has been successfully tested to work with Q-Learning, SAC, PPO, DDPG and Decision Transformer agents).

### our contributions 
- Benchmarked IQ-Learn against SOAR-IL on continuous control tasks (Pendulum-v1, CartPoleContinuous-v0) across varying demo sizes K=1,5,10,20,50
- Added Pendulum-v1 sweep: SAC expert trained with Stable-Baselines3, demos collected at K=1,5,10,20,50
- Added CartPoleContinuous-v0 sweep with the same K values
- Added `envs/cartpole_continuous.py` implementing the CartPoleContinuous-v0 custom Gymnasium environment
- Added `expert_train_sac.py` to train SAC experts via SB3 for new continuous environments
- Added `expert_collect_sac.py` to roll out and save expert demos in the `.pkl` format IQ-Learn expects
- Added `scripts/run_pendulum.sh` and `scripts/run_cartpole.sh` for parallel multi-GPU sweeps over demo sizes and seeds
- Added `generate_progress_csv.py` to convert IQ-Learn stdout logs into the SOAR-IL `progress.csv` format for side-by-side comparison
- Released `expert_generation` script to generate experts from trained RL agents for new environments
- Updated `requirements.txt` to resolve dependency conflicts in the original repo

## Requirements

- pytorch (>= 1.4)
- gym / gymnasium
- stable-baselines3 (for SAC expert training)
- wandb
- tensorboardX
- hydra-core=1.0 (>= 1.1 is incompatible currently)

## Installation

```bash
pip install -r requirements.txt
```


## Workflow: from expert to IQ-Learn results

### Step 1 — Train a SAC expert

```bash
# Pendulum-v1
python expert_train_sac.py --env-name Pendulum-v1 --total-steps 100000

# CartPoleContinuous-v0
python expert_train_sac.py --env-name CartPoleContinuous-v0 --total-steps 100000
```

Saves a `.zip` policy to `trained_policies/`.

### Step 2 — Collect expert demonstrations

```bash
# Collect K=1,5,10,20,50 demos in one shot
python expert_collect_sac.py --env-name Pendulum-v1 --K-list 1 5 10 20 50
python expert_collect_sac.py --env-name CartPoleContinuous-v0 --K-list 1 5 10 20 50
```

Saves `.pkl` files to `experts/<env>_K<K>.pkl`. Each smaller K is a clean prefix of the larger one.

### Step 3 — Run IQ-Learn sweeps

```bash
# Pendulum: 5 demo sizes × 3 seeds, distributed across available GPUs
bash scripts/run_pendulum.sh

# CartPoleContinuous: same sweep
bash scripts/run_cartpole.sh
```

Logs land in `outputs/<env>_iq/demos_<K>_seed_<seed>_gpu_<gpu>.log`.

### Step 4 — Generate comparison CSVs

```bash
python generate_progress_csv.py pendulum
python generate_progress_csv.py cartpole
```

Outputs `logs/<env>/exp-<K>/iq_learn/seed<seed>/progress.csv` in the same format as SOAR-IL, so both algorithms can be plotted with the same code.

## Classic examples

### CartPole-v1 — offline IL with 1 demo

```bash
python train_iq.py agent=softq method=iq env=cartpole \
  expert.demos=1 expert.subsample_freq=20 \
  agent.init_temp=0.001 method.chi=True method.loss=value_expert
```

<img src="../docs/cartpole_example.png" width="500">

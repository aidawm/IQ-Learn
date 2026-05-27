#!/bin/bash
# Run all CartPole Continuous SAC + IQ experiments
# Based on: python train_iq.py agent=sac env=cartpole_continuous
#   method.chi=True method.loss=value_expert method.regularize=False
#   agent.actor_lr=3e-4 agent.init_temp=1e-3
#   hydra.run.dir=. hydra.output_subdir=null

DEMOS=(1 5 10 20 50)
SEEDS=(1 2 3)
GPUS=(0 1 2 3 4 5 6 7)

LOG_DIR="outputs/cartpole_iq"
mkdir -p "$LOG_DIR"

job_id=0
pids=()

echo "========================================"
echo " CartPole Continuous IQ Experiment Runner"
echo " GPUs used: ${GPUS[*]}"
echo " Total jobs: $((${#DEMOS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""

for DEMO in "${DEMOS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    GPU=${GPUS[$((job_id % ${#GPUS[@]}))]}
    EXP_NAME="K${DEMO}_seed${SEED}"
    LOG_FILE="$LOG_DIR/demos_${DEMO}_seed_${SEED}_gpu_${GPU}.log"

    echo "Launching job $job_id | demos=$DEMO seed=$SEED gpu=$GPU exp=$EXP_NAME"

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u train_iq.py \
      agent=sac \
      env=cartpole_continuous \
      env.demo=CartPoleContinuous-v0_K${DEMO} \
      expert.demos=$DEMO \
      method.chi=True \
      method.loss=value_expert \
      method.regularize=False \
      agent.actor_lr=3e-4 \
      agent.init_temp=1e-3 \
      seed=$SEED \
      exp_name=$EXP_NAME \
      hydra.run.dir=. \
      hydra.output_subdir=null \
      > "$LOG_FILE" 2>&1 &

    pids+=($!)
    job_id=$((job_id + 1))
  done
done

echo ""
echo "All $job_id jobs submitted."
echo "Waiting for all jobs to finish..."
echo ""

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "Job with PID $pid failed."
    failed=1
  fi
done

echo ""
if [ "$failed" -eq 0 ]; then
  echo "All experiments completed successfully."
else
  echo "Some experiments failed. Check logs in $LOG_DIR."
fi
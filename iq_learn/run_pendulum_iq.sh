#!/bin/bash

# Run all Pendulum SAC + IQ experiments
# Submit all jobs simultaneously
# GPUs used: 1-7 only, assigned round-robin

DEMOS=(1 5 10 20 50)
SEEDS=(1 2 3)
GPUS=(1 2 3 4 5 6 7)

# Best config from hparam sweep: batch=256, alpha=5.0, capped at 30k steps
BATCH=256
ALPHA=5.0

LOG_DIR="outputs/pendulum_iq"
mkdir -p "$LOG_DIR"

job_id=0
pids=()

echo "========================================"
echo " Pendulum IQ Experiment Runner"
echo " GPUs used: ${GPUS[*]}"
echo " Total jobs: $((${#DEMOS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""

for DEMO in "${DEMOS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    GPU=${GPUS[$((job_id % ${#GPUS[@]}))]}

    LOG_FILE="$LOG_DIR/demos_${DEMO}_seed_${SEED}_gpu_${GPU}.log"

    echo "Launching job $job_id | demos=$DEMO seed=$SEED gpu=$GPU"

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u train_iq.py \
      env=pendulum \
      agent=sac \
      method=iq \
      expert.demos=$DEMO \
      seed=$SEED \
      method.regularize=true \
      method.chi=true \
      method.alpha=$ALPHA \
      train.batch=$BATCH \
      env.learn_steps=30000 \
      > "$LOG_FILE" 2>&1 &

    pids+=($!)

    job_id=$((job_id + 1))

  done
done

echo ""
echo "All jobs submitted."
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
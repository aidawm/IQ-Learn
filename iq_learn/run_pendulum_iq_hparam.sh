#!/bin/bash

# Hyperparameter sweep for Pendulum IQ-Learn
# Varies train.batch and method.alpha relative to baseline (batch=32, alpha=0.5)
#   batch:  32 (1x) | 128 (4x) | 256 (8x)
#   alpha:  0.1 (0.2x) | 0.5 (1x) | 1.0 (2x) | 5.0 (10x)
# Fixed: demos=10, seeds=1-3
# Total jobs: 3 x 4 x 3 = 36

BATCHES=(32 128 256)
ALPHAS=(0.1 0.5 1.0 5.0)
SEEDS=(1)
GPUS=(1 2 3 4 5 6 7)

LOG_DIR="outputs/pendulum_iq_hparam"
mkdir -p "$LOG_DIR"

job_id=0
pids=()

echo "========================================"
echo " Pendulum IQ Hyperparameter Sweep"
echo " batch: ${BATCHES[*]}"
echo " alpha: ${ALPHAS[*]}"
echo " seeds: ${SEEDS[*]}"
echo " GPUs:  ${GPUS[*]}"
echo " Total jobs: $((${#BATCHES[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""

for BATCH in "${BATCHES[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      GPU=${GPUS[$((job_id % ${#GPUS[@]}))]}
      LOG_FILE="$LOG_DIR/batch_${BATCH}_alpha_${ALPHA}_seed_${SEED}_gpu_${GPU}.log"

      echo "Launching job $job_id | batch=$BATCH alpha=$ALPHA seed=$SEED gpu=$GPU"

      CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u train_iq.py \
        env=pendulum \
        agent=sac \
        method=iq \
        expert.demos=10 \
        seed=$SEED \
        method.regularize=true \
        method.chi=true \
        method.alpha=$ALPHA \
        train.batch=$BATCH \
        > "$LOG_FILE" 2>&1 &

      pids+=($!)
      job_id=$((job_id + 1))

    done
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

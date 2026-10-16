#!/bin/bash
# Train the point-cloud normal estimator used for Poisson surface extraction.
# Total batch = 8 GPUs x 24 shapes = 192 shapes.
# Requires `python representations/data/prepare_normal_data.py --data_root $DATA_ROOT`.
DATA_ROOT=${DATA_ROOT:-./data/objaverse}
OUTPUT_DIR=${OUTPUT_DIR:-exp/normal_estimator}
NUM_GPUS=${NUM_GPUS:-8}

accelerate launch --num_processes $NUM_GPUS train_normal_estimator.py \
  --data_root $DATA_ROOT \
  --train_splits train \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 24 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 40000 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine \
  --lr_warmup_steps 50 \
  --allow_tf32 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 10 \
  --resume_from_checkpoint latest \
  "$@"

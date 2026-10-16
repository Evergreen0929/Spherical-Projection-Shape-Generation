#!/bin/bash
# Stage 1: finetune the AutoEncoder on SP depth maps.
# Total batch = 8 GPUs x 2 objects x 4 layers = 64 SP maps.
# Extra arguments are forwarded to train_vae.py.
DATA_ROOT=${DATA_ROOT:-./data/objaverse}
OUTPUT_DIR=${OUTPUT_DIR:-exp/vae}
NUM_GPUS=${NUM_GPUS:-8}

accelerate launch --num_processes $NUM_GPUS --mixed_precision fp16 train_vae.py \
  --pretrained_vae madebyollin/sdxl-vae-fp16-fix \
  --data_root $DATA_ROOT \
  --train_splits train \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 40000 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 50 \
  --edge_loss_weight 0.1 \
  --freq_loss_weight 5e-4 \
  --kl_weight 1e-8 \
  --gradient_checkpointing \
  --allow_tf32 \
  --mixed_precision fp16 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 10 \
  --resume_from_checkpoint latest \
  "$@"

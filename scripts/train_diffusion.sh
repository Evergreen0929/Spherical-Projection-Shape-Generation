#!/bin/bash
# Stage 3: finetune the SDXL UNet with Layer-wise Self-Attention on SP map latents.
# Total batch = 8 GPUs x 5 objects x 2 accumulation steps = 80 objects.
DATA_ROOT=${DATA_ROOT:-./data/objaverse}
VAE_PATH=${VAE_PATH:-exp/vae/pretrained-40000/vae}
OUTPUT_DIR=${OUTPUT_DIR:-exp/diffusion}
NUM_GPUS=${NUM_GPUS:-8}
# reduces memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

accelerate launch --num_processes $NUM_GPUS --mixed_precision fp16 train_diffusion.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --vae_path $VAE_PATH \
  --image_encoder facebook/dinov2-large \
  --data_root $DATA_ROOT \
  --train_splits train \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 5 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 80000 \
  --learning_rate 1e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 1000 \
  --noise_offset 0.05 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --allow_tf32 \
  --mixed_precision fp16 \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 2 \
  --resume_from_checkpoint latest \
  "$@"

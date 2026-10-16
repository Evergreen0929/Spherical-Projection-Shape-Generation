#!/bin/bash
# Stage 2: encode the SP depth maps with the finetuned AutoEncoder.
DATA_ROOT=${DATA_ROOT:-./data/objaverse}
VAE_PATH=${VAE_PATH:-exp/vae/pretrained-40000/vae}
NUM_GPUS=${NUM_GPUS:-8}

accelerate launch --num_processes $NUM_GPUS encode_latents.py \
  --vae_path $VAE_PATH \
  --data_root $DATA_ROOT \
  --splits train \
  "$@"

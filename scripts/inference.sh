#!/bin/bash
# Single image to 3D shape.
# The AutoEncoder and the normal estimator default to the released weights:
#   huggingface-cli download jdzhang0929/SPGen-VAE --local-dir weights/SPGen-VAE
INPUT=${INPUT:?set INPUT to an image or a folder of images}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
UNET_PATH=${UNET_PATH:-exp/diffusion/checkpoint-80000}
VAE_PATH=${VAE_PATH:-weights/SPGen-VAE/vae}
NORMAL_ESTIMATOR_PATH=${NORMAL_ESTIMATOR_PATH:-weights/SPGen-VAE/normal_estimator}

python inference.py \
  --input $INPUT \
  --output_dir $OUTPUT_DIR \
  --unet_path $UNET_PATH \
  --vae_path $VAE_PATH \
  --normal_estimator_path $NORMAL_ESTIMATOR_PATH \
  --steps 40 \
  --guidance_scale 4.5 \
  "$@"

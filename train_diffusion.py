"""Stage 3: finetune the SDXL UNet with Layer-wise Self-Attention to generate multi-layer SP map latents
from a single image."""
import argparse
import json
import logging
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from spgen.data import SPLatentDataset
from spgen.models import SPUNet
from spgen.pipelines.pipeline_spgen import encode_image_tokens, pooled_embed_dim
from spgen.utils import (get_optimizer_cls, get_weight_dtype, prune_checkpoints, resume_from_checkpoint,
                         set_circular_padding)

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SDXL to generate multi-layer SP maps.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_path", type=str, required=True, help="Finetuned SP map AutoEncoder (stage 1).")
    parser.add_argument("--image_encoder", type=str, default="facebook/dinov2-large")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="exp/diffusion")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--train_batch_size", type=int, default=5, help="Objects per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=80000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--noise_offset", type=float, default=0.05)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help='Path of a saved state or "latest".')
    return parser.parse_args()


def save_model(accelerator, model, train_scheduler, args, save_dir):
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.unet.save_pretrained(os.path.join(save_dir, "unet"))
    train_scheduler.save_pretrained(os.path.join(save_dir, "scheduler"))
    with open(os.path.join(save_dir, "spgen_config.json"), "w") as f:
        json.dump({"num_layers": args.num_layers}, f, indent=2)


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir,
                                            logging_dir=os.path.join(args.output_dir, "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    weight_dtype = get_weight_dtype(accelerator.mixed_precision)

    writer = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # models
    train_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    rescale_betas_zero_snr=True, prediction_type="v_prediction")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    model = SPUNet(unet)
    set_circular_padding(model)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # the SP map latents are pre-computed, only the scaling factor of the AutoEncoder is needed
    vae_scaling_factor = AutoencoderKL.load_config(args.vae_path)["scaling_factor"]

    image_processor = AutoImageProcessor.from_pretrained(args.image_encoder)
    image_encoder = AutoModel.from_pretrained(args.image_encoder)
    image_encoder.requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    # data
    train_dataset = SPLatentDataset(
        args.data_root, args.train_splits, image_processor, vae_scaling_factor,
        num_layers=args.num_layers, cond_drop_prob=args.cond_drop_prob,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True, generator=generator,
        num_workers=args.dataloader_num_workers,
    )

    optimizer = get_optimizer_cls(args.use_8bit_adam)(
        model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,
                                                                            lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    logger.info("***** Running training *****")
    logger.info(f"  Num training objects = {len(train_dataset)}")
    logger.info(f"  Total batch size (objects, w. parallel & accumulation) = "
                f"{args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = resume_from_checkpoint(accelerator, args.resume_from_checkpoint, args.output_dir)
    first_epoch = global_step // num_update_steps_per_epoch
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    pooled_dim = pooled_embed_dim(accelerator.unwrap_model(model).unet.config)
    train_loss = 0.0
    model.train()
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                latents = batch["target_latents"].to(weight_dtype)  # (B, L * 4, h, w)
                bsz = latents.shape[0]

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn((bsz, latents.shape[1], 1, 1), device=latents.device)
                timesteps = torch.randint(0, train_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents.device).long()
                noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    image_embeds = encode_image_tokens(image_encoder, batch["image_encoder_input"].to(weight_dtype))
                    image_embeds[batch["drop_condition"]] = 0

                added_cond_kwargs = {
                    "text_embeds": torch.zeros(bsz, pooled_dim, device=latents.device, dtype=weight_dtype),
                    "time_ids": torch.zeros(bsz, 6, device=latents.device, dtype=weight_dtype),
                }

                model_pred = model(
                    rearrange(noisy_latents, "b (f c) h w -> (b f) c h w", f=args.num_layers),
                    timesteps, image_embeds,
                    num_layers=args.num_layers,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                model_pred = rearrange(model_pred, "(b f) c h w -> b (f c) h w", f=args.num_layers)

                target = train_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                train_loss += accelerator.gather(loss.detach().reshape(1)).mean().item() / args.gradient_accumulation_steps

            if not accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            global_step += 1
            if accelerator.is_main_process:
                writer.add_scalar("train/loss", train_loss, global_step=global_step)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step=global_step)
            progress_bar.set_postfix(loss=train_loss, lr=lr_scheduler.get_last_lr()[0])
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0 or global_step >= args.max_train_steps:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                if accelerator.is_main_process:
                    prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                accelerator.save_state(save_dir)
                if accelerator.is_main_process:
                    save_model(accelerator, model, train_scheduler, args, save_dir)
                    logger.info(f"Saved state to {save_dir}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()

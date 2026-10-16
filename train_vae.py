"""Stage 1: finetune the SDXL AutoEncoder on single-channel SP depth maps with geometry regularization.

    L = L_recon + alpha * L_edge + beta * L_spec + lambda * KL
"""
import argparse
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from spgen.data import SPMapDataset
from spgen.losses import EdgeLoss, frequency_domain_loss
from spgen.utils import get_optimizer_cls, prune_checkpoints, resume_from_checkpoint, set_circular_padding, to_uint8

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune the SDXL AutoEncoder on SP depth maps.")
    parser.add_argument("--pretrained_vae", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="exp/vae")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--train_batch_size", type=int, default=2,
                        help="Objects per GPU, each object contributes `num_layers` SP maps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=40000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1e-8)
    parser.add_argument("--edge_loss_weight", type=float, default=0.1, help="alpha, 0 disables L_edge.")
    parser.add_argument("--edge_threshold", type=float, default=0.55)
    parser.add_argument("--edge_ratio", type=float, default=0.95, help="mu in L_edge.")
    parser.add_argument("--edge_kernel_size", type=int, default=7)
    parser.add_argument("--freq_loss_weight", type=float, default=5e-4, help="beta, 0 disables L_spec.")
    parser.add_argument("--freq_radius", type=int, default=64, help="Radius of the high pass filter H.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500, help="Export the VAE weights every X steps.")
    parser.add_argument("--image_logging_steps", type=int, default=50)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help='Path of a saved state or "latest".')
    return parser.parse_args()


def build_vae(pretrained_vae):
    """Loads the SDXL AutoEncoder and turns its RGB input / output convs into single-channel ones."""
    vae = AutoencoderKL.from_pretrained(pretrained_vae)
    if vae.config.in_channels != 1:
        first_block_out_channels = vae.config.block_out_channels[0]
        with torch.no_grad():
            conv_in = nn.Conv2d(1, first_block_out_channels, kernel_size=3, stride=1, padding=1)
            conv_in.weight.copy_(vae.encoder.conv_in.weight.mean(1, keepdim=True))
            conv_in.bias.copy_(vae.encoder.conv_in.bias)
            vae.encoder.conv_in = conv_in

            conv_out = nn.Conv2d(first_block_out_channels, 1, kernel_size=3, padding=1)
            conv_out.weight.copy_(vae.decoder.conv_out.weight.mean(0, keepdim=True))
            conv_out.bias.copy_(vae.decoder.conv_out.bias.mean(0, keepdim=True))
            vae.decoder.conv_out = conv_out
        vae.register_to_config(in_channels=1, out_channels=1)
    set_circular_padding(vae)
    return vae


class AutoEncoderModule(nn.Module):
    """Runs encode + decode in ``forward`` so that the whole round trip goes through DDP."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, maps):
        posterior = self.vae.encode(maps).latent_dist
        latents = posterior.mode()
        return self.vae.decode(latents).sample, posterior.kl()


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

    writer = None
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    vae = build_vae(args.pretrained_vae)
    vae.requires_grad_(True)
    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    model = AutoEncoderModule(vae)

    edge_loss_fn = EdgeLoss(args.edge_threshold, args.edge_ratio, args.edge_kernel_size).to(accelerator.device)

    optimizer = get_optimizer_cls(args.use_8bit_adam)(
        vae.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    train_dataset = SPMapDataset(args.data_root, args.train_splits, num_layers=args.num_layers)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
        collate_fn=SPMapDataset.collate_fn, num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,
                                                                            lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (args.train_batch_size * args.num_layers * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num training objects = {len(train_dataset)}")
    logger.info(f"  Total batch size (SP maps, w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = resume_from_checkpoint(accelerator, args.resume_from_checkpoint, args.output_dir)
    first_epoch = global_step // num_update_steps_per_epoch
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    stats = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "edge": 0.0, "freq": 0.0}
    model.train()
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                gts = batch["maps"]

                # random azimuthal rotation and flips; the prediction is mapped back before computing the loss
                width = gts.shape[-1]
                roll = np.random.randint(width)
                h_flip, v_flip = np.random.rand() >= 0.5, np.random.rand() >= 0.5
                inputs = torch.roll(gts, roll, dims=[-1])
                if h_flip:
                    inputs = torch.flip(inputs, dims=[-1])
                if v_flip:
                    inputs = torch.flip(inputs, dims=[-2])

                pred, kl = model(inputs)

                if v_flip:
                    pred = torch.flip(pred, dims=[-2])
                if h_flip:
                    pred = torch.flip(pred, dims=[-1])
                pred = torch.roll(pred, width - roll, dims=[-1])

                num_maps = kl.shape[0]
                rec_loss = F.l1_loss(pred.float(), gts.float()) / num_maps
                kl_loss = kl.sum() / num_maps * args.kl_weight
                edge_loss = edge_loss_fn(pred.float(), gts.float()) if args.edge_loss_weight > 0 else torch.zeros_like(rec_loss)
                freq_loss = (frequency_domain_loss(pred.float(), gts.float(), radius=args.freq_radius)
                             if args.freq_loss_weight > 0 else torch.zeros_like(rec_loss))
                loss = rec_loss + args.edge_loss_weight * edge_loss + args.freq_loss_weight * freq_loss + kl_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                for key, value in zip(stats, (loss, rec_loss, kl_loss, edge_loss, freq_loss)):
                    stats[key] += accelerator.gather(value.detach().reshape(1)).mean().item() / args.gradient_accumulation_steps

            if not accelerator.sync_gradients:
                continue

            progress_bar.update(1)
            global_step += 1
            if accelerator.is_main_process:
                for key, value in stats.items():
                    writer.add_scalar(f"train/{key}", value, global_step=global_step)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step=global_step)
                if global_step == 1 or global_step % args.image_logging_steps == 0:
                    writer.add_image("train/pred_gt", np.concatenate([to_uint8(pred[0]), to_uint8(gts[0])], axis=2),
                                     global_step=global_step)
                    if args.edge_loss_weight > 0:
                        edges = edge_loss_fn.edge_masks_for_logging(pred[:1], gts[:1])
                        writer.add_image("train/edge_pred_gt", np.concatenate(edges, axis=2), global_step=global_step)
            progress_bar.set_postfix(loss=stats["loss"], lr=lr_scheduler.get_last_lr()[0])
            stats = dict.fromkeys(stats, 0.0)

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

            if accelerator.is_main_process and global_step % args.save_steps == 0:
                vae.save_pretrained(
                    os.path.join(args.output_dir, f"pretrained-{global_step}", "vae"))

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae.save_pretrained(os.path.join(args.output_dir, f"pretrained-{global_step}", "vae"))
    accelerator.end_training()


if __name__ == "__main__":
    main()

"""Train the point-cloud normal estimator used for Poisson surface extraction.

Training data are oriented point clouds sampled from the GT meshes, see
``representations/data/prepare_normal_data.py``.
"""
import argparse
import logging
import math
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from spgen.data import PointCloudDataset
from spgen.models.normal_estimator import NormalEstimator
from spgen.utils import get_optimizer_cls, prune_checkpoints, resume_from_checkpoint

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the point-cloud normal estimator.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--pcd_filename", type=str, default="points.npz")
    parser.add_argument("--num_points", type=int, default=25600, help="Points sampled per shape.")
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--f_maps", type=int, default=128)
    parser.add_argument("--num_levels", type=int, default=5)
    parser.add_argument("--grid_resolution", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="exp/normal_estimator")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--train_batch_size", type=int, default=24, help="Shapes per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=40000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=1000, help="Export the model weights every X steps.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help='Path of a saved state or "latest".')
    return parser.parse_args()


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

    model = NormalEstimator(hidden_dim=args.hidden_dim, f_maps=args.f_maps, num_levels=args.num_levels,
                            grid_resolution=args.grid_resolution)
    logger.info(f"Normal estimator params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = get_optimizer_cls(args.use_8bit_adam)(
        model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )

    train_dataset = PointCloudDataset(args.data_root, args.train_splits, args.num_points, args.pcd_filename)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True,
        num_workers=args.dataloader_num_workers,
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
    logger.info(f"  Num training shapes = {len(train_dataset)}")
    logger.info(f"  Total batch size (w. parallel & accumulation) = "
                f"{args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = resume_from_checkpoint(accelerator, args.resume_from_checkpoint, args.output_dir)
    first_epoch = global_step // num_update_steps_per_epoch
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    train_loss = 0.0
    model.train()
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                points, normals = batch["points"], batch["normals"]
                pred_normals = model(points)
                loss = F.l1_loss(pred_normals.float(), normals) / points.shape[0]

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

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

            if accelerator.is_main_process and global_step % args.save_steps == 0:
                accelerator.unwrap_model(model).save_pretrained(
                    os.path.join(args.output_dir, f"pretrained-{global_step}"))

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(os.path.join(args.output_dir, f"pretrained-{global_step}"))
    accelerator.end_training()


if __name__ == "__main__":
    main()

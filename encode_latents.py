"""Stage 2: encode the SP depth maps into latents offline with the finetuned AutoEncoder.

Outputs (unscaled latent means, the scaling factor is applied by the training dataset):
    <data_root>/<split>/latents/<subdir>/<uid>/maps/depth_{i}.pt

Run with ``accelerate launch`` to shard the objects over several GPUs.
"""
import argparse
import os

import numpy as np
import torch
from accelerate import PartialState
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

from spgen.data.datasets import iter_objects, layer_files
from spgen.utils import load_16bit_png, set_circular_padding


def parse_args():
    parser = argparse.ArgumentParser(description="Encode SP maps into latents.")
    parser.add_argument("--vae_path", type=str, required=True, help="Finetuned SP map AutoEncoder (stage 1).")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    state = PartialState()
    device = state.device

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    set_circular_padding(vae)

    objects = []
    for split in args.splits:
        objects += [(split, subdir, uid, obj_dir) for subdir, uid, obj_dir in iter_objects(args.data_root, split,
                                                                                          "preprocessed")]

    with state.split_between_processes(objects) as shard:
        for split, subdir, uid, object_dir in tqdm(shard, disable=not state.is_local_main_process):
            map_files = layer_files(os.path.join(object_dir, "maps"), ".png")[:args.num_layers]
            if len(map_files) < args.num_layers:
                print(f"Skip {uid}: found {len(map_files)} SP maps, expected {args.num_layers}")
                continue
            latent_dir = os.path.join(args.data_root, split, "latents", subdir, uid, "maps")
            out_files = [os.path.join(latent_dir, os.path.basename(f).replace(".png", ".pt")) for f in map_files]
            if not args.overwrite and all(os.path.isfile(f) for f in out_files):
                continue
            maps = torch.from_numpy(np.stack([load_16bit_png(f) for f in map_files])[:, None]) * 2 - 1
            latents = vae.encode(maps.to(device)).latent_dist.mode().cpu()
            os.makedirs(latent_dir, exist_ok=True)
            for latent, out_file in zip(latents, out_files):
                torch.save(latent.clone(), out_file)


if __name__ == "__main__":
    main()

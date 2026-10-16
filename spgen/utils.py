import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image


def set_circular_padding(model):
    """Uses circular padding in every Conv2d, since SP maps wrap around along the azimuth."""
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = "circular"
            count += 1
    return count


def load_16bit_png(path):
    """Loads a 16-bit SP map and returns a float32 array in [0, 1]."""
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(path)
    return (image / 65535.0).astype(np.float32)


def save_16bit_png(array, path):
    """Saves an array in [0, 1] as a 16-bit PNG."""
    cv2.imwrite(path, (np.clip(array, 0, 1) * 65535.0).round().astype(np.uint16))


def open_image_with_background(path, color="white"):
    """Opens an image and composites its alpha channel (if any) over a solid background."""
    img = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", img.size, color=color)
    background.paste(img, (0, 0), img)
    return background.convert("RGB")


def to_uint8(images):
    """[-1, 1] tensor -> uint8 numpy array."""
    images = (images.detach().float() / 2 + 0.5).clamp(0, 1).cpu().numpy()
    return (images * 255).round().astype(np.uint8)


def get_weight_dtype(mixed_precision):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def get_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not dirs:
        return None
    return os.path.join(output_dir, max(dirs, key=lambda x: int(x.split("-")[1])))


def resume_from_checkpoint(accelerator, resume, output_dir):
    """Restores the accelerator state and returns the global step (0 if nothing is resumed)."""
    if not resume:
        return 0
    path = get_latest_checkpoint(output_dir) if resume == "latest" else resume
    if path is None or not os.path.isdir(path):
        accelerator.print(f"Checkpoint '{resume}' does not exist. Starting a new training run.")
        return 0
    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(path)
    return int(os.path.basename(os.path.normpath(path)).split("-")[1])


def prune_checkpoints(output_dir, total_limit):
    """Keeps at most ``total_limit - 1`` checkpoints so that a new one can be saved."""
    if total_limit is None:
        return
    checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                         key=lambda x: int(x.split("-")[1]))
    for name in checkpoints[:max(len(checkpoints) - total_limit + 1, 0)]:
        shutil.rmtree(os.path.join(output_dir, name))


def get_optimizer_cls(use_8bit_adam):
    if not use_8bit_adam:
        return torch.optim.AdamW
    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError("Please install bitsandbytes to use 8-bit Adam: `pip install bitsandbytes`") from e
    return bnb.optim.AdamW8bit


def sp_depth_to_points(depth, sphere_radius=1.0):
    """Lifts an SP depth map to 3D points.

    The SP map stores, for the ray cast from the sphere point (polar angle phi, azimuth theta)
    towards the origin, the distance between the intersection and the origin (normalized by the
    sphere radius). Pixels without intersection are 0.

    Args:
        depth: (H, W) array in [0, 1], H indexes the polar angle and W the azimuth.
    Returns:
        (H, W, 3) array of points.
    """
    h, w = depth.shape
    phi, theta = np.meshgrid(np.linspace(0, np.pi, h), np.linspace(0, 2 * np.pi, w))
    x = sphere_radius * np.sin(phi) * np.cos(-theta - np.pi / 2)
    z = sphere_radius * np.sin(phi) * np.sin(-theta - np.pi / 2)
    y = sphere_radius * np.cos(phi)
    origins = np.stack((x.T, y.T, z.T), axis=-1)
    directions = -origins / (np.linalg.norm(origins, axis=-1, keepdims=True) + 1e-8)
    return origins + directions * (sphere_radius - depth[..., None])

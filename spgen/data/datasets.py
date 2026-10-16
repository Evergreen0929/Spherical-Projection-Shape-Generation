"""Datasets for the three training stages.

Expected layout (see ``representations/README.md`` for how each part is produced)::

    <data_root>/<split>/
    ├── preprocessed/<subdir>/<uid>/
    │   ├── mesh.ply
    │   ├── maps/depth_{0..K-1}.png              # 16-bit SP depth maps
    │   └── render_images/<uid>/rgb_*.webp       # rendered condition images
    ├── latents/<subdir>/<uid>/maps/depth_{0..K-1}.pt   # SP map latents (encode_latents.py)
    └── pointclouds/<subdir>/<uid>/points.npz    # oriented surface points (normal estimator)
"""
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from spgen.utils import load_16bit_png

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def iter_objects(data_root, split, stage):
    """Yields ``(subdir, uid, object_dir)`` for every object of ``<data_root>/<split>/<stage>``."""
    root = os.path.join(data_root, split, stage)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} does not exist")
    for subdir in sorted(os.listdir(root)):
        subdir_path = os.path.join(root, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for uid in sorted(os.listdir(subdir_path)):
            object_dir = os.path.join(subdir_path, uid)
            if os.path.isdir(object_dir):
                yield subdir, uid, object_dir


def layer_files(maps_dir, ext, prefix="depth"):
    """Returns the SP map files of one object sorted by layer index."""
    if not os.path.isdir(maps_dir):
        return []
    pattern = re.compile(rf"^{prefix}_(\d+){re.escape(ext)}$")
    files = [(int(m.group(1)), name) for name in os.listdir(maps_dir) if (m := pattern.match(name))]
    return [os.path.join(maps_dir, name) for _, name in sorted(files)]


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def render_image_dir(object_dir, uid):
    nested = os.path.join(object_dir, "render_images", uid)
    return nested if os.path.isdir(nested) else os.path.join(object_dir, "render_images")


def render_image_files(object_dir, uid):
    image_dir = render_image_dir(object_dir, uid)
    if not os.path.isdir(image_dir):
        return []
    return [os.path.join(image_dir, name) for name in sorted(os.listdir(image_dir))
            if name.startswith("rgb_") and name.lower().endswith(IMAGE_EXTS)]


class SPMapDataset(Dataset):
    """SP depth maps of each object, used to finetune the AutoEncoder."""

    def __init__(self, data_root, splits, num_layers=4):
        self.num_layers = num_layers
        self.items = []
        for split in splits:
            for _, uid, object_dir in iter_objects(data_root, split, "preprocessed"):
                files = layer_files(os.path.join(object_dir, "maps"), ".png")
                if len(files) >= num_layers:
                    self.items.append((uid, files[:num_layers]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, files = self.items[index]
        maps = np.stack([load_16bit_png(path) for path in files])[:, None]  # (L, 1, H, W)
        return {"maps": torch.from_numpy(maps) * 2 - 1, "uid": uid}

    @staticmethod
    def collate_fn(examples):
        return {
            "maps": torch.cat([e["maps"] for e in examples], dim=0),  # (B * L, 1, H, W)
            "uid": [e["uid"] for e in examples],
        }


class SPLatentDataset(Dataset):
    """Pre-computed SP map latents paired with a randomly chosen rendered view as condition."""

    def __init__(self, data_root, splits, image_processor, vae_scaling_factor, num_layers=4, cond_drop_prob=0.1,
                 random_view=True):
        """
        Args:
            image_processor: DINOv2 image processor.
            vae_scaling_factor: scaling factor applied to the stored (unscaled) latents.
            cond_drop_prob: probability of dropping the condition (classifier-free guidance).
            random_view: pick a random view as condition, otherwise always the first one.
        """
        self.image_processor = image_processor
        self.vae_scaling_factor = vae_scaling_factor
        self.cond_drop_prob = cond_drop_prob
        self.random_view = random_view

        self.items = []
        for split in splits:
            for subdir, uid, latent_dir in iter_objects(data_root, split, "latents"):
                latent_files = layer_files(os.path.join(latent_dir, "maps"), ".pt")
                object_dir = os.path.join(data_root, split, "preprocessed", subdir, uid)
                images = render_image_files(object_dir, uid)
                if len(latent_files) < num_layers or not images:
                    continue
                self.items.append({"uid": uid, "latents": latent_files[:num_layers], "images": images})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        drop = random.random() < self.cond_drop_prob
        view = random.randrange(len(item["images"])) if self.random_view else 0

        target = torch.cat([load_tensor(path) for path in item["latents"]], dim=0)

        image = Image.open(item["images"][view])
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]
        if drop:
            pixel_values = torch.zeros_like(pixel_values)

        return {
            "target_latents": target * self.vae_scaling_factor,  # (L * 4, h, w)
            "image_encoder_input": pixel_values,
            "drop_condition": torch.tensor(drop),
            "image_path": item["images"][view],
            "uid": item["uid"],
        }


class PointCloudDataset(Dataset):
    """Oriented point clouds sampled from the GT meshes, used to train the normal estimator."""

    def __init__(self, data_root, splits, num_points=25600, filename="points.npz"):
        self.num_points = num_points
        self.items = []
        for split in splits:
            for _, uid, object_dir in iter_objects(data_root, split, "pointclouds"):
                path = os.path.join(object_dir, filename)
                if os.path.isfile(path):
                    self.items.append((uid, path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, path = self.items[index]
        data = np.load(path)
        points, normals = data["points"], data["normals"]
        choice = torch.randint(0, points.shape[0], (self.num_points,)).numpy()
        return {
            "points": torch.from_numpy(points[choice]).float(),
            "normals": torch.from_numpy(normals[choice]).float(),
            "uid": uid,
        }

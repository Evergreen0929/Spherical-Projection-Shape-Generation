"""Single image to 3D shape: SP map generation, lifting to points, normal estimation and Poisson reconstruction."""
import argparse
import json
import os

import numpy as np
import open3d as o3d
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from spgen.losses import SobelOperator
from spgen.models import SPUNet
from spgen.models.normal_estimator import NormalEstimator
from spgen.pipelines import SPGenPipeline
from spgen.pipelines.pipeline_spgen import load_schedulers
from spgen.utils import open_image_with_background, save_16bit_png, set_circular_padding, sp_depth_to_points

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3D shapes from single images with SPGen.")
    parser.add_argument("--input", type=str, required=True, help="An image or a folder of images.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--unet_path", type=str, required=True,
                        help="Folder containing `unet/`, `scheduler/` and `spgen_config.json` (train_diffusion.py).")
    parser.add_argument("--vae_path", type=str, required=True, help="Finetuned SP map AutoEncoder.")
    parser.add_argument("--normal_estimator_path", type=str, required=True)
    parser.add_argument("--image_encoder", type=str, default="facebook/dinov2-large")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of SP layers, defaults to the value stored in `spgen_config.json`.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=25600, help="Points fed to the normal estimator.")
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--edge_threshold", type=float, default=0.7,
                        help="Pixels whose SP map gradient exceeds this value are dropped (depth discontinuities).")
    parser.add_argument("--save_maps", action="store_true", help="Also save the generated SP maps.")
    parser.add_argument("--save_pcd", action="store_true", help="Also save the oriented point cloud.")
    return parser.parse_args()


def build_pipeline(args, device, dtype=torch.float16):
    config_path = os.path.join(args.unet_path, "spgen_config.json")
    config = json.load(open(config_path)) if os.path.isfile(config_path) else {}
    num_layers = args.num_layers or config.get("num_layers", 4)

    _, scheduler = load_schedulers(args.unet_path)
    unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder="unet").to(dtype=dtype)
    unet = SPUNet(unet).to(device).eval()
    set_circular_padding(unet)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device, torch.float32).eval()
    set_circular_padding(vae)
    image_encoder = AutoModel.from_pretrained(args.image_encoder).to(device, dtype).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.image_encoder)

    pipeline = SPGenPipeline(vae=vae, unet=unet, scheduler=scheduler, image_encoder=image_encoder,
                             image_processor=image_processor)
    return pipeline, num_layers


def sample_points(points, num_points, voxel_size=1 / 128):
    """Voxel down-sampling followed by farthest point sampling (random sampling if PyTorch3D is unavailable)."""
    voxels = torch.floor(points / voxel_size).int()
    _, inverse = torch.unique(voxels, dim=0, return_inverse=True)
    num_voxels = int(inverse.max()) + 1
    points = torch.zeros(num_voxels, 3, device=points.device).index_add_(0, inverse, points)
    points = points / torch.zeros(num_voxels, device=points.device).index_add_(
        0, inverse, torch.ones_like(inverse, dtype=torch.float32))[:, None]

    if points.shape[0] < num_points:
        extra = torch.randint(0, points.shape[0], (num_points - points.shape[0],), device=points.device)
        return torch.cat([points, points[extra]])
    try:
        from pytorch3d.ops import sample_farthest_points
        return sample_farthest_points(points[None], K=num_points)[0][0]
    except ImportError:
        return points[torch.randperm(points.shape[0], device=points.device)[:num_points]]


@torch.no_grad()
def sp_maps_to_mesh(sp_maps, normal_estimator, sobel, args):
    """sp_maps: (L, 1, H, W) in [-1, 1] -> oriented point cloud and Poisson mesh."""
    depths = (sp_maps.float() + 1) / 2
    all_points = []
    for depth in depths:
        edge_mask = (sobel(depth[None]) < args.edge_threshold)[0, 0].cpu().numpy()
        depth = depth[0].cpu().numpy()
        mask = (depth > 0.02) & edge_mask
        all_points.append(sp_depth_to_points(depth)[mask])
    points = torch.from_numpy(np.concatenate(all_points)).float().to(sp_maps.device)

    points = sample_points(points, args.num_points)
    normals = normal_estimator(points[None])[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.cpu().numpy())
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth)
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
    mesh.compute_vertex_normals()
    return pcd, mesh


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        inputs = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input)) if f.lower().endswith(IMAGE_EXTS)]
    else:
        inputs = [args.input]

    pipeline, num_layers = build_pipeline(args, device)
    normal_estimator = NormalEstimator.from_pretrained(args.normal_estimator_path).to(device)
    sobel = SobelOperator().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for path in tqdm(inputs):
        name = os.path.splitext(os.path.basename(path))[0]
        image = open_image_with_background(path).resize((512, 512))
        with torch.autocast(device.type):
            sp_maps, _ = pipeline(image, num_layers=num_layers, num_inference_steps=args.steps,
                                  guidance_scale=args.guidance_scale, generator=generator)

        if args.save_maps:
            map_dir = os.path.join(args.output_dir, name, "maps")
            os.makedirs(map_dir, exist_ok=True)
            for i, sp_map in enumerate(sp_maps):
                save_16bit_png(sp_map[0].float().cpu().numpy() / 2 + 0.5, os.path.join(map_dir, f"depth_{i}.png"))

        pcd, mesh = sp_maps_to_mesh(sp_maps, normal_estimator, sobel, args)
        if args.save_pcd:
            o3d.io.write_point_cloud(os.path.join(args.output_dir, f"{name}_pcd.ply"), pcd)
        o3d.io.write_triangle_mesh(os.path.join(args.output_dir, f"{name}.ply"), mesh)


if __name__ == "__main__":
    main()

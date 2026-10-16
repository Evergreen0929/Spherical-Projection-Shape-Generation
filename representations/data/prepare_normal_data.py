"""Samples oriented surface points from the normalized GT meshes to train the normal estimator.

Objaverse meshes are often non-manifold, open or have inconsistently oriented faces, so every mesh is
first converted into a watertight surface: the unsigned distance field (UDF) is evaluated on a dense
grid, the enclosed interior is filled (UDF set to 0), and the iso-surface one voxel away from the
shape is extracted with marching cubes. Points and outward facing normals are then sampled from it.

Input:  <data_root>/<split>/preprocessed/<subdir>/<uid>/mesh.ply   (prepare_sp_depth_map.py)
Output: <data_root>/<split>/pointclouds/<subdir>/<uid>/points.npz   (float32 `points` and `normals`, N x 3)
"""
import argparse
import os
from multiprocessing import Pool

import mcubes
import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--grid_size", type=int, default=196)
    parser.add_argument("--num_points", type=int, default=51200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def watertight_remesh(mesh, grid_size, mesh_scale=0.95):
    """Returns a watertight, outward oriented surface of ``mesh`` in the original coordinate frame."""
    center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
    scale = (mesh.vertices.max(0) - mesh.vertices.min(0)).max() / 2 / mesh_scale
    vertices = (mesh.vertices - center) / scale  # fits in [-mesh_scale, mesh_scale]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(mesh.faces.astype(np.uint32)))

    coords = np.linspace(-1, 1, grid_size + 1, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    udf = scene.compute_distance(o3d.core.Tensor(grid.reshape(-1, 3))).numpy().reshape(grid.shape[:3])

    voxel = 2.0 / grid_size
    solid = ndimage.binary_fill_holes(udf <= voxel)
    udf = np.where(solid, 0.0, udf)

    verts, faces = mcubes.marching_cubes(udf, voxel)
    remeshed = trimesh.Trimesh(verts * voxel - 1, faces, process=True)
    if remeshed.volume < 0:
        remeshed.invert()
    remeshed.vertices = remeshed.vertices * scale + center
    return remeshed


def process(task):
    mesh_path, out_path, grid_size, num_points = task
    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        remeshed = watertight_remesh(mesh, grid_size)
        points, face_idx = trimesh.sample.sample_surface(remeshed, num_points)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, points=np.asarray(points, dtype=np.float32),
                 normals=np.asarray(remeshed.face_normals[face_idx], dtype=np.float32))
        return True
    except Exception as e:
        print(f"Failed to process {mesh_path}: {e}")
        return False


def main():
    args = parse_args()
    tasks = []
    for split in args.splits:
        src_root = os.path.join(args.data_root, split, "preprocessed")
        dst_root = os.path.join(args.data_root, split, "pointclouds")
        for subdir in sorted(os.listdir(src_root)):
            for uid in sorted(os.listdir(os.path.join(src_root, subdir))):
                mesh_path = os.path.join(src_root, subdir, uid, "mesh.ply")
                out_path = os.path.join(dst_root, subdir, uid, "points.npz")
                if os.path.isfile(mesh_path) and (args.overwrite or not os.path.isfile(out_path)):
                    tasks.append((mesh_path, out_path, args.grid_size, args.num_points))

    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process, tasks), total=len(tasks)))
    print(f"Processed {sum(results)} / {len(tasks)} meshes.")


if __name__ == "__main__":
    main()

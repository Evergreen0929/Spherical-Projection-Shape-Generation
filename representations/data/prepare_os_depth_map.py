import numpy as np
import trimesh
import cv2
import os
from tqdm import tqdm
import open3d as o3d

def normalize_mesh(mesh):
    mesh.apply_translation(-mesh.centroid)

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]

    scale = 1.0 / max(extents)
    mesh.apply_scale(scale)

    return mesh


def load_glb(glb_file):
    try:
        if glb_file.endswith(".glb") or glb_file.endswith(".obj"):
            try:
                mesh = trimesh.load(glb_file)
                mesh = normalize_mesh(mesh)

                return mesh

            except Exception as e:
                print(f'Error loading mesh: {e}')
                return None
        else:
            raise FileNotFoundError(f'{glb_file} is not a .glb file!')
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return None
    except Exception as e:
        print(f'Unexpected error: {e}')
        return None


def fast_project_mesh_to_sphere(mesh, sphere_radius=1, polar_resolution=256, azimuthal_resolution=256 * 2, max_depth=4):
    # Generate points on the sphere surface
    phi = np.linspace(0, np.pi, polar_resolution)  # Polar angle
    theta = np.linspace(0, 2 * np.pi, azimuthal_resolution)  # Azimuthal angle
    phi, theta = np.meshgrid(phi, theta)

    # Coordinates of points on the sphere surface
    x = sphere_radius * np.sin(phi) * np.cos(-theta - np.pi / 2)
    z = sphere_radius * np.sin(phi) * np.sin(-theta - np.pi / 2)
    y = sphere_radius * np.cos(phi)

    # Flatten the coordinates for batch processing
    ray_origins = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    directions = -ray_origins  # Since the ray points towards the center
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize direction vectors

    # Calculate intersection points using ray.intersects_location
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, directions)

    # Initialize maps
    depth_map = [np.zeros((polar_resolution, azimuthal_resolution, 1)) for _ in range(max_depth)]

    # Process intersections
    intersections = []
    intersection_counts = np.zeros(ray_origins.shape[0], dtype=int)

    for loc, tri_idx, ray_idx in zip(locations, index_tri, index_ray):
        distance = np.linalg.norm(loc - ray_origins[ray_idx])
        if distance <= sphere_radius:
            intersections.append(loc)
            intersection_counts[ray_idx] += 1

            depth = intersection_counts[ray_idx] - 1
            if depth < max_depth:
                i, j = divmod(ray_idx, polar_resolution)
                depth_map[depth][j, i, :] = sphere_radius - distance  # Corrected indices

    return depth_map


def save_arrays_as_16bit_images(depth_map, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, depth in enumerate(depth_map):
        normalized_depth = (depth * 65535).round().astype(np.uint16)
        cv2.imwrite(f'{path}/depth_{i}.png', normalized_depth)


def process_data(src_dir, dst_dir, params):
    subdirs = os.listdir(src_dir)
    all_files = {}
    ids = {}
    successful_loads = 0
    failed_loads = 0

    for subdir in subdirs:
        all_files[subdir] = [os.path.join(src_dir, subdir, file) for file in os.listdir(os.path.join(src_dir, subdir))
                             if file.endswith('.glb') or file.endswith('.obj')]
        ids[subdir] = [file.split('.')[0] for file in os.listdir(os.path.join(src_dir, subdir))
                       if file.endswith('.glb') or file.endswith('.obj')]

    for subdir, files in all_files.items():
        for i, file in enumerate(tqdm(files, desc=f"Processing {subdir}", leave=False)):
            id = ids[subdir][i]
            save_path = os.path.join(dst_dir, subdir, id)
            os.makedirs(save_path, exist_ok=True)

            try:
                mesh = load_glb(file)
                if mesh is None:
                    raise ValueError("Mesh loading failed")
                mesh.export(os.path.join(save_path, 'mesh.ply'))
                mesh = trimesh.load(os.path.join(save_path, 'mesh.ply'))
                depth_map = fast_project_mesh_to_sphere(mesh, polar_resolution=params['resolution'] // 2,
                                                                           azimuthal_resolution=params['resolution'],
                                                                           max_depth=params['max_depth'])
                save_arrays_as_16bit_images(depth_map, path=os.path.join(save_path, 'maps'))
                successful_loads += 1
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                failed_loads += 1

            # if i > 10:
            #     break
        # break

    total_files = sum([len(value) for value in all_files.values()])
    print(f"Total {successful_loads + failed_loads} / {total_files} data have been processed!")
    print(f"Successfully loaded: {successful_loads}")
    print(f"Failed to load: {failed_loads}")


if __name__ == '__main__':
    src_path = './objaverse-12k/test/glbs'
    dst_path = './objaverse-12k/test/preprocessed'

    params = {
        'resolution': 512,
        'max_depth': 4,
    }

    # assert params['max_depth'] % 2 == 0

    process_data(src_path, dst_path, params)

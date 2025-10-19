import os
import shutil
import cv2
from tqdm import tqdm
import trimesh
import numpy as np
import open3d as o3d

def load_16bit_image(filepath, mode='pos'):
    image = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    image = image / 65535 * 2 - 1
    # image = upsample_image(image)
    if mode == 'pos':
        image = image / 2
    return image

def convert_pos_normal_to_pcd(pos, normal):
    pos_mask = np.linalg.norm(pos, axis=-1) > 0.001
    normal_mask = np.linalg.norm(normal, axis=-1) > 0.001
    mask = pos_mask * normal_mask
    selected_pos = pos[mask]
    selected_normal = normal[mask]
    return selected_pos, selected_normal


def trimesh_to_open3d(trimesh_mesh):
    vertices = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def cal_chamfer_distance(points, normals, gt_mesh, num_points=10000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    try:
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        return None, (None, gt_mesh)

    if isinstance(gt_mesh, trimesh.Trimesh):
        gt_mesh = trimesh_to_open3d(gt_mesh)

    gt_mesh.compute_vertex_normals()

    pcd_poisson = poisson_mesh.sample_points_uniformly(number_of_points=num_points)
    pcd_gt = gt_mesh.sample_points_uniformly(number_of_points=num_points)
    
    dists1 = pcd_poisson.compute_point_cloud_distance(pcd_gt)
    dists2 = pcd_gt.compute_point_cloud_distance(pcd_poisson)

    chamfer_distance = np.mean(dists1) + np.mean(dists2)

    return chamfer_distance, (poisson_mesh, gt_mesh)


def validation_check(pos, normal):
    pos_direction = pos / np.linalg.norm(pos, axis=-1)[:, :, np.newaxis]
    angle = (pos_direction * normal).sum(-1)
    return (angle < 0).astype(np.float32).sum()

def cal_all_chamfer_distances(src_dir, dst_dir):
    subdirs = os.listdir(src_dir)
    all_files = {}
    ids = {}

    for subdir in subdirs:
        all_files[subdir] = [os.path.join(src_dir, subdir, file) for file in os.listdir(os.path.join(src_dir, subdir))]
        ids[subdir] = os.listdir(os.path.join(src_dir, subdir))

    cds = {}
    for subdir, files in all_files.items():
        # if subdir in ['000-000', '000-001', '000-002']:
        #     continue
        cds[subdir] = []
        dis_save_path = os.path.join(dst_dir, subdir)
        os.makedirs(dis_save_path, exist_ok=True)
        for i, file in enumerate(tqdm(files, desc=f"Calculating {subdir}", leave=False)):
            id = ids[subdir][i]
            maps_path = os.path.join(file, 'maps')
            if not os.path.exists(maps_path):
                print(f'{file} discarded! Empty file.')
                shutil.rmtree(file)
                continue

            try:
                gt_mesh = trimesh.load(os.path.join(file, 'mesh.ply'))
                pos_list = [load_16bit_image(os.path.join(file, 'maps', f), mode=f.split('_')[0]) for f
                            in os.listdir(maps_path) if 'pos' in f]
                normal_list = [load_16bit_image(os.path.join(file, 'maps', f), mode=f.split('_')[0]) for f
                               in os.listdir(maps_path) if 'normal' in f]
                flag = validation_check(pos_list[0], normal_list[0])

                points = []
                normals = []
                for _pos, _normal in zip(pos_list, normal_list):
                    point, normal = convert_pos_normal_to_pcd(_pos, _normal)
                    points.append(point)
                    normals.append(normal)
                points = np.concatenate(points, axis=0)
                normals = np.concatenate(normals, axis=0)

                cd, meshes = cal_chamfer_distance(points, normals, gt_mesh)

                if cd == None:
                    print(f'Unknown error when recon {file}.')
                    error_save_path = os.path.join('./hf-objaverse-v1/error', subdir)
                    os.makedirs(error_save_path, exist_ok=True)
                    shutil.move(file, error_save_path)
                elif cd > 0.05:
                    if flag > 8192:
                        print(f'{file} discarded! Large Recon Error {cd} and Invalid Normal {flag}.')
                    else:
                        print(f'{file} discarded! Large Recon Error {cd}.')

                    save_path = os.path.join('./hf-objaverse-v1/recon', subdir, id)
                    os.makedirs(save_path, exist_ok=True)
                    o3d.io.write_triangle_mesh(os.path.join(save_path, "recon_mesh.ply"), meshes[0])
                    o3d.io.write_triangle_mesh(os.path.join(save_path, "gt_mesh.ply"), meshes[1])

                    shutil.move(file, dis_save_path)
                else:
                    cds[subdir].append(cd)
                    # save_path = os.path.join('./objaverse-12k/train/recon', subdir, id)
                    # os.makedirs(save_path, exist_ok=True)
                    # o3d.io.write_triangle_mesh(os.path.join(save_path, "recon_mesh.ply"), meshes[0])
                    # o3d.io.write_triangle_mesh(os.path.join(save_path, "gt_mesh.ply"), meshes[1])

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        print(f'mean chamfer distance for {subdir} is {sum(cds[subdir]) / len(cds[subdir])}')

        #     if i > 10:
        #         break
        # break

    return cds

if __name__ == '__main__':
    # src_path = './hf-objaverse-v1/preprocessed'
    # dst_path = './hf-objaverse-v1/discarded'

    src_path = './objaverse-12k/train/preprocessed'
    dst_path = './objaverse-12k/train/discarded'

    all_cds = cal_all_chamfer_distances(src_path, dst_path)
    mean_cds = {key: sum(value) / len(value) for key, value in all_cds.items()}
    mean = sum(mean_cds.values()) / len(mean_cds.values())

    for k, v in mean_cds.items():
        print(f'{k}:\t{v}')
    print(f"All mean chamfer distance for {sum([len(value) for value in all_cds.values()])} data:\t", mean)
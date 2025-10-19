import os
from tqdm import tqdm
import shutil

def delete_all_files(folder_path):
    shutil.rmtree(folder_path)
    print(f"All files in {folder_path} have been deleted.")

def validation_check(src_dir):
    subdirs = os.listdir(src_dir)
    all_files = {}
    ids = {}
    successful_loads = 0
    failed_loads = 0
    failed_info = {}

    for subdir in subdirs:
        all_files[subdir] = [os.path.join(src_dir, subdir, file) for file in os.listdir(os.path.join(src_dir, subdir))]
        ids[subdir] = [file.split('.')[0] for file in os.listdir(os.path.join(src_dir, subdir))]

    for subdir, files in all_files.items():
        for i, file in enumerate(tqdm(files, desc=f"Processing {subdir}", leave=False)):
            id = ids[subdir][i]
            mesh_path = os.path.join(file, 'mesh.ply')
            render_path = os.path.join(file, 'render_images')
            map_path = os.path.join(file, 'maps')

            render_latent_path = render_path.replace('preprocessed', 'latents')
            map_latent_path = map_path.replace('preprocessed', 'latents')

            if os.path.isfile(mesh_path) is False:
                failed_loads += 1
                print(f'No mesh ply! ', subdir, ': ', id)
                failed_info[f'{subdir}.{id}'] = 'No mesh ply'
            elif os.path.isdir(render_path) is False or len(os.listdir(os.path.join(render_path, id))) != 14:
                failed_loads += 1
                print(f'No rendered images! ', subdir, ': ', id)
                if os.path.isdir(render_path) is False:
                    failed_info[f'{subdir}.{id}'] = 'No rendered images'
                else:
                    failed_info[f'{subdir}.{id}'] = f'Insufficient render images {len(os.listdir(os.path.join(render_path, id)))}/14'
            elif os.path.isdir(map_path) is False or len(os.listdir(map_path)) != 8:
                failed_loads += 1
                print(f'No rendered maps! ', subdir, ': ', id)
                if os.path.isdir(map_path) is False:
                    failed_info[f'{subdir}.{id}'] = 'No rendered maps'
                else:
                    failed_info[f'{subdir}.{id}'] = f'Insufficient render maps {len(os.listdir(map_path))}/8'
            elif os.path.isdir(render_latent_path) is False or len(os.listdir(os.path.join(render_latent_path, id))) != 13:
                failed_loads += 1
                print(f'No rendered image latents! ', subdir, ': ', id)
                if os.path.isdir(render_latent_path) is False:
                    failed_info[f'{subdir}.{id}'] = 'No image latents'
                else:
                    failed_info[f'{subdir}.{id}'] = f'Insufficient image latents {len(os.listdir(os.path.join(render_latent_path, id)))}/13'
            elif os.path.isdir(map_latent_path) is False or len(os.listdir(map_latent_path)) != 8:
                failed_loads += 1
                print(f'No rendered map latents! ', subdir, ': ', id)
                if os.path.isdir(map_latent_path) is False:
                    failed_info[f'{subdir}.{id}'] = 'No map latents'
                else:
                    failed_info[f'{subdir}.{id}'] = f'Insufficient maps_latents {len(os.listdir(map_latent_path))}/8'
            else:
                successful_loads += 1


    total_files = sum([len(value) for value in all_files.values()])
    print(f"Total {successful_loads + failed_loads} / {total_files} data have been processed!")
    print(f"Successfully loaded: {successful_loads}")
    print(f"Failed to load: {failed_loads}, items: \n")

    if failed_info != {}:
        for k, v in failed_info.items():
            print(k, v)
            # delete_all_files(os.path.join(src_path, k.split('.')[0], k.split('.')[1]))

if __name__ == '__main__':
    # src_path = '/home/jdzhang/datasets/hf-objaverse-v1/preprocessed'
    src_path = '/home/jdzhang/datasets/objaverse-12k/additional/preprocessed'
    validation_check(src_path)
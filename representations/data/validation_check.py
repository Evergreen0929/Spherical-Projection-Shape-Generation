import argparse
import os

from tqdm import tqdm

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def count_files(path, prefix, exts):
    if not os.path.isdir(path):
        return 0
    return len([f for f in os.listdir(path) if f.startswith(prefix) and f.lower().endswith(exts)])


def render_dir(object_dir, uid):
    nested = os.path.join(object_dir, 'render_images', uid)
    return nested if os.path.isdir(nested) else os.path.join(object_dir, 'render_images')


def check_object(object_dir, uid, num_layers, check_latents):
    if not os.path.isfile(os.path.join(object_dir, 'mesh.ply')):
        return 'No mesh ply'
    num_maps = count_files(os.path.join(object_dir, 'maps'), 'depth_', ('.png',))
    if num_maps < num_layers:
        return f'Insufficient SP maps {num_maps}/{num_layers}'
    num_images = count_files(render_dir(object_dir, uid), 'rgb_', IMAGE_EXTS)
    if num_images == 0:
        return 'No rendered images'

    latent_dir = object_dir.replace(f'{os.sep}preprocessed{os.sep}', f'{os.sep}latents{os.sep}')
    if check_latents:
        num_latents = count_files(os.path.join(latent_dir, 'maps'), 'depth_', ('.pt',))
        if num_latents < num_layers:
            return f'Insufficient SP map latents {num_latents}/{num_layers}'
    return None


def validation_check(src_dir, num_layers=4, check_latents=True):
    """Checks that every object under ``<split>/preprocessed`` has all files required for training."""
    src_dir = os.path.normpath(src_dir)
    successful, failed_info = 0, {}
    for subdir in sorted(os.listdir(src_dir)):
        for uid in tqdm(sorted(os.listdir(os.path.join(src_dir, subdir))), desc=f"Checking {subdir}", leave=False):
            reason = check_object(os.path.join(src_dir, subdir, uid), uid, num_layers, check_latents)
            if reason is None:
                successful += 1
            else:
                failed_info[f'{subdir}/{uid}'] = reason

    print(f"Total {successful + len(failed_info)} objects checked.")
    print(f"Complete: {successful}")
    print(f"Incomplete: {len(failed_info)}")
    for k, v in failed_info.items():
        print(k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True, help='<data_root>/<split>/preprocessed')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--skip_latents', action='store_true')
    args = parser.parse_args()
    validation_check(args.src_path, args.num_layers, not args.skip_latents)

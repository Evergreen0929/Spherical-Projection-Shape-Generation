"""Downloads the Objaverse objects of a split and arranges them as <data_root>/<split>/glbs/<subdir>/<uid>.glb.

The uid lists in ``uids/`` contain the objects used in the paper:
    train.txt  159,912 training objects
    test.txt     1,993 held-out objects (Objaverse evaluation split), never used for training
"""
import argparse
import multiprocessing
import os
import shutil

import objaverse

UID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uids")


def load_uids(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--copy", action="store_true", help="Copy the files instead of creating symlinks.")
    args = parser.parse_args()

    for split in args.splits:
        uids = load_uids(os.path.join(UID_DIR, f"{split}.txt"))
        print(f"Downloading {len(uids)} objects of the {split} split with {args.processes} processes.")
        objects = objaverse.load_objects(uids=uids, download_processes=args.processes)
        print(f"{len(objects)} / {len(uids)} objects downloaded.")

        for uid, path in objects.items():
            # objaverse stores the files as .../glbs/<subdir>/<uid>.glb
            dst = os.path.join(args.data_root, split, "glbs", os.path.basename(os.path.dirname(path)), f"{uid}.glb")
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if args.copy:
                shutil.copy(path, dst)
            else:
                os.symlink(os.path.abspath(path), dst)


if __name__ == "__main__":
    main()

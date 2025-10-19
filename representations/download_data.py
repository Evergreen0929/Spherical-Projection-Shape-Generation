import numpy as np
import objaverse
import multiprocessing

NPZ_FILE = "uid_dict.npz"

def load_uids(npz_file):
    data = np.load(npz_file)
    return data['train'], data['test']


def main():
    processes = multiprocessing.cpu_count()
    print(f"use {processes} processes for dowloading。")

    train_uids, test_uids = load_uids(NPZ_FILE)
    print(f"loaded train {len(train_uids)}, test {len(test_uids)} UIDs。")

    train_objects = objaverse.load_objects(uids=train_uids, download_processes=processes)
    print(f"train set dowload finished! {len(train_objects)} objects dowloaded")

    test_objects = objaverse.load_objects(uids=test_uids, download_processes=processes)
    print(f"test set dowload finished! {len(test_objects)} objects dowloaded")


if __name__ == "__main__":
    main()

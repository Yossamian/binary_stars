import torch
from torch.utils.data import random_split
from binary_stars.datasets import GaiaDataset
from pathlib import Path
import numpy as np

def split_dataset(train, val, target_param, paths, seed=42):
    full_data = GaiaDataset(paths, target_param=target_param)
    train_len = int(len(full_data) * train)
    val_len = int(len(full_data)*val)
    test_len = len(full_data) - train_len - val_len
    train_data, val_data, test_data = random_split(full_data, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    return train_data, val_data, test_data


def split_dataset_new(val_test_len, target_param, paths, seed=42):
    full_data = GaiaDataset(paths, target_param=target_param)
    train_len = len(full_data) - (2 * val_test_len)
    train_data, val_data, test_data = random_split(full_data, [train_len, val_test_len, val_test_len], generator=torch.Generator().manual_seed(seed))

    return train_data, val_data, test_data


def list_values(num=0):
    root_dir = Path("/media/sam/data/work/stars/gaia")
    file_directory = root_dir.joinpath(f'{num}.npz')
    data = np.load(file_directory)
    return data.files

if __name__ == "__main__":
    f = list_values(0)
    print(f)
    print("******")
    for i in f:
        print(i)
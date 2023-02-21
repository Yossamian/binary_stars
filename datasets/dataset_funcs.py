import torch
from torch.utils.data import random_split
from binary_stars.datasets import GaiaDataset


def split_dataset(train, val, target_param, paths, seed=42):
    full_data = GaiaDataset(paths, target_param=target_param)
    train_len = int(len(full_data) * train)
    val_len = int(len(full_data)*val)
    test_len = len(full_data) - train_len - val_len
    train_data, val_data, test_data = random_split(full_data, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    return train_data, val_data, test_data
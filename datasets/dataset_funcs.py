import random

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


def gen_11may_data():
    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']

    root_dir = Path("/media/sam/data/work/stars/gaia")

    # Empty files for creating x, y lists
    list_total_spectra = []
    list_total_labels = []

    # Iterate through dataset files
    for num in range(22):
        # Read file
        file_directory = root_dir.joinpath(f'{num}.npz')
        data = np.load(file_directory)

        # Save spectra, append to list
        file_spectra = data['fi']
        list_total_spectra.append(file_spectra)

        print(f"Adding {file_spectra.shape} spectra array from {num}.npz")

        # Read all relevant y, append to list
        list_labels = [data[label] for label in labels]
        file_labels = np.column_stack(list_labels)
        list_total_labels.append(file_labels)

    # Turn lists into np
    total_spectra = np.row_stack(list_total_spectra)
    total_labels = np.row_stack(list_total_labels)

    print(f"COMPLETE all dataset: {total_spectra.shape} spectra array, {total_labels.shape} label array")

    # ensure input data is floats
    spectra = torch.from_numpy(total_spectra).float()

    # For new order:
    new_order = []
    for i in range(len(labels)):
        if i % 2 == 0:
            new_order.append(i + 1)
        else:
            new_order.append(i - 1)

    y_alt = total_labels[:, new_order]
    y = np.stack((total_labels, y_alt), axis=-1)
    labels = torch.from_numpy(y).float()

    return spectra, labels


def list_values(num=0):
    root_dir = Path("/media/sam/data/work/stars/gaia")
    file_directory = root_dir.joinpath(f'{num}.npz')
    data = np.load(file_directory)
    return data.files


def get_random_selection(total_spectra, total_labels, num=15000, seed=42):
    np.random.seed(seed)
    total_indices = np.array(range(total_spectra.shape[0]))
    val_test_set_indices = np.random.choice(total_indices, size=num*2, replace=False)

    val_set_indices = val_test_set_indices[:len(val_test_set_indices)//2]
    test_set_indices = val_test_set_indices[len(val_test_set_indices)//2:]
    train_indices = np.delete(total_indices, val_test_set_indices)

    val_set = total_spectra[val_set_indices]
    val_labels = total_labels[val_set_indices]
    test_set = total_spectra[test_set_indices]
    test_labels = total_labels[test_set_indices]
    train_set = total_spectra[train_indices]
    train_labels = total_labels[train_indices]

    return train_set, train_labels, val_set, val_labels, test_set, test_labels


if __name__ == "__main__":

    # spectra, labels = gen_11may_data()
    # train, train_l, val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels)
    #
    # print(train.shape, train_l.shape)
    # print(val.shape, val_l.shape)
    # print(test.shape, test_l.shape)

    # for i in range(3):
    #     train, train_l, val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels)
    #     print("TEST")
    #     print(train[10700][50], train_l[10700][1])
    #     print(val[0][75], val_l[0][7])
    #     print(test[0][43], test_l[0][10])

    # np.save("/media/sam/data/work/stars/test_sets/11May/train_set.npy", train)
    # np.save("/media/sam/data/work/stars/test_sets/11May/train_set_labels.npy", train_l)
    # np.save("/media/sam/data/work/stars/test_sets/11May/val_set.npy", val)
    # np.save("/media/sam/data/work/stars/test_sets/11May/val_set_labels.npy", val_l)
    # np.save("/media/sam/data/work/stars/test_sets/11May/test_set.npy", test)
    # np.save("/media/sam/data/work/stars/test_sets/11May/test_set_labels.npy", test_l)

    train = np.load("/media/sam/data/work/stars/test_sets/11May/train_set.npy")
    val = np.load("/media/sam/data/work/stars/test_sets/11May/val_set.npy")
    test = np.load("/media/sam/data/work/stars/test_sets/11May/test_set.npy")
    train_labels = np.load("/media/sam/data/work/stars/test_sets/11May/train_set_labels.npy")
    val_labels = np.load("/media/sam/data/work/stars/test_sets/11May/val_set_labels.npy")
    test_labels = np.load("/media/sam/data/work/stars/test_sets/11May/test_set_labels.npy")

    print(train.shape, val.shape, test.shape, train_labels.shape, val_labels.shape, test_labels.shape)
    print(type(train), type(train_labels))
    order = ["v_sin_i", "metal", "alpha", "temp", "log_g", "lumin"]

    for i in range(len(order)):
        for group in [train_labels, val_labels, test_labels]:
            values = group[:, (2*i):(2*i)+2]
            print(order[i], np.max(values))




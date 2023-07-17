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


def gen_17jul_data():
    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']

    root_dir = Path("/media/sam/data/work/stars/gaia")

    # Empty files for creating x, y lists
    list_total_spectra = []
    list_total_labels = []

    # Iterate through dataset files
    for num in range(10):
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
    labels = torch.from_numpy(total_labels).float()

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


def generate_datasets(target_folder_name, num=15000, seed=42):

    spectra, labels = gen_17jul_data()
    train, train_l, val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels, num=num, seed=seed)

    print(train.shape, train_l.shape)
    print(val.shape, val_l.shape)
    print(test.shape, test_l.shape)

    target_folder = f"/media/sam/data/work/stars/test_sets/{target_folder_name}"

    np.save(f"{target_folder_name}/train_set.npy", train)
    np.save(f"{target_folder_name}/train_set_labels.npy", train_l)
    np.save(f"{target_folder_name}/val_set.npy", val)
    np.save(f"{target_folder_name}/val_set_labels.npy", val_l)
    np.save(f"{target_folder_name}/test_set.npy", test)
    np.save(f"{target_folder_name}/test_set_labels.npy", test_l)

    return


def get_values(dataset):

    mins = []
    maxs = []
    means = []
    stds = []

    for i in range(0, 12, 2):

        maxs.append(np.max(dataset[:, i:i+2]))
        mins.append(np.min(dataset[:, i:i+2]))
        means.append(np.mean(dataset[:, i:i+2]))
        stds.append(np.std(dataset[:, i:i+2]))

    maxs = [val for val in maxs for _ in (0, 1)]
    mins = [val for val in mins for _ in (0, 1)]
    means = [val for val in means for _ in (0, 1)]
    stds = [val for val in stds for _ in (0, 1)]

    norm_values = {
        'min': np.array(mins),
        'max': np.array(maxs),
        'mean': np.array(means),
        'std': np.array(stds),
    }

    return norm_values

#
# def get_values(dataset):
#
#     norm_values = {}
#     norm_values["max"] = np.max(dataset, axis=0)
#     norm_values["min"] = np.min(dataset, axis=0)
#     norm_values["mean"] = np.mean(dataset, axis=0)
#     norm_values["std"] = np.std(dataset, axis=0)
#
#     return norm_values
#
# def match_values(norm_values):
#     for i in range(0, 12, 2):
#         norm_values["min"][i:i+1] = np.min(norm_values["min"][i:i+1])
#         norm_values["max"][i:i+1] = np.max(norm_values["max"][i:i+1])
#         norm_values["mean"][i:i+1] = np.mean(norm_values["mean"][i:i+1])
#         norm_values["std"]


def normalize(dataset, mode="range"):

    norm_values = get_values(dataset)

    if mode == "range":
        range = norm_values['max'] - norm_values['min']
        new_dataset = dataset - norm_values['min']
        new_dataset = new_dataset / range

    elif mode == "z":
        new_dataset = dataset - norm_values['std']
        new_dataset = new_dataset / norm_values['mean']

    else:
        raise ValueError("incorrect normalization selected")

    return new_dataset

def reorder(dataset, target):

    if target == "all":
        for j in range(dataset.shape[0]):
            for k in range(0, 12, 2):
                choice = np.argmin(dataset[j, k:k + 2])

                if choice == 0:
                    pass
                else:
                    val1 = dataset[j, k]
                    val2 = dataset[j, k + 1]
                    dataset[j, k] = val2
                    dataset[j, k + 1] = val1

    else:
        if target == "vsini":
            i = 0
        elif target == "metal":
            i = 2
        elif target == "alpha":
            i = 4
        elif target == "temp":
            i = 6
        elif target == "log_g":
            i = 8
        elif target == "lumin":
            i = 10

        for j in range(dataset.shape[0]):
            choice = np.argmin(dataset[j, i:i+2])

            if choice == 0:
                pass
            else:
                for k in range(0, 12, 2):
                    val1 = dataset[j, k]
                    val2 = dataset[j, k+1]
                    dataset[j, k] = val2
                    dataset[j, k+1] = val1

    return dataset


def load_and_preprocess_data(parameters):

    train = np.load(f"{parameters['data_folder']}/train_set.npy")
    val = np.load(f"{parameters['data_folder']}/val_set.npy")
    test = np.load(f"{parameters['data_folder']}/test_set.npy")
    train_labels = np.load(f"{parameters['data_folder']}/train_set_labels.npy")
    val_labels = np.load(f"{parameters['data_folder']}/val_set_labels.npy")
    test_labels = np.load(f"{parameters['data_folder']}/test_set_labels.npy")

    if parameters["normalize"] is not None:
        train_labels = normalize(train_labels, mode=parameters["normalize"])
        val_labels = normalize(val_labels, mode=parameters["normalize"])
        test_labels = normalize(test_labels, mode=parameters["normalize"])

    if parameters["reorder"] is not None:
        if parameters["reorder"] == "all":
            reorder_key = "all"
        else:
            reorder_key = parameters["target_param"]

        train_labels = reorder(train_labels, target=reorder_key)
        val_labels = reorder(val_labels, target=reorder_key)
        test_labels = reorder(test_labels, target=reorder_key)

    return train, train_labels, val, val_labels, test, test_labels


if __name__ == "__main__":


    spectra, labels = gen_17jul_data()
    train, train_l, val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels, num=8500)

    print(train.shape, train_l.shape)
    print(val.shape, val_l.shape)
    print(test.shape, test_l.shape)

    np.save("/media/sam/data/work/stars/test_sets/17jul/train_set.npy", train)
    np.save("/media/sam/data/work/stars/test_sets/17jul/train_set_labels.npy", train_l)
    np.save("/media/sam/data/work/stars/test_sets/17jul/val_set.npy", val)
    np.save("/media/sam/data/work/stars/test_sets/17jul/val_set_labels.npy", val_l)
    np.save("/media/sam/data/work/stars/test_sets/17jul/test_set.npy", test)
    np.save("/media/sam/data/work/stars/test_sets/17jul/test_set_labels.npy", test_l)

    # train = np.load("/media/sam/data/work/stars/test_sets/17jul/train_set.npy")
    # val = np.load("/media/sam/data/work/stars/test_sets/17jul/val_set.npy")
    # test = np.load("/media/sam/data/work/stars/test_sets/17jul/test_set.npy")
    # train_labels = np.load("/media/sam/data/work/stars/test_sets/17jul/train_set_labels.npy")
    # val_labels = np.load("/media/sam/data/work/stars/test_sets/17jul/val_set_labels.npy")
    # test_labels = np.load("/media/sam/data/work/stars/test_sets/17jul/test_set_labels.npy")
    #
    # values = get_values(train_labels)
    # train_new = normalize(train_labels, mode="range")
    # val_new = normalize(val_labels, mode="range")
    # test_new = normalize(test_labels, mode="range")
    #
    # test_new = reorder(test_new, 'all')
    # for func in [np.min, np.max, np.mean, np.std]:
    #     t = func(train_new, axis = 0)
    #     v = func(val_new, axis = 0)
    #     te = func(test_new, axis = 0)
    #     print(func)
    #     print(train_new[0])
    #     print(t-v)
    #     print(t-te)
    # a = get_values(train_labels)
    # b = get_values(val_labels)
    # c = get_values(test_labels)
    # print(train.shape, val.shape, test.shape, train_labels.shape, val_labels.shape, test_labels.shape)
    # print(type(train), type(train_labels))
    #
    # for v in ['min', 'max', 'mean', 'std']:
    #     print(a[v])
    #     print(f'{v}, a-b: ', a[v] - b[v])
    #     print(f'{v}, a-c: ', a[v] - c[v])

    # order = ["v_sin_i", "metal", "alpha", "temp", "log_g", "lumin"]
    #
    # for i in range(len(order)):
    #     for group in [train_labels, val_labels, test_labels]:
    #         values = group[:, (2*i):(2*i)+2]
    #         print(order[i], np.max(values))




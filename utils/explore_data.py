import random

import matplotlib.pyplot as plt
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


def gen_snr_data():
    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']

    root_dir = Path("/media/sam/data/work/stars/test_sets/new_snr_data/")

    for file in root_dir.iterdir():
        data_file = np.load(file)
        all_labels = data_file.files
        print(file)
        print(all_labels)
        for label in all_labels:
            if label != "list_snr" and label != "fi":
                x = data_file[label]
                print(label, len(x), np.mean(x), np.min(x), np.max(x), np.std(x))
                plt.scatter(np.arange(len(x))[:200], x[:200])
                plt.title(f"{file} {label}")
                plt.savefig(f"/media/sam/data/work/stars/test_sets/new_snr_data/exploration/new/{file.stem}{label}.png")
                plt.close()


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

    """

    :param dataset: a Nx12 array of target data. 6 different parameters, with 2 values per target
    :return: norm_values: A dictionary with 4 different keys: min, max, mean and std.
     Each key is a length-12 np. vector of the given key parameters for the six parameters in the dataset
    """

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


def normalize(dataset, mode="range"):
    '''

    :param dataset:  Nx12 array of target data. 6 different parameters, with 2 values per target
    :param mode: Type of normalization - either "range" or "z:
        range: Simply scales all values to [0,1] by subtracting the min value and dividing by the range
        z: Changes dataset to normal gaussian dist by subtract std and dividing by mean

    :return: new_dataset: same dataset, but now normalized
    '''

    # Call get_values to get min, max, mean, and std for each variable
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
    """
    Reorder the dataset to put each two-star pair in min, max order
    """

    if target == "all":
        for j in range(dataset.shape[0]):
            for k in range(0, 12, 2):
                choice = np.argmin(dataset[j, k:k + 2])

                if choice == 0:
                    # If already in min, max order -> do nothing
                    pass
                else:
                    # Flip values if not in correct order
                    val_1 = dataset[j, k]
                    val_2 = dataset[j, k + 1]
                    dataset[j, k] = val_2
                    dataset[j, k + 1] = val_1

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

    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']

    root_dir = Path("/media/sam/data/work/stars/test_sets/snr_data_15MAR23")

    for file in root_dir.iterdir():
        if not file.is_dir():
            data_file = np.load(file)
            all_labels = data_file.files
            print(file)
            print(all_labels)
            for label in all_labels:
                if label != "fi":
                    x = data_file[label]
                    print(label, len(x), np.mean(x), np.min(x), np.max(x), np.std(x))
                    plt.scatter(np.arange(len(x))[:250], x[:250])
                    plt.title(f"{file} {label}")
                    plt.savefig(f"/media/sam/data/work/stars/test_sets/snr_data_15MAR23/explorations/{file.stem}_{label}.png")
                    plt.close()



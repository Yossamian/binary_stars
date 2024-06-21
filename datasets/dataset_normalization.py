import random
import torch
from torch.utils.data import random_split
from binary_stars.datasets import GaiaDataset
from pathlib import Path
import numpy as np

def get_dataset_norm_values(dataset):

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


def normalize(dataset, mode="range", norm_values=None):
    '''

    :param dataset:  Nx12 array of target data. 6 different parameters, with 2 values per target
    :param mode: Type of normalization - either "range" or "z:
        range: Simply scales all values to [0,1] by subtracting the min value and dividing by the range
        z: Changes dataset to normal gaussian dist by subtract std and dividing by mean
    :param norm_values -> If this gets passed , then it allo

    :return: new_dataset: same dataset, but now normalized
    '''

    # Call get_values to get min, max, mean, and std for each variable
    if norm_values is None:
        norm_values = get_dataset_norm_values(dataset)

    extra_data = dataset[:, 12:]
    main_data = dataset[:, :12]

    if mode == "range":
        range = norm_values['max'] - norm_values['min']
        new_dataset = main_data - norm_values['min']
        new_dataset = new_dataset / range

    elif mode == "z":
        new_dataset = main_data - norm_values['std']
        new_dataset = new_dataset / norm_values['mean']

    else:
        raise ValueError("incorrect normalization selected")

    new_dataset = np.column_stack((new_dataset, extra_data))

    return new_dataset, norm_values

def denormalize(dataset, mode="range", norm_values=None):
    '''

    :param dataset:  Nx12 array of target data. 6 different parameters, with 2 values per target
    :param mode: Type of normalization - either "range" or "z:
        range: Simply scales all values to [0,1] by subtracting the min value and dividing by the range
        z: Changes dataset to normal gaussian dist by subtract std and dividing by mean
    :param norm_values -> If this gets passed , then it allo

    :return: new_dataset: same dataset, but now normalized
    '''

    # Call get_values to get min, max, mean, and std for each variable
    if norm_values is None:
        new_dataset = dataset

    else:

        if mode == "range":
            range = norm_values['max'] - norm_values['min']
            new_dataset = dataset * range
            new_dataset = new_dataset + norm_values['min']

        elif mode == "z":
            new_dataset = dataset * norm_values['mean']
            new_dataset = new_dataset + norm_values['std']

    return new_dataset


def normalize_labels(train_labels, val_labels, test_labels, normalize_param):
    if normalize_param is not None:
        train_labels, normalization_values = normalize(train_labels, mode=normalize_param)
        val_labels, _ = normalize(val_labels, mode=normalize_param, norm_values=normalization_values)
        test_labels, _ = normalize(test_labels, mode=normalize_param, norm_values=normalization_values)
    else:
        normalization_values = None
    return train_labels, val_labels, test_labels, normalization_values




if __name__ == "__main__":

    # generate_datasets("06_aug_b", num=15000, seed=42)

    target_folder ="/media/sam/data/work/stars/test_sets/new_snr_data/npy/"
    files_list = ["/media/sam/data/work/stars/test_sets/new_snr_data/0.npz", "/media/sam/data/work/stars/test_sets/new_snr_data/1.npz"]
    spec, labels = gen_snr_data(files_list, append_snr=True)

    np.save(f"{target_folder}/spectra.npy", spec)
    np.save(f"{target_folder}/labels.npy", labels)




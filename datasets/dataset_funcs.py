import random
import torch
from torch.utils.data import random_split
from binary_stars.datasets import GaiaDataset
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def generate_apr24_data():
    """
    For generating just the val and test sets delivered by Avraham, with SNR data
    :return:
    """

    npz_loc = "/media/sam/data/work/stars/new_snr_data/npzs"

    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2',
              'list_p', 'list_t', 'list_v1', 'list_v2', 'list_snr']

    # Empty files for creating x, y lists
    list_total_spectra = []
    list_total_labels = []

    # Iterate through dataset files
    for file_directory in Path(npz_loc).iterdir():
        # Read file
        data = np.load(file_directory)

        # Save spectra, append to list
        file_spectra = data['fi']
        list_total_spectra.append(file_spectra)
        print(f"Adding {file_spectra.shape} spectra array from {file_directory}")

        # Read all relevant y, append to list
        list_labels = [data[label] for label in labels]
        file_labels = np.column_stack(list_labels)
        list_total_labels.append(file_labels)

    # Turn lists into np
    total_spectra = np.row_stack(list_total_spectra)
    total_labels = np.row_stack(list_total_labels)

    print(f"COMPLETE all dataset: {total_spectra.shape} spectra array, {total_labels.shape} label array")

    return total_spectra, total_labels

def generate_val_test_datasets(seed=42):

    """
    splits files into test, val from /media/sam/data/work/stars/new_snr_data/npzs
    :param seed:
    :return:
    """

    spectra, labels = generate_apr24_data()
    val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels, seed=seed)

    print("Val_set sizes: ", val.shape, val_l.shape)
    print("Test set sizes: ", test.shape, test_l.shape)

    target_folder = f"/media/sam/data/work/stars/new_snr_data/test_sets"
    if not Path(target_folder).is_dir():
        Path(target_folder).mkdir()

    np.save(f"{target_folder}/val_set.npy", val)
    np.save(f"{target_folder}/val_set_labels.npy", val_l)
    np.save(f"{target_folder}/test_set.npy", test)
    np.save(f"{target_folder}/test_set_labels.npy", test_l)

    return


def file_explore(filename):

    # When loading npz files

    data = np.load(filename)

    for k in data.files:
        min = np.min(data[k])
        max = np.max(data[k])
        mean = np.mean(data[k])
        std = np.std(data[k])
        unq = len(np.unique(data[k]))
        print(f"{k}, shape: {data[k].shape}, min: {min}, max: {max}, mean: {mean}, unique vals: {unq}, std: {std}")


def get_random_selection(total_spectra, total_labels, seed=42):

    np.random.seed(seed)

    number_of_rows = total_spectra.shape[0]
    val_len = int(number_of_rows / 2)

    total_indices = np.array(range(total_spectra.shape[0]))
    val_set_indices = np.random.choice(total_indices, size=val_len, replace=False)
    test_set_indices = np.delete(total_indices, val_set_indices)

    val_set = total_spectra[val_set_indices]
    val_labels = total_labels[val_set_indices]
    test_set = total_spectra[test_set_indices]
    test_labels = total_labels[test_set_indices]

    return val_set, val_labels, test_set, test_labels


def load_data():
    """

    :param parameters: yaml file containing the data folder with the correct train, val, and set sets
    :return: data, normalized and reordered if desired
    """

    data_folder = "/media/sam/data/work/stars/new_snr_data/test_sets"

    val = np.load(f"{data_folder}/val_set.npy")
    test = np.load(f"{data_folder}/test_set.npy")
    val_labels = np.load(f"{data_folder}/val_set_labels.npy")
    test_labels = np.load(f"{data_folder}/test_set_labels.npy")

    train = np.load(f"/media/sam/data/work/stars/test_sets/06_aug_b/train_set.npy")
    train_labels = np.load(f"/media/sam/data/work/stars/test_sets/06_aug_b/train_set_labels.npy")

    return train, train_labels, val, val_labels, test, test_labels


def pd_df_of_all_data():
    cols = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
            'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2',
            'list_p', 'list_t', 'list_v1', 'list_v2', 'list_snr']
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    spec, labels = generate_apr24_data()
    pd_df = pd.DataFrame(data=labels, columns=cols)

    piv = pd_df.pivot_table(columns=pd.cut(pd_df['list_snr'], bins), aggfunc='size')

    return piv

def get_snr_counts(np_dataset):
    cols = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
            'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2',
            'list_p', 'list_t', 'list_v1', 'list_v2', 'list_snr']
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    pd_df = pd.DataFrame(data=np_dataset, columns=cols)

    bin_labels = []
    for i in range(1, len(bins)):
        bin_labels.append(f"[{bins[i-1]} - {bins[i]})")

    pd_df['snr_group'] = pd.cut(pd_df.list_snr, bins, labels=bin_labels, right=False)
    df = pd_df.reindex().sample(300, random_state=42)

    sns.relplot(data=df, x='list_vsini_1', y='list_vsini_2', hue='snr_group', aspect=1.61)
    plt.show()

    counts = pd_df.pivot_table(columns=pd.cut(pd_df['list_snr'], bins), aggfunc='size')

    return counts

if __name__ == "__main__":

    # generate_val_test_datasets(42)

    train, train_labels, val, val_labels, test, test_labels = load_data()
    # val, val_labels, test, test_labels = load_val_test_data()

    for a in [val_labels, test_labels]:
        counts = get_snr_counts(a)
        print(counts)

    # piv = pd_df_of_all_data()
    # print(piv)



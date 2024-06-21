import random
import torch
from torch.utils.data import random_split
from binary_stars.datasets import GaiaDataset
from pathlib import Path
import numpy as np


def split_dataset(val_test_len, target_param, paths, seed=42):
    full_data = GaiaDataset(paths, target_param=target_param)
    train_len = len(full_data) - (2 * val_test_len)
    train_data, val_data, test_data = random_split(full_data, [train_len, val_test_len, val_test_len], generator=torch.Generator().manual_seed(seed))

    return train_data, val_data, test_data

#
# def gen_17jul_data():
#     labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
#               'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']
#
#     root_dir = Path("/media/sam/data/work/stars/gaia")
#
#     # Empty files for creating x, y lists
#     list_total_spectra = []
#     list_total_labels = []
#
#     # Iterate through dataset files
#     for num in range(10):
#         # Read file
#         file_directory = root_dir.joinpath(f'{num}.npz')
#         data = np.load(file_directory)
#
#         # Save spectra, append to list
#         file_spectra = data['fi']
#         list_total_spectra.append(file_spectra)
#
#         print(f"Adding {file_spectra.shape} spectra array from {num}.npz")
#
#         # Read all relevant y, append to list
#         list_labels = [data[label] for label in labels]
#         file_labels = np.column_stack(list_labels)
#         list_total_labels.append(file_labels)
#
#     # Turn lists into np
#     total_spectra = np.row_stack(list_total_spectra)
#     total_labels = np.row_stack(list_total_labels)
#
#     print(f"COMPLETE all dataset: {total_spectra.shape} spectra array, {total_labels.shape} label array")
#
#     # ensure input data is floats
#     spectra = torch.from_numpy(total_spectra).float()
#     labels = torch.from_numpy(total_labels).float()
#
#     return spectra, labels


def gen_06aug_data(number_files=None):

    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2',
              'list_p', 'list_t', 'list_v1', 'list_v2']

    root_dir = Path("/media/sam/data/work/stars/gaia_aug_23/NEW_SB2_DATA_4_SAM_040823/gaia")

    # Empty files for creating x, y lists
    list_total_spectra = []
    list_total_labels = []

    if number_files is None:
        number_files = len([file for file in root_dir.iterdir()])

    # Iterate through dataset files
    for num in range(number_files):
        # Read file
        file_directory = root_dir.joinpath(f'{num}.npz')
        data = np.load(file_directory)

        # Save spectra, append to list
        file_spectra = data['fi']

        # print("Spectra shape: ", file_spectra.shape)

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

    return total_spectra, total_labels


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

def generate_datasets(target_folder_name, num=15000, seed=42):

    # spectra, labels = gen_17jul_data()
    spectra, labels = gen_06aug_data(number_files=15)
    train, train_l, val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels, num=num, seed=seed)

    print(train.shape, train_l.shape)
    print(val.shape, val_l.shape)
    print(test.shape, test_l.shape)

    target_folder = f"/media/sam/data/work/stars/test_sets/{target_folder_name}"
    if not Path(target_folder).is_dir():
        Path(target_folder).mkdir()

    np.save(f"{target_folder}/train_set.npy", train)
    np.save(f"{target_folder}/train_set_labels.npy", train_l)
    np.save(f"{target_folder}/val_set.npy", val)
    np.save(f"{target_folder}/val_set_labels.npy", val_l)
    np.save(f"{target_folder}/test_set.npy", test)
    np.save(f"{target_folder}/test_set_labels.npy", test_l)

    return

def gen_snr_data(files_list, append_snr=False):

    labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
              'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2',
              'list_p', 'list_t', 'list_v1', 'list_v2']

    # Empty files for creating x, y lists
    list_total_spectra = []
    list_total_labels = []


    # Iterate through dataset files
    for file_directory in files_list:
        # Read file
        data = np.load(file_directory)

        # Save spectra, append to list
        file_spectra = data['fi']

        # print("Spectra shape: ", file_spectra.shape)

        list_total_spectra.append(file_spectra)
        print(f"Adding {file_spectra.shape} spectra array from {file_directory}")

        # Read all relevant y, append to list
        list_labels = [data[label] for label in labels]
        if append_snr:
            snr_value = data["list_snr"]
            snr_array = np.full((len(data["list_t_2"]),), snr_value)
            list_labels.append(snr_array)

        file_labels = np.column_stack(list_labels)

        list_total_labels.append(file_labels)

    # Turn lists into np
    total_spectra = np.row_stack(list_total_spectra)
    total_labels = np.row_stack(list_total_labels)

    print(f"COMPLETE all dataset: {total_spectra.shape} spectra array, {total_labels.shape} label array")

    return total_spectra, total_labels

def file_explore(filename):

    data = np.load(filename)

    for k in data.iterkeys():
        min = np.min(data[k])
        max = np.max(data[k])
        mean = np.mean(data[k])
        std = np.std(data[k])
        unq = len(np.unique(data[k]))
        print(f"{k}, shape: {data[k].shape}, min: {min}, max: {max}, mean: {mean}, unique vals: {unq}")


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

    # spectra, labels = gen_17jul_data()
    spectra, labels = gen_06aug_data(number_files=15)
    train, train_l, val, val_l, test, test_l = get_random_selection(total_spectra=spectra, total_labels=labels, num=num, seed=seed)

    print(train.shape, train_l.shape)
    print(val.shape, val_l.shape)
    print(test.shape, test_l.shape)

    target_folder = f"/media/sam/data/work/stars/test_sets/{target_folder_name}"
    if not Path(target_folder).is_dir():
        Path(target_folder).mkdir()

    np.save(f"{target_folder}/train_set.npy", train)
    np.save(f"{target_folder}/train_set_labels.npy", train_l)
    np.save(f"{target_folder}/val_set.npy", val)
    np.save(f"{target_folder}/val_set_labels.npy", val_l)
    np.save(f"{target_folder}/test_set.npy", test)
    np.save(f"{target_folder}/test_set_labels.npy", test_l)

    return



def load_data(parameters):
    """

    :param parameters: yaml file containing the data folder with the correct train, val, and set sets
    :return: data, normalized and reordered if desired
    """

    train = np.load(f"{parameters['data_folder']}/train_set.npy")
    val = np.load(f"{parameters['data_folder']}/val_set.npy")
    test = np.load(f"{parameters['data_folder']}/test_set.npy")
    train_labels = np.load(f"{parameters['data_folder']}/train_set_labels.npy")
    val_labels = np.load(f"{parameters['data_folder']}/val_set_labels.npy")
    test_labels = np.load(f"{parameters['data_folder']}/test_set_labels.npy")

    return train, train_labels, val, val_labels, test, test_labels


if __name__ == "__main__":

    # generate_datasets("06_aug_b", num=15000, seed=42)
    #
    # target_folder ="/media/sam/data/work/stars/test_sets/new_snr_data/npy/"
    # files_list = ["/media/sam/data/work/stars/test_sets/new_snr_data/0.npz", "/media/sam/data/work/stars/test_sets/new_snr_data/1.npz"]
    # spec, labels = gen_snr_data(files_list, append_snr=True)
    # np.save(f"{target_folder}/spectra.npy", spec)
    # np.save(f"{target_folder}/labels.npy", labels)

    generate_apr24_data()




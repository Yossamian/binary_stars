import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader_experimental import DataLoader2
from torchvision import transforms, utils
from torch.utils.data import random_split
from pathlib import Path


def explore():
    root_dir = Path("/media/sam/data/work/stars/gaia")
    paths = range(13)
    for num in paths:
        # Read file
        file_directory = root_dir.joinpath(f'{num}.npz')
        data = np.load(file_directory)


class GaiaDataset(Dataset):

    # load the dataset
    def __init__(self, paths, target_param):

        labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2',
                  'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']

        if target_param == "all":
            labels = labels
        elif target_param == "vsini":
            labels = labels[:2]
        elif target_param == "metal":
            labels = labels[2:4]
        elif target_param == "alpha":
            labels = labels[4:6]
        elif target_param == "temp":
            labels = labels[6:8]
        elif target_param == "log_g":
            labels = labels[8:10]
        elif target_param == "lumin":
            labels = labels[10:12]
        else:
            raise ValueError("Incorrect target_param selection for creation of Gaia Dataset")

        root_dir = Path("/media/sam/data/work/stars/gaia")

        # Empty files for creating x, y lists
        list_total_spectra = []
        list_total_labels = []

        # Iterate through dataset files
        for num in paths:
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

        print(f"COMPLETE {target_param} dataset: {total_spectra.shape} spectra array, {total_labels.shape} label array")

        # ensure input data is floats
        self.spectra = torch.from_numpy(total_spectra).float()

        # For new order:
        new_order = []
        for i in range(len(labels)):
            if i % 2 == 0:
                new_order.append(i + 1)
            else:
                new_order.append(i - 1)

        y_alt = total_labels[:, new_order]
        y = np.stack((total_labels, y_alt), axis=-1)
        self.labels = torch.from_numpy(y).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.spectra)

    # get a row at an index
    def __getitem__(self, idx):
        return (self.spectra[idx], self.labels[idx])


class GaiaDataset2(Dataset):

    # load the dataset
    def __init__(self, dataset, dataset_labels, no_reverse=False):

        self.spectra = dataset
        self.labels = dataset_labels

        if no_reverse:
            self.labels = dataset_labels[:, :, 0]

    # number of rows in the dataset
    def __len__(self):
        return len(self.spectra)

    # get a row at an index
    def __getitem__(self, idx):
        # SPECTRA IS shape Bx903
        # Labels are shape Bx(No labels)x2
        return self.spectra[idx], self.labels[idx]


if __name__ == "__main__":
    # opts = ['all', 'v_sin_i', 'metal', 'alpha', 'temp', 'log_g', 'lumin', 'none']
    # for opt in opts:
    #     dataset = GaiaDataset2(paths=range(15), target_param=opt)
    #     a = dataset.spectra
    #     lab = dataset.labels
    #     print(f'************{opt}*************')
    #     print(a.shape, lab.shape)
    #     print(torch.min(lab))
    #     print(torch.max(lab))
    #     print()
    train = np.load("/media/sam/data/work/stars/test_sets/11May/train_set.npy")
    val = np.load("/media/sam/data/work/stars/test_sets/11May/val_set.npy")
    test = np.load("/media/sam/data/work/stars/test_sets/11May/test_set.npy")
    train_labels = np.load("/media/sam/data/work/stars/test_sets/11May/train_set_labels.npy")
    val_labels = np.load("/media/sam/data/work/stars/test_sets/11May/val_set_labels.npy")
    test_labels = np.load("/media/sam/data/work/stars/test_sets/11May/test_set_labels.npy")

    # print("Dataset has been split")
    # torch.save(test_data, f"/media/sam/data/work/stars/test_sets/{parameters['target_param']}_test_set")
    train_data = GaiaDataset2(dataset=train, dataset_labels=train_labels)
    val_data = GaiaDataset2(dataset=val, dataset_labels=val_labels)
    test_data = GaiaDataset2(dataset=test, dataset_labels=test_labels)
    print("g")

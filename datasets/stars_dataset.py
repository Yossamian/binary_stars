import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader_experimental import DataLoader2
from torchvision import transforms, utils
from torch.utils.data import random_split
from pathlib import Path


class GaiaDataset(Dataset):

    # load the dataset
    def __init__(self, dataset, dataset_labels, no_reverse=True):

        self.spectra = dataset
        self.labels = dataset_labels

        # if no_reverse:
        #     self.labels = dataset_labels[:, :, 0]

    # number of rows in the dataset
    def __len__(self):
        return len(self.spectra)

    # get a row at an index
    def __getitem__(self, idx):
        # SPECTRA IS shape (903,) -> Bx903
        # Labels are shape (12,) -> Bx12 if no_reverse=True, (12, 2) Bx12x2 if no_reverse=False
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
    # train_data = GaiaDataset(dataset=train, dataset_labels=train_labels)
    # val_data = GaiaDataset(dataset=val, dataset_labels=val_labels)
    # test_data = GaiaDataset(dataset=test, dataset_labels=test_labels)
    print("g")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader_experimental import DataLoader2
from torchvision import transforms, utils
from torch.utils.data import random_split
from pathlib import Path

from mlxtend.preprocessing import standardize


class StarDataset_sim_gaia(Dataset):
    # load the dataset
    def __init__(self, paths):
        
        root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/sim_gaia")

        #Empty files for creating x, y lists
        list_total_X=[]
        list_total_y=[]

        # Iterate through dataset files
        for num in paths:

          # Read file
          x_dir = root_dir.joinpath(f'spectra/spectra_{num}.npy')
          data_x = np.load(x_dir)
          print(f"Adding {data_x.shape} spectra array from spectra_{num}.npy")

          # Save X, append to list
          list_total_X.append(data_x)

          # Read all relevant y, append to list
          y_dir = root_dir.joinpath(f'labels/labels_{num}.npy')
          data_y = np.load(y_dir)
          print(f"Adding {data_y.shape} labels array from labels_{num}.npy")

          list_total_y.append(data_y)

        # Turn lists into np
        total_X = np.row_stack(list_total_X)
        total_y = np.row_stack(list_total_y)

        print (f"COMPLETE dataset: {total_X.shape} spectra array, {total_y.shape} label array")

        # ensure input data is floats
        self.x = torch.from_numpy(total_X).float()

        # For new order:
        new_order = []
        for i in range(12):
          if i%2==0:
            new_order.append(i+1)
          else:
            new_order.append(i-1)
        
        y_alt = total_y[:,new_order]
        y = np.stack((total_y, y_alt), axis=-1)
        self.y=torch.from_numpy(y).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class StarDataset_original(Dataset):
    # load the dataset
    def __init__(self, paths):
        
        root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/")

        #Empty files for creating x, y lists
        list_total_X=[]
        list_total_y=[]

        # Iterate through dataset files
        for num in paths:

          # Read file
          x_dir = root_dir.joinpath(f'{num}.npz')
          data = np.load(x_dir)

          # Save X, append to list
          file_X = data['fi']
          list_total_X.append(file_X)

          # Read all relevant y, append to list
          y_dir = root_dir.joinpath(f'processed/standardized/s_{num}.npy')
          file_y = np.load(y_dir)
          list_total_y.append(file_y)

        # Turn lists into np
        total_X = np.row_stack(list_total_X)
        total_y = np.row_stack(list_total_y)

        # ensure input data is floats
        self.x = torch.from_numpy(total_X).float()

        # For new order:
        new_order = []
        for i in range(12):
          if i%2==0:
            new_order.append(i+1)
          else:
            new_order.append(i-1)
        
        y_alt = total_y[:,new_order]
        y = np.stack((total_y, y_alt), axis=-1)
        self.y=torch.from_numpy(y).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

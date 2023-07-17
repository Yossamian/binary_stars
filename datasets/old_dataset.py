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


class GaiaDataset_old(Dataset):

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



class StarDataset_original(Dataset):
    # load the dataset
    def __init__(self, paths):

        root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/")

        # Empty files for creating x, y lists
        list_total_X = []
        list_total_y = []

        # Iterate through dataset files
        for num in paths:
            # Read file
            x_dir = root_dir.joinpath(f'{num}.npy')
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
            if i % 2 == 0:
                new_order.append(i + 1)
            else:
                new_order.append(i - 1)

        y_alt = total_y[:, new_order]
        y = np.stack((total_y, y_alt), axis=-1)
        self.y = torch.from_numpy(y).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.x)

    # get a row at an index
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class BasicDataset(Dataset):
    # This is the original
    # Loads the full input chunk (~10,000 rows) and target chunk (standardized)
    # This makes for easy management... however, multiple datasets/dataloaders need to be created during training
    def __init__(self, number):
        root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/")
        self.x_dir = root_dir.joinpath(f'{number}.npz')
        self.y_dir = root_dir.joinpath(f'processed/standardized/s_{number}.npy')

        data = np.load(self.x_dir)
        x = data['fi']
        self.x = torch.from_numpy(x).float()

        y_initial = np.load(self.y_dir)

        # For new order:
        new_order = []
        for i in range(12):
            if i% 2 == 0:
                new_order.append(i + 1)
            else:
                new_order.append(i - 1)

                # Add reverse order for
        y_alt = y_initial[:, new_order]
        y = np.stack((y_initial, y_alt), axis=-1)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class BestDataset(Dataset):
    # This dataset loads single rows of data, that were broken u
    # Goal was to make processing fast... but it did not work
    def __init__(self):
        root_dir = Path(
            "/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/processed/standardized")
        self.x_dir = root_dir.joinpath('spectra')
        self.y_dir = root_dir.joinpath('outputs')

        # Index our dataset
        rdbms = []
        for a in self.y_dir.iterdir():
            rdbms.append(a.name[3:])

        self.locations = np.array(rdbms)

    def __len__(self):
        return self.locations.shape[0]

    def __getitem__(self, idx):
        name = self.locations[idx]
        x_file = self.x_dir.joinpath(f's_x{name}')
        y_file = self.y_dir.joinpath(f's_y{name}')
        x = np.load(x_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)
        data = (x, y)
        sources = (x_file.name, y_file.name)
        return data, sources


class BestDataset_v1(Dataset):
    # This dataset uses 100 line chunks to load all of the data
    # For _get_item_, it returns a single row of data (using sub_indexing)
    def __init__(self):
        root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/processed")
        self.x_dir = root_dir.joinpath('spectra')
        self.y_dir = root_dir.joinpath('labels')

        # Index our dataset
        i = 0
        rdbs = []
        for chunk in self.y_dir.iterdir():
            y = np.load(chunk, allow_pickle=True)
            for j in range(y.shape[0]):
                row = [i, chunk.name[6:], j]
                rdbs.append(row)
                i += 1

        self.locations = np.array(rdbs)

    def __len__(self):
        return self.locations.shape[0]

    def __getitem__(self, idx):
        _, name, sub_idx = self.locations[idx]
        sub_idx = int(sub_idx)
        x_file = self.x_dir.joinpath(f'spectra{name}')
        y_file = self.y_dir.joinpath(f'labels{name}')
        x = np.load(x_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)
        data = (x[sub_idx], y[sub_idx])
        sources = (x_file.name, y_file.name, sub_idx)
        return data, sources


class BestDataset_v2(Dataset):
    # This dataset uses 100 line chunks to load all of the data
    # For _get_item_, it returns the full chunk of data
    def __init__(self):
        root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data/processed")
        self.x_dir = root_dir.joinpath('spectra')
        self.y_dir = root_dir.joinpath('labels')

        # Index our dataset
        i = 0
        rdbs = []
        for chunk in self.y_dir.iterdir():
            row = [i, chunk.name[6:]]
            rdbs.append(row)
            i += 1

        # For new order:
        new_order = []
        for i in range(12):
            if i % 2 == 0:
                new_order.append(i + 1)
            else:
                new_order.append(i - 1)

        self.new_order = new_order

        self.locations = np.array(rdbs)

    def __len__(self):
        return self.locations.shape[0]

    def __getitem__(self, idx):
        _, name = self.locations[idx]
        x_file = self.x_dir.joinpath(f'spectra{name}')
        y_file = self.y_dir.joinpath(f'labels{name}')
        x = np.load(x_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)

        # Add reverse order for
        y_alt = y[:, self.new_order]
        y = np.stack((y, y_alt), axis=-1)

        data = (x, y)
        sources = (x_file.name, y_file.name)
        return data
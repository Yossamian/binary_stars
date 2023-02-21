import torch
from torch.utils.data import random_split, DataLoader
from trainer_gaia import Trainer
import torch.nn as nn
import torch.nn.functional as F
from datasets import GaiaDataset, split_dataset
# from models.models_original import ConvolutionalNet
import models
import yaml
from pathlib import Path


def main(config_loc, experiment_name=None):

    print(f'Running main() for {config_loc.name}')

    # Read in parameters
    with open(config_loc, 'r') as f:
        parameters = yaml.safe_load(f)

    print('PARAMETERS:')
    for key, value in parameters.items():
        print(key, value)

    if parameters['target_param'] == "all":
        num_outputs = 12
    else:
        num_outputs = 2

    # Load model
    model = getattr(models, parameters['model'])(num_outputs=num_outputs)
    print('MODEL ARCHITECTURE:')
    print(model)

    # Create datasets and dataloaders with given # of path files
    # num_sets = 2
    # paths = list(range(num_sets))
    paths = list(range(13))
    num_sets = len(paths)
    train_data, val_data, test_data = split_dataset(train=0.8, val=0.1, target_param=parameters['target_param'], paths=paths)
    # full_data = GaiaDataset(paths, target_param=parameters['target_param'])
    # train_len = int(len(full_data) * 0.75)
    # val_len = len(full_data) - train_len
    # train_data, val_data = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    print(f'Dataset created with {len(train_data)} training examples, {len(val_data)} val examples, and {len(test_data)} test examples')

    # Create Dataloaders
    train_loader = DataLoader(train_data, shuffle=False, batch_size=256)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=256)

    trainer = Trainer(
        model=model,
        parameters=parameters,
        num_sets=num_sets,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name
    )

    trainer.train()


if __name__ == "__main__":

    root_config = Path('/media/sam/data/work/stars/configurations/config_loc')
    # config_loc = root_config.joinpath('basic.yaml')

    config_loc = next(root_config.joinpath('config_start').iterdir())
    main(config_loc, experiment_name=None)

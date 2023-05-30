import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
# from trainer_gaia import Trainer
from trainer2023 import Trainer
import torch.nn as nn
import torch.nn.functional as F
from datasets import GaiaDataset, split_dataset, split_dataset_new, GaiaDataset2
# from models.models_original import ConvolutionalNet
import models
import yaml
from pathlib import Path


def main(config_loc, experiment_name=None, selection="train"):

    print(f'Running main() for {config_loc}')

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
    # paths = list(range(parameters['num_sets']))
    # num_sets = len(paths)
    # # train_data, val_data, test_data = split_dataset(train=0.8, val=0.1, target_param=parameters['target_param'], paths=paths)
    # train_data, val_data, test_data = split_dataset_new(val_test_len=15000, target_param=parameters['target_param'], paths=paths)

    train = np.load("/media/sam/data/work/stars/test_sets/11May/train_set.npy")
    val = np.load("/media/sam/data/work/stars/test_sets/11May/val_set.npy")
    test = np.load("/media/sam/data/work/stars/test_sets/11May/test_set.npy")
    train_labels = np.load("/media/sam/data/work/stars/test_sets/11May/train_set_labels.npy")
    val_labels = np.load("/media/sam/data/work/stars/test_sets/11May/val_set_labels.npy")
    test_labels = np.load("/media/sam/data/work/stars/test_sets/11May/test_set_labels.npy")

    # print("Dataset has been split")
    # torch.save(test_data, f"/media/sam/data/work/stars/test_sets/{parameters['target_param']}_test_set")
    train_data = GaiaDataset2(dataset=train, dataset_labels=train_labels, no_reverse=True)
    val_data = GaiaDataset2(dataset=val, dataset_labels=val_labels, no_reverse=True)
    test_data = GaiaDataset2(dataset=test, dataset_labels=test_labels, no_reverse=True)

    print(f'Dataset created with {len(train_data)} training examples, {len(val_data)} val examples, and {len(test_data)} test examples')

    # Create Dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=512, drop_last=True)
    # train_loader = DataLoader(train_data, shuffle=False, num_workers=8, batch_size=64)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=256)
    # val_loader = DataLoader(val_data, shuffle=True, num_workers=8, batch_size=64)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=256)
    # test_loader = DataLoader(test_data, shuffle=True, num_workers=8, batch_size=64)

    trainer = Trainer(
        model=model,
        parameters=parameters,
        # num_sets=num_sets,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        experiment_name=experiment_name,
        parallel=True
    )

    if selection == "train":
        trainer.train()
    elif selection == "test":
        trainer.test()
    else:
        raise ValueError("Must select either 'train' or 'test'")


if __name__ == "__main__":

    root_config = Path('/media/sam/data/work/stars/configurations/config_loc')
    # config_loc = next(root_config.joinpath('config_start').iterdir())

    # main(config_loc, experiment_name=None)

    config_loc = "/media/sam/data/work/stars/configurations/saved_models/DenseNet_temp_BootlegMSE_2023_05_29_1279/" + "config.yaml"
    main(config_loc, experiment_name=None, selection="test")
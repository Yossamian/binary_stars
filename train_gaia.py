import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from trainer2023 import Trainer
from datasets import GaiaDataset, load_data, normalize_labels, reorder_labels
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

    num_outputs = 12 if parameters['target_param'] == "all" else 2

    # Load model
    model = getattr(models, parameters['model'])(num_outputs=num_outputs)

    # Loads data from .npy files, reorder & normalize as necessary
    train, train_labels, val, val_labels, test, test_labels = load_data()
    train_labels, val_labels, test_labels, normalization_values = normalize_labels(train_labels, val_labels, test_labels, parameters["normalize"])
    train_labels, val_labels, test_labels = reorder_labels(train_labels, val_labels, test_labels, parameters)

    # Create Datasets, Dataloaders
    train_data = GaiaDataset(dataset=train, dataset_labels=train_labels)
    val_data = GaiaDataset(dataset=val, dataset_labels=val_labels)
    test_data = GaiaDataset(dataset=test, dataset_labels=test_labels)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=512, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=256)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=256)
    print(f'Dataset created with {len(train_data)} training examples, {len(val_data)} val examples, and {len(test_data)} test examples')

    trainer = Trainer(
        model=model,
        parameters=parameters,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        normalization_values=normalization_values,
        experiment_name=experiment_name,
        parallel=True
    )

    if selection == "train":
        trainer.train()
        trainer.test()
    elif selection == "test":
        trainer.test()
    else:
        raise ValueError("Must select either 'train' or 'test'")


if __name__ == "__main__":

    root_config = Path('/media/sam/data/work/stars/configurations/config_loc')

    config_loc = "/media/sam/data/work/stars/configurations/saved_models/aug_07/DenseNet_temp_Huber_2023_08_06_97727/config.yaml"
    for folder in Path("/media/sam/data/work/stars/configurations/saved_models/11FEB_stars_huber/").iterdir():
        config_loc = folder.joinpath("config.yaml")
        with open(config_loc, 'r') as f:
            parameters = yaml.safe_load(f)
        # if parameters["loss"] == "Huber":
        #     main(config_loc, experiment_name="11FEB_stars_huber", selection="test")
        main(config_loc, experiment_name="22May_stars_huber", selection="test")
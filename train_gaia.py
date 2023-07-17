import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
# from trainer_gaia import Trainer
from trainer2023 import Trainer
from datasets import GaiaDataset, load_and_preprocess_data
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

    train, train_labels, val, val_labels, test, test_labels = load_and_preprocess_data(parameters)

    train_data = GaiaDataset(dataset=train, dataset_labels=train_labels, no_reverse=True)
    val_data = GaiaDataset(dataset=val, dataset_labels=val_labels, no_reverse=True)
    test_data = GaiaDataset(dataset=test, dataset_labels=test_labels, no_reverse=True)

    print(f'Dataset created with {len(train_data)} training examples, {len(val_data)} val examples, and {len(test_data)} test examples')

    # Create Dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=512, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=256)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=256)

    trainer = Trainer(
        model=model,
        parameters=parameters,
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
    config_loc = next(root_config.joinpath('config_start').iterdir())

    main(config_loc, experiment_name=None)

    # option_dict2 = {'temp': "DenseNet_temp_BootlegMSE_2023_05_30_2832",
    #                 "log_g": "DenseNet_log_g_BootlegMSE_2023_05_30_861",
    #                 "metal": "DenseNet_metal_BootlegMSE_2023_05_30_2997",
    #                 "alpha": "DenseNet_alpha_BootlegMSE_2023_05_30_1234",
    #                 "vsini": "DenseNet_vsini_BootlegMSE_2023_05_31_345",
    #                 "lumin": "DenseNet_lumin_BootlegMSE_2023_05_30_1276",
    #                 }
    #
    # for key, value in option_dict2.items():
    #     config_loc = f"/media/sam/data/work/stars/configurations/saved_models/{value}/" + "config.yaml"
    #     main(config_loc, experiment_name=None, selection="test")
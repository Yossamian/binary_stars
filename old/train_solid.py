import torch
from torch.utils.data import random_split, DataLoader
from trainer_solid import Trainer
import torch.nn as nn
import torch.nn.functional as F
from datasets.stars import StarDataset_original
# from models.models_original import ConvolutionalNet
import models
import yaml
from pathlib import Path

def main(config_loc):

  print(f'Running main() for {config_loc.name}')

  with open(config_loc, 'r') as f:
    parameters = yaml.safe_load(f)

  print('PARAMETERS:')
  for key, value in parameters.items():
    print(key, value)
  
  model = getattr(models, parameters['model'])()
  print('MODEL ARCHITECTURE:')
  print(model)

  # Create datasets and dataloaders with given # of path files
  num_sets = 2
  paths = list(range(num_sets))
  full_data = StarDataset_original(paths)
  train_len = int(len(full_data)*0.8)
  val_len = len(full_data) - train_len
  train_data, val_data = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(42))
  print(f'Dataset created with {len(train_data)} training examples and {len(val_data)} val examples')
  train_loader = DataLoader(train_data, batch_size=16)
  val_loader = DataLoader(val_data, batch_size=16)

  trainer = Trainer(
    model=model,
    parameters=parameters,
    num_sets = num_sets,
    train_loader = train_loader,
    val_loader=val_loader,
    experiment_name=None
  )

  trainer.train()


if __name__=="__main__":
  root_config = Path('/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/config_loc/config_start')
  config_loc = root_config.joinpath('basic.yaml')
  main(config_loc)

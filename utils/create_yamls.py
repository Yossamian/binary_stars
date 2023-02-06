import yaml
from pathlib import Path
from itertools import product

def create_yaml(parameters, folder):

  model = parameters['model']
  loss = parameters['loss']
  opt = parameters['optimizer']
  lr = parameters['lr']

  name = f'{model}_{loss}_{opt}_{lr}'
  parameters['name'] = name

  file_name = folder.joinpath(f'{name}.yaml')

  file_name = file_check(file_name)

  with open(file_name, 'w') as f:
    yaml.dump(parameters, f)

def folder_check(*args):
  for folder in args:
    if not folder.exists():
      folder.mkdir()

def file_check(filename):
  root_dir = filename.parent
  i=2

  if filename.exists():
    stem = filename.stem
    suffix = filename.suffix
    new_name = f'{stem}_v{i}{suffix}'
    filename = root_dir.joinpath(new_name)

    while filename.exists():
      remove_end = len(str(i))
      stem = filename.stem[:-remove_end]
      suffix = filename.suffix
      i += 1
      new_name = f'{stem}{i}{suffix}'
      filename = root_dir.joinpath(new_name)
  
  return filename


def multi_yaml():
  root = Path('/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/configurations')
  config_loc = root.joinpath('config_loc')
  config_start = config_loc.joinpath('config_start')
  config_finish = config_loc.joinpath('config_finish')
  folder_check(config_loc, config_start, config_finish)

  # all_parameters={'model': ['ConvolutionalNet', 'AttentionBlockModel'],
  #             'optimizer': ['Adam'],
  #             'loss': ['OA_mse_loss', 'OA_mae_loss'],
  #             'lr': [.01, .001, .0001],
  #             'wd': [0, 0.001, 0.0001],
  #             'epochs': [30],
  #             'early_stopping': [6]
  #             }

  all_parameters={'model': ['DenseNet', 'ConvolutionalNet', 'InceptionNet'],
            'optimizer': ['Adam'],
            'loss': ['MAPE_adjusted', 'SMAPE_adjusted', 'MASE'],
            'lr': [.0001],
            'wd': [0.0001],
            'epochs': [200],
            'early_stopping': [10]
            }
  
  for combo in product(*all_parameters.values()):
    params = dict(zip(all_parameters.keys(), combo))
    # print(params)
    create_yaml(params, config_start)


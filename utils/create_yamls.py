import yaml
from pathlib import Path
from itertools import product
from datetime import date
from random import randint


def create_yaml(parameters, folder):

    model = parameters['model']
    loss = parameters['loss']
    target = parameters['target_param']

    today = date.today().strftime('%Y_%m_%d')
    name = f'{model}_{target}_{loss}_{today}_{randint(0,3000)}'
    parameters['name'] = name

    file_name = folder.joinpath(f'{name}.yaml')

    # file_name = file_check(file_name)

    with open(file_name, 'w') as f:
        yaml.dump(parameters, f)


def folder_check(*args):
    for folder in args:
        if not folder.exists():
            folder.mkdir()


def file_check(filename):
    root_dir = filename.parent
    i = 2

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
    root = Path('/media/sam/data/work/stars/configurations/')
    config_loc = root.joinpath('config_loc')
    config_start = config_loc.joinpath('config_start')
    config_finish = config_loc.joinpath('config_finish')
    folder_check(config_loc, config_start, config_finish)

    all_parameters = {'model': ['DenseNet'],  # ['DenseNet', 'ConvolutionalNet', 'InceptionNet', 'DenseNetMultiHead'],
                      'optimizer': ['Adam'],  # 'Adam', 'SGD'
                      'loss': ['BootlegMSE'],   #['MSE', 'MAPE_adjusted', 'SMAPE_adjusted', 'MASE', 'MAE'], USE MASE!!!!!!!!!
                      'lr': [.01],
                      'wd': [0],
                      'epochs': [50],
                      'early_stopping': [50],
                      'sets_between_eval': [3],
                      'optimizer_step': [150],
                      'optimizer_gamma': [0.1],
                      'target_param': ['temp', 'metal', 'alpha', 'log_g', 'vsini', 'lumin'],  # 'temp', 'metal', 'alpha', 'log_g', 'vsini', 'lumin'
                      'patch_size': [30],
                      'num_sets': [22],
                      'momentum': [0.9],
                      'normalize': ['range', 'z', None],  # range, z, None
                      'reorder': ['target'],  # target, all, None
                      'data_folder': ["/media/sam/data/work/stars/test_sets/17jul"]


                      }

    for combo in product(*all_parameters.values()):
        params = dict(zip(all_parameters.keys(), combo))
        # print(params)
        create_yaml(params, config_start)


if __name__ == "__main__":
  multi_yaml()

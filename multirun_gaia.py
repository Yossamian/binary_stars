import shutil
from pathlib import Path
from train_gaia import main
import yaml
from datetime import date


def multirun(experiment_name=None):
    root_config = Path('/media/sam/data/work/stars/configurations/config_loc')
    config_finish = root_config.joinpath('config_finish')
    config_start = root_config.joinpath('config_start')

    for config_loc in config_start.iterdir():

        filename = config_loc.name

        main(config_loc, experiment_name=f'{date.today().strftime("%m_%d")}_{experiment_name}')
        shutil.move(config_loc, config_finish.joinpath(filename))


if __name__ == "__main__":
    multirun(experiment_name="stars_new")
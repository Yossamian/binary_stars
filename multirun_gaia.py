import shutil
from pathlib import Path
from train_gaia import main
import yaml


def multirun(experiment_name=None):
    root_config = Path('/media/sam/data/work/stars//configurations/config_loc')
    config_finish = root_config.joinpath('config_finish')
    config_start = root_config.joinpath('config_start')
    config_fail = root_config.joinpath('config_fail')

    for config_loc in config_start.iterdir():

        # Read in parameters
        with open(config_loc, 'r') as f:
            parameters = yaml.safe_load(f)

        filename = config_loc.name

        # try:
        main(config_loc, experiment_name=f'{experiment_name}')
        shutil.move(config_loc, config_finish.joinpath(filename))

        # except:
        #     print('*********************************')
        #     print(f"main() failed on {config_loc.name}")
        #     shutil.move(config_loc, config_fail.joinpath(filename))
        #     print('*********************************')


if __name__ == "__main__":
    multirun(experiment_name="binary_stars_21_feb")
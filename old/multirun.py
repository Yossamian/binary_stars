import shutil
from pathlib import Path
from train_solid import main

def multirun():
  root_config = Path('/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/config_loc')
  config_finish = root_config.joinpath('config_finish')
  config_start = root_config.joinpath('config_start')
  config_fail = root_config.joinpath('config_fail')

  for config_loc in config_start.iterdir():

    filename = config_loc.name

    try:
      main(config_loc)
      shutil.move(config_loc, config_finish.joinpath(filename))
      
    except:
      print('*********************************')
      print(f"main() failed on {config_loc.name}")
      shutil.move(config_loc, config_fail.joinpath(filename))
      print('*********************************')


import numpy as np
from pathlib import Path
from mlxtend.preprocessing import standardize
import pandas as pd
import torch


### Change single .npz file to two gaia-format files 
def gaiafy(file_number):

  root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")
  # Load npz file
  path=root_dir.joinpath(f'{file_number}.npz')
  data = np.load(path)

  # Resample to go from 10197 -> 973 samples in freq domain
  fi = resample_naive(data['fi'])

  # Cycle through labels
  labels = data.files[1:]
  fields = []
  for label in labels:
    field = data[label]
    if "_l_" in label:
      field = np.log10(field)
    fields.append(field)
  
  z = np.array(fields).T

  return fi, z
      

def create_gaias():
  root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")

  new_X_direc = root_dir.joinpath('sim_gaia').joinpath('spectra')
  new_y_direc = root_dir.joinpath('sim_gaia').joinpath('labels')

  for i in range(10):
    new_fi, new_labels = gaiafy(i)
    print(f'Saving {i}.npz as {new_fi.shape} spectrum file and {new_labels.shape} label file')
    np.save(new_X_direc.joinpath(f'spectra_{i}.npy'), new_fi)
    np.save(new_y_direc.joinpath(f'labels_{i}.npy'), new_labels)

  return



### Naive resampling of frequency signal
def resample_naive(x):
  # TO reduce from (10193,:) to (973,:)
  # Simply takes every tenth value to reduce to (1020, :)
  # Then only takes middle portion of the signal
  new = x[:, ::10]
  new = new[:, 23:-24]
  return new


######## Load the SINGLE data file
def create_new_npy_chunks(size=100):
  # Creates separate .npy "chunks" of data in a given size
  # Inputs: desired size (in number of samples/rows) of chunks
  # Returns: none, just creates files and prints completion 
  root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")

  numbers = list(range(10))
  # Iterate through dataset files
  for file_number in numbers:
    path=root_dir.joinpath(f'{file_number}.npz')
    data = np.load(path)
    x = data['fi']
    print(type(x), x.shape)
    total_samples = x.shape[0]
    labels = data.files[1:]
    y = [data[g] for g in labels]
    z=np.array(y).T
    print(type(z), z.shape)

    new_X_direc = root_dir.joinpath('processed').joinpath('spectra')
    new_y_direc = root_dir.joinpath('processed').joinpath('labels')

    ######### Make chunks
    start = 0
    done = False
    chunks = []
    while not done:
      finish = start+size
      if finish >= total_samples:
        finish = total_samples
        done = True
      chunk = list(range(start, finish))
      chunks.append(chunk)
      start=finish

    print(f'total: {total_samples} and chunks {len(chunks)}')

    # Actually save the files
    for num, chunk in enumerate(chunks):
      x_chunk = x[chunk]
      z_chunk = z[chunk]

      np.save(new_X_direc.joinpath(f'spectra_{file_number}_{num}.npy'), x_chunk)
      np.save(new_y_direc.joinpath(f'labels_{file_number}_{num}.npy'), z_chunk)
    
    print(f"****Created {len(chunks)} files for the {total_samples} spectra in file number {file_number}******")



def get_normalization_info(list_files=list(range(10))):
  # Gets the scale parameters (avg and std dev) for all columns in a dataset
  # Input: Number of files 
  # Return: The avgs and stddevs, in two different formats
  # Note: Saved .npy is saved at "project_954437307_066610346/saved_states/scale_params_full.npy"

  root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")
 
  # Create large list of all target values, unnormalized
  list_total=[]
  for file_number in list_files:
    path=root_dir.joinpath(f'{file_number}.npz')
    data = np.load(path)
    labels = data.files[1:]
    y = [data[g] for g in labels]
    z = np.array(y).T
    list_total.append(z)

  # Turn list into array of target values, unnormalized
  final = np.vstack(list_total)
  print(final.shape)

  # Returns, normalized array, and scale params
  final_new, scale_params = standardize(final, columns=np.array(range(final.shape[1])), return_params=True)

  avgs = scale_params['avgs']
  std_dev = scale_params['stds']

  # Reformat scale parameters to meet data - only 6 real pairs!
  scale_factors_tuples = []
  new_avgs = []
  new_stddevs=[]
  for i in list(range(0,12,2)):
    avg = np.average(avgs[i:i+2])
    stddev = np.average(std_dev[i:i+2])
    scale_factors_tuples.append((avg, stddev))

    new_avgs.append(avg)
    new_avgs.append(avg)

    new_stddevs.append(stddev)
    new_stddevs.append(stddev)

  scale_factors_correct = {'avgs':torch.tensor(new_avgs), 'stds': torch.tensor(new_stddevs)}

  return scale_factors_tuples, scale_factors_correct



def create_new_full_npys(scale_params, numbers = list(range(10))):
  # Creates full_size target files, STANDARDIZED

  root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")
  
  # Iterate through dataset files
  for file_number in numbers:
    path=root_dir.joinpath(f'{file_number}.npz')
    data = np.load(path)
    labels = data.files[1:]
    y = [data[g] for g in labels]
    y_array =np.array(y).T
    
    # Standardize using given scale_params input
    y_new = standardize(y_array, columns=np.array(range(y_array.shape[1])), params=scale_params)

    # Save new target file
    new_path = root_dir.joinpath('processed').joinpath('standardized')
    np.save(new_path.joinpath(f's_{file_number}.npy'), y_new)


def create_single_row_files(numbers = list(range(10))):
  # Creates single row pickle files for all data
  # Inputs: numbers (list of how many files to turn into single row files)
  # Returns: None, just saves file
  # Note: goal was to speed up dataloader performance, but this did not workl
  x_root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")
  y_root_dir = x_root_dir.joinpath("processed/standardized")
  target_x_dir = x_root_dir.joinpath("processed/standardized/spectra")
  target_y_dir = x_root_dir.joinpath("processed/standardized/outputs")

  numbers = list(range(10))
  # Iterate through dataset files
  for file_number in numbers:
    x_path=x_root_dir.joinpath(f'{file_number}.npz')
    data = np.load(x_path,allow_pickle=True)
    x = data['fi']
    total_samples = x.shape[0]

    y_path = y_root_dir.joinpath(f's_{file_number}.npy')
    y = np.load(y_path, allow_pickle=True)

    print('x', x.shape)
    print('y', y.shape)

    j=0
    for i in list(range(total_samples)):
      x_chunk = x[i]
      y_chunk = y[i]

      np.save(target_x_dir.joinpath(f's_x_{file_number}_{i}.npy'), x_chunk)
      np.save(target_y_dir.joinpath(f's_y_{file_number}_{i}.npy'), y_chunk)
      j +=1
    
    print(f"****Created {j} files for the {total_samples} spectra in file number {file_number}******")

def get_labels():
  file_number = 0
  root_dir = Path("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/data")
  path=root_dir.joinpath(f'{file_number}.npz')
  data = np.load(path)
  labels = data.files[1:]
  labels_clean = [label[5:] for label in labels]
  return labels_clean













from sys import prefix
import torch
from pathlib import Path
import time
import numpy as np
from datasets.stars import BasicDataset
from torch.utils.data import random_split, DataLoader
import pandas as pd
from comet_ml import Experiment
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, MetricCollection

class Trainer(object):
  def __init__(self, model, lr, wd, epochs, num_sets, experiment_name = "binary__stars"):
    self.epochs = epochs
    self.model = model
    self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    self.scale_params = np.load("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/saved_states/scale_params_full.npy", allow_pickle=True).item()
    self.labels = labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']

    self.hyper_params={
      'name': self.model.name,
      'num_epochs': self.epochs,
      'optimizer': 'Adam',
      'learning_rate': lr,
      'weight_decay': wd,
      'num_sets': num_sets
    }

    # self.experiment = Experiment(project_name = experiment_name)
    # self.experiment.log_parameters(self.hyper_params)

    metrics_collection = MetricCollection([MeanAbsolutePercentageError(), MeanSquaredError()])
    self.temp_metrics = metrics_collection.clone(prefix='t_')
    self.vsini_metrics = metrics_collection.clone(prefix='vsini_')
    self.alpha_metrics = metrics_collection.clone(prefix='a_')
    self.lumin_metrics = metrics_collection.clone(prefix='lum_')
    self.log_g_metrics = metrics_collection.clone(prefix='logg_')
    self.metal_metrics = metrics_collection.clone(prefix='metal_')
  
  def train(self):

    for epoch in range(self.epochs):
      for i in range(self.hyper_params['num_sets']):
        full_data = BasicDataset(i)
        train_len = int(len(full_data)*0.8)
        val_len = len(full_data) - train_len

        train_data, val_data = random_split(full_data, [train_len, val_len], generator=torch.Generator().manual_seed(42))

        print(f'Dataset {i} created:', len(train_data), len(val_data))

        train_loader = DataLoader(train_data, batch_size=16)
        val_loader = DataLoader(val_data, batch_size=16)
        train_batch_loss = self._train_one_epoch(train_loader)
        print(train_batch_loss)
        val_loss = self._evaluate_one_epoch(val_loader)
        print(val_loss)

  def _evaluate_one_epoch(self, dataloader, wordy=True):
    self.model.eval()
    with torch.no_grad():
        norm_losses = []
        denorm_losses = []
        for j, (inputs, targets) in enumerate(dataloader):

            yhat = self.model(inputs)
            norm_loss, target_correct = weighted_mse_loss(yhat, targets) # Loss function, and determine best target to evaluate for
            norm_losses.append(norm_loss)
            yhat_denorm = denormalize_tensor(yhat, self.scale_params) # Denormalize the prediction tensor - now Bx12
            target_denorm = denormalize_tensor(target_correct, self.scale_params) # Denormalize the target tensor - now Bx12
            
            i=0
            for label in self.labels:
              yhat_selection = yhat_denorm[:, i:i+2]
              target_selection = target_denorm[:, i:i+2]
              loss_select = getattr(self, label+"_metrics")
              output = loss_select(yhat_selection, target_selection)
              i += 2

            if wordy:
              # target_correct = get_correct_layout(yhat, targets) # Reduce targets to just the correct one
              print_sample_outputs(yhat_denorm, target_denorm) 
              print()
              print_denorm_loss(yhat_denorm, target_denorm)             
              wordy = False

        for label in self.labels:
          loss_select = getattr(self, label+"_metrics")
          var_loss = loss_select.compute()
          print(var_loss)

    return sum(norm_losses)/len(norm_losses)

  def _train_one_epoch(self, train_loader):
    self.model.train()
    
    total_batch_loss=0
    for j, (inputs, targets) in enumerate(train_loader):
        # clear the gradients
        # print('inputs shape:', inputs.shape) # Bx10197
        # print('targets shape', targets.shape) # Bx12x2
        self.optimizer.zero_grad()
        # compute the model output
        yhat = self.model(inputs)
        # targets = targets
        # print('actual output', yhat.shape) # B x 12
        loss, _ = weighted_mse_loss(yhat, targets)
        total_batch_loss+=loss

        # credit assignment
        loss.backward()
        # update model weights
        self.optimizer.step()
        break
    
    return(total_batch_loss/j)

        


###### OLD

def weighted_mse_loss(yhat, target):

  # Adjust weight and input dimensions for broadcasting
  yhat = yhat.unsqueeze(dim=-1)
  # print('new shape ', yhat.shape)

  # Calculate the mean for each option (Star A first then Star B, and the reverse)
  options = torch.sum(((yhat - target) ** 2), dim=1) # Gives a Bx2 array 
  # print(options.shape)

  # Take the minimum of the options
  minimum, indices = torch.min(options, dim=-1) #Picks the minimum, gives a size-B vector
  # print(minimum.shape)
  
  target_correct_list = [target[row, : , idx] for row, idx in enumerate(indices)]
  target_correct = torch.vstack(target_correct_list)

  # Return the minimum, averaged across the batch
  return torch.mean(minimum), target_correct



def denormalize_tensor(x, scale_params):
  # Rather self-explanatory
  avgs = scale_params['avgs']
  stddevs = scale_params['stds']
  new_arg = x*stddevs+avgs
  return new_arg

def denormalized_mse_loss(yhat, y, scale_params):
  # Based on proof that the unnormalized difference is really just (yhat-y)^2*a^2
  avgs = scale_params['avgs']
  difference = yhat-y
  value = (difference^2) * (avgs^2)
  return value

def print_sample_outputs(yhats, targets, number=1):
  labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2', 'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']
  for i in range(number):
    yhat = np.round(yhats[0].numpy(), decimals=3)
    target = np.round(targets[0].numpy(), decimals=3)
    table = [yhat, target]
    df = pd.DataFrame(table, columns = labels, index=['predicted', 'actual'])
    print(df)


def print_denorm_loss(yhats, targets, number=1):
  labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
  # print(yhats[0])
  # print(targets[0])
  differences = yhats - targets
  squared_differences = torch.sum(torch.square(differences), dim=0)
  # print(squared_differences)
  losses = []
  i=0
  for label in labels:
    label_loss = torch.sum(squared_differences[i:i+2])
    # print(label, label_loss)
    losses.append(np.round(label_loss.numpy(), decimals = 5))
    i+=2

  table = np.array(losses).reshape((1,6))
  # print(table.shape)
  df = pd.DataFrame(table, columns = labels, index=["Loss"])
  print(df)





def evaluate_model_denormalizd(data, model, scale_params, j=0, wordy=False):
  # Inputs:data (Bx1X)
  # 
  #
    std_dev = np.reshape(scale_params['stds'], (1,-1))
    avg = np.reshape(scale_params['avgs'], (1,-1))
    param_len = std_dev.shape[1]
    model.eval()
    with torch.no_grad():
        losses = []
        for x, y in data:
            scores = model(x)
            scores_unnormalized = scores*std_dev+avg
            scores_unnormalized = scores_unnormalized[:, 2*j:2*j+2]
            if wordy:
              #for i in range(0, param_len, 2):
                print("Predictions: ", scores_unnormalized[0,:].tolist())
                print("Actual: ", y[0,2*j:2*j+2,0].tolist())
            losses.append(weighted_mse_loss(scores_unnormalized, y[:, 2*j:2*j+2]))
    return sum(losses)/len(losses)

def evaluate_model_denormalize(data, model, scale_params, wordy=False):
    std_dev = np.reshape(scale_params['stds'], (1,-1))
    avg = np.reshape(scale_params['avgs'], (1,-1))
    param_len = std_dev.shape[1]
    model.eval()
    params = ['vsini', 'metallicity', 'alpha', 'temperature', 'log_g', 'luminosity']
    with torch.no_grad():
        losses = []
        for x, y in data:
            scores = model(x)
            scores_unnormalized = scores*std_dev+avg
            if wordy:
              j=0
              for i in range(0, param_len, 2):
                print(f"{params[j]} predictions: ", scores_unnormalized[0,i:i+2].tolist())
                print(f"{params[j]} actual: ", y[0,i:i+2,0].tolist())
                j+=1
            losses.append(weighted_mse_loss(scores_unnormalized, y))
            print("")
    return sum(losses)/len(losses)


def train(data, model, epochs, lr, wd=0, name='', schedule=False):
    trn, vld, tst = data
    tic = timeit.default_timer()
    test_loss_best = 10e30
    trn_loss = []
    tst_loss = []
    vld_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if schedule:
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    print(f"Starting training for model: {name}.\n")
    for epoch in range(epochs):
        model.train()
        
        for i, (inputs, targets) in enumerate(trn):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            #loss = criterion(yhat, targets)
            loss = weighted_mse_loss(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        model.eval()
        tst_loss.append(evaluate_model(tst, model))
        trn_loss.append(evaluate_model(trn, model))
        vld_loss.append(evaluate_model(vld, model))

        if schedule:
          #print("Learning rate: {}".format(lr))
          scheduler.step()

        print("Epoch : {:d} || Train set loss : {:.3f}".format(
            epoch+1, trn_loss[-1]))
        print("Epoch : {:d} || Validation set loss : {:.3f}".format(
            epoch+1, vld_loss[-1]))
        print("Epoch : {:d} || Test set loss : {:.3f}".format(
            epoch+1, tst_loss[-1]))
        
        test_loss = tst_loss[-1]
        if test_loss < test_loss_best:
          test_loss_best = test_loss
          print("Saving model")
          torch.save(model.state_dict(), f"./state_dicts/{name}_model.pt")
        print("*************************************************\n")
    print("Training is over.")
    return trn_loss, tst_loss, vld_loss

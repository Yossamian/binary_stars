from sys import prefix
import torch
from pathlib import Path
import time
import numpy as np
from torch._C import device
from datasets.stars import BasicDataset
from torch.utils.data import random_split, DataLoader
import pandas as pd
import comet_ml
from comet_ml import Experiment
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, MetricCollection, CosineSimilarity
from contextlib import nullcontext
import utils.metrics

# comet_ml.config.save(api_key="8EKfM7gNRhoWg9I2sQz6rcHls")

class Trainer(object):
  def __init__(self, model, parameters, num_sets, train_loader, val_loader, experiment_name = "binary__stars"):
    
    # Device management
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Folder management
    self.root_dir = Path('/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/')
    self.folder = self.root_dir.joinpath(f'saved_models/{parameters["name"]}')
    check_exists(self.folder)

    # Model Parameters
    self.model = model
    self.model = self.model.to(self.device)
    
    # Optimizer Parameters
    lr = parameters['lr']
    wd = parameters['wd']
    self.optimizer = getattr(torch.optim, parameters['optimizer'])(model.parameters(), lr=lr, weight_decay=wd)

    # Loss function
    self.loss_func = getattr(utils.metrics, parameters['loss'])

    # Data and Training Parameters
    self.scale_params = np.load("/content/drive/Shareddrives/Learning_Deep/project_954437307_066610346/saved_states/scale_params_full.npy", allow_pickle=True).item()
    self.labels = labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.epochs = parameters['epochs']
    self.early_stopping = parameters['early_stopping']

    # Comet hyperparametrs and logging
    self.hyper_params={
      'name': parameters['name'],
      'num_epochs': self.epochs,
      'optimizer': parameters['optimizer'],
      'learning_rate': lr,
      'weight_decay': wd,
      'num_sets': num_sets
    }

    if experiment_name:
      self.comet=True
      self.experiment = Experiment(project_name = experiment_name)
      self.experiment.log_parameters(self.hyper_params)
    else: 
      self.comet = False

    # METRICS
    ranges = [10, 1, 0.8, 3000, 3, 478827900000]
    dict_denom = {label: num for label, num in iter(zip(self.labels, ranges))}
    self.metrics = {}
    for key, range_value in dict_denom.items():
      self.metrics[key] = utils.metrics.StarMetrics(range_value).to(self.device)

  
  def train(self):
    
    val_loss_best = 1e9
    time_since_improved = 0

    print('****Beginning Training****')
    print(f'Training on GPU:{next(self.model.parameters()).device}')

    for epoch in range(self.epochs):

      train_batch_loss = self._train_one_epoch().item()
      print(f"Epoch {epoch} training loss: {train_batch_loss}")
      val_loss = self._evaluate_one_epoch().item()
      print(f"Epoch {epoch} validation Loss: {val_loss}")

      if val_loss < val_loss_best:
          val_loss_best = val_loss
          print("Saving model")
          torch.save(self.model.state_dict(), self.folder.joinpath('best_model.pt'))
          time_since_improved = 0
          print("*************************************************\n")
      else:
        time_since_improved+=1

      if time_since_improved == self.early_stopping:
        print(f"No model improvement in {self.early_stopping} epochs: terminating training")
        break

  def _evaluate_one_epoch(self, wordy=True):
    self.model.eval()
    with self.experiment.test() if self.comet else nullcontext():
      with torch.no_grad():
          total_norm_losses = 0
          denorm_losses = []
          for j, (inputs, targets) in enumerate(self.val_loader):

              inputs = inputs.to(self.device)
              targets = targets.to(self.device)

              yhat = self.model(inputs)
              norm_loss, target_correct = self.loss_func(yhat, targets) # Loss function, and determine best target to evaluate for
              total_norm_losses += norm_loss

              yhat_denorm = self.denormalize_tensor(yhat, self.scale_params) # Denormalize the prediction tensor - now Bx12
              target_denorm = self.denormalize_tensor(target_correct, self.scale_params) # Denormalize the target tensor - now Bx12
              
              i=0
              for label in self.labels:
                yhat_selection = yhat_denorm[:, i:i+2]
                target_selection = target_denorm[:, i:i+2]
                # loss_select = getattr(self, label+"_metrics")
                self.metrics[label].update(yhat_selection, target_selection)
                
                # spec_loss = adjusted_MAPE(yhat_selection, target_selection, label)
                i += 2

              if wordy:
                # target_correct = get_correct_layout(yhat, targets) # Reduce targets to just the correct one
                print("SAMPLE MODEL OUTPUTS FROM VAL SET")
                print_sample_outputs(yhat_denorm, target_denorm) 
                # print()
                # print("SAMPLE MSE LOSS")
                # print_denorm_loss(yhat_denorm, target_denorm)             
                wordy = False
          
          total_range_loss = 0
          for label in self.labels:
            # loss_select = getattr(self, label+"_metrics")
            var_loss = self.metrics[label].compute()
            total_range_loss += var_loss['RangeLoss']
            if self.comet:
              self.comet_log_metrics(label, var_loss)

            # print(label.upper(), var_loss)
          
          epoch_loss = total_norm_losses/j
        
          if self.comet:
            self.experiment.log_metric('val_loss', epoch_loss)
            self.experiment.log_metric('total_range_loss', total_range_loss)

    return epoch_loss

  def _train_one_epoch(self):

    self.model.train()
    
    with self.experiment.train() if self.comet else nullcontext():
      total_loss=0
      for j, (inputs, targets) in enumerate(self.train_loader):

          inputs = inputs.to(self.device)
          targets = targets.to(self.device)
        
          # clear the gradients
          # print('inputs shape:', inputs.shape) # Bx10197
          # print('targets shape', targets.shape) # Bx12x2
          self.optimizer.zero_grad()
          # compute the model output
          yhat = self.model(inputs)
          # targets = targets
          # print('actual output', yhat.shape) # B x 12
          loss, _ = self.loss_func(yhat, targets)
          total_loss+=loss

          # credit assignment
          loss.backward()
          # update model weights
          self.optimizer.step()
      
      epoch_loss = total_loss/j
      
      if self.comet:
        self.experiment.log_metric('train_loss', epoch_loss)
    
    return epoch_loss
    
  def denormalize_tensor(self, x, scale_params):
    # Rather self-explanatory
    avgs = scale_params['avgs'].to(self.device)
    stddevs = scale_params['stds'].to(self.device)
    new_arg = x*stddevs+avgs
    return new_arg

  def comet_log_metrics(self, label, results):
    list_of_metrics = [m for m, _ in self.metrics[label].items()]
    for metric in list_of_metrics:
      # self.experiment.log_metrics({f'{label}_{metric}': loss for loss in list(zip(self.classes, results[metric]))})
      self.experiment.log_metric(name=f'{label}_{metric}', value=results[metric])


def print_sample_outputs(yhats, targets, number=1):
  # Prints sample outputs of all
  labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2', 'list_t_1', 'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']
  for i in range(number):
    yhat = np.round(yhats[0].cpu().numpy(), decimals=3)
    target = np.round(targets[0].cpu().numpy(), decimals=3)
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
    losses.append(np.round(label_loss.cpu().numpy(), decimals = 5))
    i+=2

  table = np.array(losses).reshape((1,6))
  # print(table.shape)
  df = pd.DataFrame(table, columns = labels, index=["Loss"])
  print(df)

def check_exists(folder):
  if not folder.exists():
    folder.mkdir()


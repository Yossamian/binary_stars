import torch
from pathlib import Path
import time
import numpy as np

class Trainer(object):
  def __init__(self, model, optimizer, epochs, train_loader, val_loader):
    self.epochs = epochs
    self.model = model
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
  
  def train(self):
    for epoch in range(self.epochs):
      self._train_one_epoch()
      val_loss = self._evaluate_one_epoch(self.val_loader)

  def _evaluate_one_epoch(self, dataloader, wordy=False):
    self.model.eval()
    with torch.no_grad():
        losses = []
        for j, (inputs, targets) in enumerate(dataloader):
          for i in range(inputs.shape[1]):
            actual_input = inputs[:,i,:].float()
            actual_target = targets[:,i,:].float()
            scores = self.model(actual_input)
            losses.append(weighted_mse_loss(scores, actual_target))
            if wordy:
              param_len = scores.shape[1]
              for i in range(0, param_len, 2):
                print("Predictions: ", scores[0,i:i+2].tolist())
                print("Actual: ", actual_target[0,i:i+2,0].tolist())
    return sum(losses)/len(losses)

  def _train_one_epoch(self):
    self.model.train()
        
    for j, (inputs, targets) in enumerate(self.train_loader):
        # clear the gradients
        print('inputs shape:', inputs.shape)
        print('targets shape', targets.shape)
        print(f'Training batch {j}')
        self.optimizer.zero_grad()
        # compute the model output
        for i in range(inputs.shape[1]):
          actual_input = inputs[:, i, :].float()
          actual_target = targets[:, i, :].float()
          print('actual input', actual_input.shape)
          print('actual target', actual_target.shape)
          yhat = self.model(actual_input)
          # print('actual output', yhat.shape)
          loss = weighted_mse_loss(yhat, actual_target)
          # credit assignment
          loss.backward()
          # update model weights
          self.optimizer.step()


###### OLD

def weighted_mse_loss(inputs, target, weights=[1]):

  # Create array of weights (if desired - default is no weight whange)
  weight = np.array(weights)

  # Adjust weight and input dimensions for broadcasting
  weight = torch.tensor(weight).reshape(1, weight.shape[0], 1)
  inputs = inputs.unsqueeze(dim=-1)

  # Calculate the mean for each option (Star A first then Star B, and the reverse)
  options = torch.sum((weight * (inputs - target) ** 2), dim=1)

  # Take the minimum of the options
  minimum, _ = torch.min(options, dim=-1)

  # Return the minimum, averaged
  return torch.mean(minimum)


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

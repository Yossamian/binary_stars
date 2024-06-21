import torchmetrics as tm
from comet_ml import Experiment
import torch
import torch.nn as nn


class Huber(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.Huber = nn.HuberLoss(reduction='mean', delta=delta)

    def forward(self, output, target):
        loss = self.Huber(output, target)
        return loss


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSE = nn.MSELoss()

    def forward(self, output, target):
        loss = self.MSE(output, target)
        return loss


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAE = nn.L1Loss()

    def forward(self, output, target):
        loss = self.MAE(output, target)
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.Huber = nn.HuberLoss(reduction='mean', delta=1.0)

    def forward(self, output, target):
        loss = self.Huber(output, target)
        return loss



def DiffLoss(predictions, targets, a=1, b=1):
    """
    Mean Absolute Error
    https://en.wikipedia.org/wiki/Mean_absolute_error
    Inputs:
    predictions: a tensor of shape (B, L), where B is batch size and L is label size
    targets: a tensor of shape (B, L, 2)
    epsilon = value to ensure that we do not divide by zero
    Added epsilon to denominator to ensure no division by zero
    for_MASE = this will change the way thte result is returned
    """

    min_predictions, _ = torch.min(predictions, dim=1)
    min_targets, _ = torch.min(targets, dim=1)

    diff_predictions = torch.abs(predictions[:, 1]-predictions[:, 0])
    diff_targets = torch.abs(targets[:, 1]-targets[:, 0])

    error_min = torch.mean((min_targets-min_predictions)**2)
    error_diff = torch.mean((diff_targets-diff_predictions)**2)

    total_error = a*error_min + b*error_diff


    return total_error


def BootlegMSE(predictions, targets, a=1, b=1):

    min_predictions, _ = torch.min(predictions, dim=1)
    max_predictions, _ = torch.max(predictions, dim=1)
    min_targets, _ = torch.min(targets, dim=1)
    max_targets, _ = torch.max(targets, dim=1)

    # max_predictions = torch.abs(predictions[:, 1]-predictions[:, 0])
    # max_targets = torch.abs(targets[:, 1]-targets[:, 0])

    error_min = torch.mean((min_targets-min_predictions)**2)
    error_max = torch.mean((max_targets-max_predictions)**2)
    # error_diff = torch.mean((diff_targets-diff_predictions)**2)

    total_error = (error_min + error_max) / 2

    return total_error
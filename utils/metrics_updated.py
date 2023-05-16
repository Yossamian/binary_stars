import torchmetrics as tm
from comet_ml import Experiment
import torch
import torch.nn as nn


def all_metrics(predictions, targets):
    mape_adjusted, _ = MAPE_adjusted(predictions, targets)
    smape_adjusted, _ = SMAPE_adjusted(predictions, targets)
    mae, _ = MAE(predictions, targets)
    mase, _ = MASE(predictions, targets)

    return mape_adjusted, smape_adjusted, mae, mase


def MAPE_adjusted(predictions, targets, epsilon=0.1):
    """
    Mean Absolute Percentage Error
    Inputs:
    predictions: a tensor of shape (B, L), where B is batch size and L is label size
    targets: a tensor of shape (B, L, 2)
    epsilon = value to ensure that we do not divide by zero
    Function performs same as normal MAPE, but with minor adjustments:
    Absolute value of the numerator (predictions - targets) and denominator (targets)
    are taken separately. This is to ensure that we can add epsilon and not divide by zero
    """
    # Expand preditions to allow for forecasting, then take difference between preds and targets
    # difference is a (B, L, 2) tensor of differences between preds and target
    predictions_unsqueezed = predictions.unsqueeze(dim=-1)
    difference = targets - predictions_unsqueezed

    # Adjusted absolute percentage error calculated on a per-label basis
    # Shape is still (B,L,2)
    percentage_difference = torch.abs(difference) / (torch.abs(targets) + epsilon)

    # Sum all of the label MAPES - necessary for determining which order is correct
    # Result is a tensor of shape (B, 2), where 2 is the two different orders
    # Each B has the total absolute percentage error for the two options
    sum_of_differences = torch.sum(percentage_difference, axis=1)

    # Remove axis=2 of size 2, by taking min, which is the correct order of the targets
    # Returns minimums and indices, both 1-d tensors of length B
    minimum, indices = torch.min(sum_of_differences, axis=-1)

    # Get the targets that were actually used to calculate the MAPE
    correct_target_order = [targets[r, :, i] for r, i in enumerate(indices)]
    correct_target_order = torch.vstack(correct_target_order)

    # Now, calculate actual MAPE using correct target order
    perc_diff = torch.abs(correct_target_order - predictions) / (torch.abs(correct_target_order) + epsilon)
    mean_across_batch = torch.mean(perc_diff, axis=0)
    mean_total = torch.mean(mean_across_batch)

    # Incorrect, removed line
    # Average out the MAPEs across the batch
    # minimum = torch.mean(minimum)

    return mean_total, correct_target_order


def SMAPE_adjusted(predictions, targets, epsilon=0.1):
    """
    Symmetrical MAPE
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    Inputs:
    predictions: a tensor of shape (B, L), where B is batch size and L is label size
    targets: a tensor of shape (B, L, 2)
    epsilon = value to ensure that we do not divide by zero
    Added epsilon to denominator to ensure no division by zero
    """
    # Expand preditions to allow for forecasting, then take difference between preds and targets
    predictions_unsqueezed = predictions.unsqueeze(dim=-1)
    difference = targets - predictions_unsqueezed

    # Adjusted MAPE calculated on a per-label basis
    numerator = torch.abs(difference)
    denominator = torch.abs(predictions_unsqueezed) + torch.abs(targets) + epsilon
    percentage_difference = numerator / denominator

    # Sum all of the label MAPES
    # Result is a tensor of shape (B, 2), where 2 is the two different orders
    sum_of_differences = torch.sum(percentage_difference, axis=1)

    # Remove axis=2 of size 2, by taking min, which is the correct order of the targets
    # Returns minimums and indices, both 1-d tensors of length B
    minimum, indices = torch.min(sum_of_differences, axis=-1)

    # Get the targets that were actually used to calculate the MAPE
    correct_target_order = [targets[r, :, i] for r, i in enumerate(indices)]
    correct_target_order = torch.vstack(correct_target_order)

    # Now, actually calculate the SMAPE using the correct order
    numerator = torch.abs(correct_target_order - predictions)
    denominator = torch.abs(predictions) + torch.abs(correct_target_order) + epsilon
    percentage_difference = numerator / denominator
    mean_across_batch = torch.mean(percentage_difference, axis=0)
    mean_total = torch.mean(mean_across_batch)

    # Incorrect, removed line
    # Average out the MAPEs across the batch
    # minimum = torch.mean(minimum)

    return mean_total, correct_target_order


def MAE(predictions, targets, for_MASE=False):
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
    # Absolute error taken on a per label basis
    predictions_unsqueezed = predictions.unsqueeze(dim=-1)
    difference = torch.abs(targets - predictions_unsqueezed)

    # Sum all of the label absolute errors - need to do this to measure which order is correct
    # Result is a tensor of shape (B, 2), where 2 is the two different orders
    sum_of_differences = torch.sum(difference, axis=1)

    # Remove axis=1 of size 2, by taking min, which is the correct order of the targets
    # Returns minimums and indices, both 1-d tensors of length B
    # Minimum is the total label error for each batch
    minimum, indices = torch.min(sum_of_differences, axis=-1)

    # Get the targets that were actually used to calculate the MAPE
    correct_target_order = [targets[r, :, i] for r, i in enumerate(indices)]
    correct_target_order = torch.vstack(correct_target_order)

    # Now we can actually calculate the MAE, across the batch
    # Result is a vector of length L, with MAE for each label
    differences = torch.abs(correct_target_order - predictions)
    mae = torch.mean(differences, axis=0)

    if not for_MASE:
        # Average out the absolute error across the batch to get MAE
        mae = torch.mean(mae)

    return mae, correct_target_order


def MAD(targets):
    """
    Mean Absolute Deviation
    https://en.wikipedia.org/wiki/Average_absolute_deviation
    targets: a tensor of shape (B, L, 2)
    """
    # First, just take one section of the targets tensor (it is repeated for multi-order)
    mean_across_batch = torch.mean(targets, axis=0)
    mean_per_label = torch.mean(mean_across_batch, axis=-1)

    # Mean per label is now a L-length vector of means for each label
    # For broadcasting
    mean_per_label = mean_per_label.unsqueeze(0).unsqueeze(axis=-1)

    deviations = torch.abs(targets - mean_per_label)

    # Take the mean of the deviations across the batch
    # This results in the mad per label, but in two axes
    mean_deviations = torch.mean(deviations, axis=0)

    mean_deviations = torch.mean(mean_deviations, axis=-1)
    # Result is a vector of length L, with mean deviation of each label

    return mean_deviations


def MASE(predictions, targets):
    """
    Mean Absolute Scaled Error
    https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    """
    # Get MAE, a vector of length L with MAE for each label
    mae, correct_target_order = MAE(predictions, targets, for_MASE=True)

    # Get MAD, a vector of length L with MAD for each label
    mad = MAD(targets)

    mase = mae / mad

    # Take the average of the MASE across all labels
    mase = torch.mean(mase)

    return mase, correct_target_order


class Torch_MSE(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MSE, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, output, target):
        loss = self.MSE(output, target)
        return loss


def MSE(predictions, targets):
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
    # Absolute error taken on a per label basis
    predictions_unsqueezed = predictions.unsqueeze(dim=-1)
    differences_squared = (targets - predictions_unsqueezed)**2

    # Sum all of the label absolute errors - need to do this to measure which order is correct
    # Result is a tensor of shape (B, 2), where 2 is the two different orders
    sum_of_differences = torch.sum(differences_squared, axis=1)

    # Remove axis=1 of size 2, by taking min, which is the correct order of the targets
    # Returns minimums and indices, both 1-d tensors of length B
    # Minimum is the total label error for each batch
    minimum, indices = torch.min(sum_of_differences, axis=-1)

    # Get the targets that were actually used to calculate the MAPE
    correct_target_order = [targets[r, :, i] for r, i in enumerate(indices)]
    correct_target_order = torch.vstack(correct_target_order)

    # Now we can actually calculate the MAE, across the batch
    # Result is a vector of length L, with MAE for each label
    differences_squared = (correct_target_order - predictions)**2
    mse = torch.mean(differences_squared)

    return mse, correct_target_order

def get_losses(predictions, targets):
    """
    Mean Absolute Error
    https://en.wikipedia.org/wiki/Mean_absolute_error
    Inputs:
    predictions: a tensor of shape (B, L), where B is batch size and L is label size
    targets: a tensor of shape (B, L)
    epsilon = value to ensure that we do not divide by zero
    Added epsilon to denominator to ensure no division by zero
    for_MASE = this will change the way thte result is returned
    """
    # Absolute error taken on a per label basis
    predictions_unsqueezed = predictions.unsqueeze(dim=-1)
    differences_squared = (targets - predictions_unsqueezed)**2

    # Sum all of the label absolute errors - need to do this to measure which order is correct
    # Result is a tensor of shape (B, 2), where 2 is the two different orders
    sum_of_differences = torch.sum(differences_squared, axis=1)

    # Remove axis=1 of size 2, by taking min, which is the correct order of the targets
    # Returns minimums and indices, both 1-d tensors of length B
    # Minimum is the total label error for each batch
    minimum, indices = torch.min(sum_of_differences, axis=-1)

    # Get the targets that were actually used to calculate the MAPE
    correct_target_order = [targets[r, :, i] for r, i in enumerate(indices)]
    correct_target_order = torch.vstack(correct_target_order)

    # Now we can actually calculate the MAE, across the batch
    # Result is a vector of length L, with MAE for each label
    differences_squared = (correct_target_order - predictions)**2
    mse = torch.mean(differences_squared)

    return mse, correct_target_order
"""
def RangeLoss(tm.Metric):
    def __init__(self, range_value):
        super().__init__()
        self.add_state("pseudo_mape", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.denominator = range_value

    def update(self, x, y):
        assert x.shape == y.shape
        a = torch.abs(y - x) / self.denominator
        self.pseudo_mape += torch.sum(a)
        self.total += x.shape[0]
        # print("Update runs: ", self.pseudo_mape, self.total)

    def compute(self):
        # print("Compute runs")
        return self.pseudo_mape / self.total
        
        """
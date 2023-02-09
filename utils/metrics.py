import torchmetrics as tm
from comet_ml import Experiment
import torch


class StarMetrics(tm.MetricCollection):
    def __init__(self, range_value):
        metrics_kwargs = {
            'num_classes': 7,
            'average': None,
            'ignore_index': None,
            'mdmc_average': 'global'
        }
        metrics = [tm.MeanAbsolutePercentageError(),
                   tm.MeanSquaredError(),
                   tm.CosineSimilarity(),
                   RangeLoss(range_value)]
        super().__init__(metrics)


class RangeLoss(tm.Metric):
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


def comet_log_metrics(self, results):
    list_of_metrics = [m for m, _ in self.metrics.items()]
    for metric in list_of_metrics:
        self.experiment.log_metrics(
            {label + f'_{metric}': loss for label, loss in list(zip(self.classes, results[metric]))})


#### Old metrics
### OA = Order Agnostic

def OA_mse_loss(yhat, target):
    # Adjust weight and input dimensions for broadcasting
    yhat = yhat.unsqueeze(dim=-1)
    # print('new shape ', yhat.shape)

    # Calculate the mean for each option (Star A first then Star B, and the reverse)
    options = torch.sum(((yhat - target) ** 2), dim=1)  # Gives a Bx2 array
    # print(options.shape)

    # Take the minimum of the options
    minimum, indices = torch.min(options, dim=-1)  # Picks the minimum, gives a size-B vector
    # print(minimum.shape)

    target_correct_list = [target[row, :, idx] for row, idx in enumerate(indices)]
    target_correct = torch.vstack(target_correct_list)

    # Return the minimum, averaged across the batch
    return torch.mean(minimum), target_correct


def OA_mae_loss(yhat, target):
    # Adjust weight and input dimensions for broadcasting
    yhat = yhat.unsqueeze(dim=-1)
    # print('new shape ', yhat.shape)

    # Calculate the mean for each option (Star A first then Star B, and the reverse)
    options = torch.sum((torch.abs(yhat - target)), dim=1)  # Gives a Bx2 array
    # print(options.shape)

    # Take the minimum of the options
    minimum, indices = torch.min(options, dim=-1)  # Picks the minimum, gives a size-B vector
    # print(minimum.shape)

    target_correct_list = [target[row, :, idx] for row, idx in enumerate(indices)]
    target_correct = torch.vstack(target_correct_list)

    # Return the minimum, averaged across the batch
    return torch.mean(minimum), target_correct


def adjusted_MAPE(yhat, target, label):
    """To create a loss function for the values that will not compute well with MAPE
  Simply absolute difference divided by range of possibly values
  metal_vals = [0 - 1]
  alpha_vals = [-0.2 -  0.6]
  temp_vals = [3500 - 6500]
  logg_vals = [3 - 6]

  vsini_vals= [0 - 10]
  l_vals - [6992100000.0 - 485820000000.0]"""

    labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
    ranges = [10, 1, 0.8, 3000, 3, 478827900000]

    dict_denom = {label: num for label, num in iter(zip(labels, ranges))}

    loss = torch.abs(yhat - target) / dict_denom[label]

    return torch.mean(loss)


def denormalized_mse_loss(yhat, y, scale_params):
    # Based on proof that the unnormalized difference is really just (yhat-y)^2*a^2
    avgs = scale_params['avgs']
    difference = yhat - y
    value = (difference ^ 2) * (avgs ^ 2)
    return value

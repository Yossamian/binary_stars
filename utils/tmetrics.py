import torchmetrics as tm
from comet_ml import Experiment
import torch


class StarMetrics(tm.MetricCollection):
    def __init__(self, range_value):
        metrics = {'MAPE': tm.MeanAbsolutePercentageError(),
                   'MAE': tm.MeanAbsoluteError(),
                   'SMAPE': tm.SymmetricMeanAbsolutePercentageError(),
                   'MSE': tm.MeanSquaredError(),
                   'RangeL': RangeLoss(range_value)}
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


from sys import prefix
import torch
from pathlib import Path
import time
import numpy as np
from torch.utils.data import random_split, DataLoader
import pandas as pd
from comet_ml import Experiment
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError, MetricCollection, CosineSimilarity
from contextlib import nullcontext
import utils.metrics
from utils.metrics_updated import MAPE_adjusted, SMAPE_adjusted, MASE, all_metrics, BootlegMSE
import utils.metrics_updated
from torch.optim.lr_scheduler import StepLR
import yaml
# comet_ml.config.save(api_key="8EKfM7gNRhoWg9I2sQz6rcHls")


class Trainer(object):
    def __init__(self,
                 model,
                 parameters,
                 train_loader,
                 val_loader,
                 test_loader=None,
                 experiment_name="binary_stars",
                 parallel=True):

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Folder management - folder for model itself, as well as subfolders for state dictionary and sample outputs
        self.root_dir = Path('/media/sam/data/work/stars/configurations')
        self.folder = self.root_dir.joinpath(f'saved_models/{parameters["name"]}')
        self.state_dict_folder = self.folder.joinpath('state_dicts')
        self.sample_output_folder = self.folder.joinpath('sample_outputs')
        self.inference_output_folder = self.folder.joinpath('sample_outputs/inference_outputs')
        check_exists(self.folder, self.state_dict_folder, self.sample_output_folder, self.inference_output_folder)
        torch.save(parameters, self.folder.joinpath(f'parameters.pt'))
        with open(self.folder.joinpath('config.yaml'), 'w') as f:
            yaml.dump(parameters, f)

        # Model Parameters
        if parallel:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

        # Optimizer Parameters
        self.optimizer = getattr(torch.optim, parameters['optimizer'])(model.parameters(),
                                                                       lr=parameters['lr'],
                                                                       weight_decay=parameters['wd'])

        self.scheduler = StepLR(self.optimizer, step_size=parameters['optimizer_step'], gamma=parameters['optimizer_gamma'])

        # Loss function
        self.loss_func = getattr(utils.metrics_updated, parameters['loss'])

        self.torch_MSE = torch.nn.MSELoss()

        # Label parameters
        self.target_param = parameters["target_param"]
        self.i1, self.i2 = self.get_data_indices()

        self.output_labels, self.param_labels = self.create_output_labels()
        dict_denom = self.create_dict_denom()


        # Dataloaders and training
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = parameters['epochs']
        self.sets_between_eval = parameters['sets_between_eval']
        self.early_stopping = parameters['early_stopping']

        if experiment_name:
            self.comet = True
            self.experiment = Experiment(project_name=experiment_name)
            self.experiment.log_parameters(parameters)
        else:
            self.comet = False

        # METRICS
        self.metrics = {}
        for key, range_value in dict_denom.items():
            self.metrics[key] = utils.metrics.StarMetrics(range_value).to(self.device)

    def train(self):

        val_loss_best = 1e9
        time_since_improved = 0

        print('****Beginning Training****')
        print(f'Training on GPU:{next(self.model.parameters()).device}')

        for epoch in range(1, self.epochs+1):

            train_batch_loss = self._train_one_epoch().item()
            # print("LR: ", self.optimizer.param_groups[0]['lr'])
            self.scheduler.step()
            print(f"Epoch {epoch} training loss: {train_batch_loss}")

            if epoch % self.sets_between_eval == 0:

                val_loss = self._evaluate_one_epoch(epoch).item()
                print(f"Epoch {epoch} validation Loss: {val_loss}")
                torch.save(self.model.state_dict(), self.state_dict_folder.joinpath(f'epoch_{epoch}_model.pt'))
                print("*************************************************")

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    print("~~~~~~~~~~~~~~~SAVING BEST MODEL~~~~~~~~~~~~~~")
                    torch.save(self.model.state_dict(), self.state_dict_folder.joinpath('best_model.pt'))
                    time_since_improved = 0
                    print("*************************************************")
                else:
                    time_since_improved += 1

                if time_since_improved == self.early_stopping:
                    print(f"No model improvement in {self.early_stopping} epochs: terminating training")
                    break

                print("\n")

    def test(self):
        print('****Beginning Testing****')
        print(f'Training on GPU:{next(self.model.parameters()).device}')
        model_state_dict = torch.load(self.state_dict_folder.joinpath('best_model.pt'))
        self.model.load_state_dict(model_state_dict)

        test_loss = self._evaluate_one_epoch(epoch="test_set", test=True).item()
        print(f"Test Loss: {test_loss}")

    def _evaluate_one_epoch(self, epoch, test=False):

        self.model.eval()
        printed = False

        if test:
            set = "test"
            loader = self.test_loader
        else:
            set = "val"
            loader = self.val_loader

        with self.experiment.test() if self.comet else nullcontext():

            with torch.no_grad():
                total_loss = 0
                total_mase = 0
                total_mape = 0
                total_smape = 0
                total_mae = 0
                total_torch_mse = 0
                bootleg_mse = 0
                num_samples = 0
                for j, (inputs, targets) in enumerate(loader):

                    inputs = inputs.to(self.device)
                    targets = targets[:, self.i1:self.i2].to(self.device)

                    predictions = self.model(inputs)
                    loss = self.loss_func(predictions, targets)  # Loss function, and determine best target to evaluate for
                    total_loss += (loss * inputs.shape[0])

                    ######### test mae
                    l1_loss = self.torch_MSE(predictions, targets)
                    total_torch_mse += l1_loss
                    bootleg_mse += BootlegMSE(predictions, targets)

                    # batch_mape, batch_smape, batch_mae, batch_mase = all_metrics(predictions, targets)
                    # total_mape += (batch_mape * inputs.shape[0])
                    # total_mase += (batch_mase * inputs.shape[0])
                    # total_smape += (batch_smape * inputs.shape[0])
                    # total_mae += (batch_mae * inputs.shape[0])
                    num_samples += inputs.shape[0]

                    if not printed:
                        print("SAMPLE MODEL OUTPUTS FROM VAL SET:")
                        self.create_sample_outputs(predictions, targets, epoch, number=50)
                        printed = True

                    if test:
                        self.create_sample_outputs_new(predictions, targets, number=predictions.shape[0], round=j)

                epoch_loss = total_loss / num_samples
                # epoch_smape = total_smape / num_samples
                # epoch_mape = total_mape / num_samples
                # epoch_mase = total_mase / num_samples
                # epoch_mae = total_mae / num_samples
                epoch_torch_mse = total_torch_mse / (j+1)
                epoch_bootleg_mse = bootleg_mse / (j+1)

                # print(f"{set} SMAPE: ", epoch_smape.item())
                # print(f"{set} MAPE: ", epoch_mape.item())
                # print(f"{set} MASE: ", epoch_mase.item())
                # print(f"{set} MAE: ", epoch_mae.item())
                print(f"{set} Torch MSE: ", epoch_torch_mse.item())
                print(f"{set} Bootleg MSE: ", epoch_bootleg_mse)

                total_range_loss = 0  # Total range_loss adds up the range losses across all metrics
                for label in self.param_labels:
                    label_losses = self.metrics[label].compute()
                    total_range_loss += label_losses['RangeLoss']
                    print(f"{label} {set} losses: {label_losses}")
                    if self.comet:
                        self.comet_log_torchmetrics(label, label_losses)
                    self.metrics[label].reset()

                print(f"{set} Range Loss: ", total_range_loss.item())

                if self.comet:
                    self.experiment.log_metric(f'{set}_loss', epoch_loss)
                    # self.experiment.log_metric(f'{set}mape', epoch_mape)
                    # self.experiment.log_metric(f'{set}_smape', epoch_smape)
                    # self.experiment.log_metric(f'{set}_mase', epoch_mase)
                    # self.experiment.log_metric(f'{set}_mae', epoch_mae)
                    self.experiment.log_metric(f'{set}_total_range_loss', total_range_loss)

        return epoch_loss

    def _train_one_epoch(self):

        self.model.train()

        with self.experiment.train() if self.comet else nullcontext():

            total_loss = 0
            total_mase = 0
            total_mape = 0
            total_smape = 0
            total_mae = 0
            num_samples = 0
            bootleg_mse = 0

            for j, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)  # Bxfreq_range
                targets = targets[:, self.i1:self.i2].to(self.device)  # Bx12x2

                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                predictions = self.model(inputs)
                loss = self.loss_func(predictions, targets)
                total_loss += (loss * inputs.shape[0])
                bootleg_mse += BootlegMSE(predictions, targets)

                # # All tracking metrics
                # batch_mape, batch_smape, batch_mae, batch_mase = all_metrics(predictions, targets)
                # total_mape += (batch_mape * inputs.shape[0])
                # total_mase += (batch_mase * inputs.shape[0])
                # total_smape += (batch_smape * inputs.shape[0])
                # total_mae += (batch_mae * inputs.shape[0])
                num_samples += inputs.shape[0]

                # Propagate gradient
                loss.backward()

                # Update model_weights
                self.optimizer.step()

            epoch_loss = total_loss / num_samples
            # epoch_smape = total_smape / num_samples
            # epoch_mape = total_mape / num_samples
            # epoch_mase = total_mase / num_samples
            # epoch_mae = total_mae / num_samples
            epoch_bootleg_mse = bootleg_mse / (j + 1)
            print(f"Train Bootleg MSE: ", epoch_bootleg_mse)

            if self.comet:
                self.experiment.log_metric('train_loss', epoch_loss)
                # self.experiment.log_metric('train_mape', epoch_mape)
                # self.experiment.log_metric('train_smape', epoch_smape)
                # self.experiment.log_metric('train_mase', epoch_mase)
                # self.experiment.log_metric('train_mae', epoch_mae)

        return epoch_loss

    def comet_log_torchmetrics(self, label, results):

        # List of metrics is the losses included in the metric collection
        list_of_metrics = [m for m, _ in self.metrics[label].items()]
        for metric in list_of_metrics:
            self.experiment.log_metric(name=f'{label}_{metric}', value=results[metric])

    def create_sample_outputs(self, predictions, targets, epoch, number=3):
        # Prints sample outputs of all
        pd.set_option('display.max_columns', None)

        for i in range(number):
            sample_prediction = np.round(predictions[i].cpu().numpy(), decimals=3)
            sample_target = np.round(targets[i].cpu().numpy(), decimals=3)
            table = [sample_prediction, sample_target]
            df = pd.DataFrame(table, columns=self.output_labels, index=['predicted', 'actual'])
            csv_path = self.sample_output_folder.joinpath(f'epoch_{epoch}_sample_{i}.csv')
            df.to_csv(csv_path)
            if self.comet:
                self.experiment.log_table(csv_path, tabular_data=table, headers=self.output_labels)
            print(df)

    def create_output_labels(self):

        labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2', 'list_t_1',
                  'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']

        if self.target_param == "all":
            labels = labels
        elif self.target_param == "vsini":
            labels = labels[:2]
        elif self.target_param == "metal":
            labels = labels[2:4]
        elif self.target_param == "alpha":
            labels = labels[4:6]
        elif self.target_param == "temp":
            labels = labels[6:8]
        elif self.target_param == "log_g":
            labels = labels[8:10]
        elif self.target_param == "lumin":
            labels = labels[10:12]

        if self.target_param == "all":
            param_labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
        else:
            param_labels = [self.target_param]

        return labels, param_labels

    def create_dict_denom(self):
        labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
        ranges = [10, 1, 0.8, 3000, 3, 1.84186775877]
        dict_denom = {label: num for label, num in iter(zip(labels, ranges))}
        if self.target_param == 'all':
            dict_denom = dict_denom
        else:
            dict_denom = {self.target_param: dict_denom[self.target_param]}

        return dict_denom

    def get_data_indices(self):
        # Based on the order of the dataset array object
        # order = ["vsini", "metal", "alpha", "temp", "log_g", "lumin"]
        if self.target_param == "all":
            i1 = 0
            i2 = 12
        elif self.target_param == "vsini":
            i1 = 0
            i2 = 2
        elif self.target_param == "metal":
            i1 = 2
            i2 = 4
        elif self.target_param == "alpha":
            i1 = 4
            i2 = 6
        elif self.target_param == "temp":
            i1 = 6
            i2 = 8
        elif self.target_param == "log_g":
            i1 = 8
            i2 = 10
        elif self.target_param == "lumin":
            i1 = 10
            i2 = 12

        return i1, i2

    def create_sample_outputs_new(self, predictions, targets, round, number=50):
        # Prints sample outputs of all
        pd.set_option('display.max_columns', None)
        print(predictions.shape, targets.shape)

        min_predictions, _ = torch.min(predictions, dim=1)
        max_predictions, _ = torch.max(predictions, dim=1)
        min_targets, _ = torch.min(targets, dim=1)
        max_targets, _ = torch.max(targets, dim=1)

        # full_table = torch.concat((predictions, targets), dim=1)
        full_table = torch.stack((min_predictions, max_predictions, min_targets, max_targets), dim=1)
        full_table = np.round(full_table.cpu().numpy(), decimals=4)

        column_labels = []
        for prefix in ["prediction", "target"]:
            new_labels = [f"{prefix}_{label}" for label in self.output_labels]
            column_labels = [*column_labels, *new_labels]

        df = pd.DataFrame(full_table, columns=column_labels)
        csv_path = self.inference_output_folder.joinpath(f'target_outputs_full_sample_{round}.csv')
        df.to_csv(csv_path)
        print(df)


def check_exists(*folders):
    for folder in folders:
        if not folder.exists():
            folder.mkdir()

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
from utils.metrics_updated import MAPE_adjusted, SMAPE_adjusted, MASE, all_metrics
import utils.metrics_updated


# comet_ml.config.save(api_key="8EKfM7gNRhoWg9I2sQz6rcHls")



class Trainer(object):
    def __init__(self,
                 model,
                 parameters,
                 num_sets,
                 train_loader,
                 val_loader,
                 experiment_name="binary_stars"):

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Folder management - folder for model itself, as well as subfolders for state dictionary and sample outputs
        self.root_dir = Path('/media/sam/data/work/stars/configurations')
        self.folder = self.root_dir.joinpath(f'saved_models/{parameters["name"]}')
        self.state_dict_folder = self.folder.joinpath('state_dicts')
        self.sample_output_folder = self.folder.joinpath('sample_outputs')
        check_exists(self.folder, self.state_dict_folder, self.sample_output_folder)

        # Model Parameters
        self.model = model.to(self.device)

        # Optimizer Parameters
        lr = parameters['lr']
        wd = parameters['wd']
        self.optimizer = getattr(torch.optim, parameters['optimizer'])(model.parameters(), lr=lr, weight_decay=wd)

        # Loss function
        self.loss_func = getattr(utils.metrics_updated, parameters['loss'])

        self.torch_MAE = torch.nn.functional.l1_loss

        # Data and Training Parameters
        self.labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = parameters['epochs']
        self.sets_between_eval = parameters['sets_between_eval']
        self.early_stopping = parameters['early_stopping']

        # Comet hyperparameters and logging
        self.hyper_params = {
            'name': parameters['name'],
            'num_epochs': self.epochs,
            'optimizer': parameters['optimizer'],
            'learning_rate': lr,
            'weight_decay': wd,
            'num_sets': num_sets
        }

        if experiment_name:
            self.comet = True
            self.experiment = Experiment(project_name=experiment_name)
            self.experiment.log_parameters(self.hyper_params)
        else:
            self.comet = False

        # METRICS
        ranges = [10, 1, 0.8, 3000, 3, 1.84186775877]
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

    def _evaluate_one_epoch(self, epoch):

        self.model.eval()
        printed = False

        with self.experiment.test() if self.comet else nullcontext():

            with torch.no_grad():
                total_loss = 0
                total_mase = 0
                total_mape = 0
                total_smape = 0
                total_mae = 0
                total_torch_mae = 0
                for j, (inputs, targets) in enumerate(self.val_loader):

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    predictions = self.model(inputs)
                    loss, target_correct = self.loss_func(predictions, targets)  # Loss function, and determine best target to evaluate for
                    total_loss += loss

                    ######### test mae
                    l1_loss = self.torch_MAE(predictions, target_correct)
                    total_torch_mae += l1_loss


                    batch_mape, batch_smape, batch_mae, batch_mase = all_metrics(predictions, targets)
                    total_mape += batch_mape
                    total_mase += batch_mase
                    total_smape += batch_smape
                    total_mae += batch_mae

                    # This updates our torchmetrics losses, each batch
                    i = 0
                    for label in self.labels:  # Cycle through each label
                        prediction_selection = predictions[:, i:i + 2]
                        target_selection = target_correct[:, i:i + 2]
                        self.metrics[label].update(prediction_selection, target_selection)
                        i += 2

                    ### Print a certain number of output samples for the val set
                    if not printed:
                        print("SAMPLE MODEL OUTPUTS FROM VAL SET:")
                        self.create_sample_outputs(predictions, target_correct, epoch, number=3)
                        printed = True

                epoch_loss = total_loss / (j + 1)
                epoch_smape = total_smape / (j + 1)
                epoch_mape = total_mape / (j + 1)
                epoch_mase = total_mase / (j + 1)
                epoch_mae = total_mae / (j + 1)
                epoch_torch_mae = total_torch_mae / (j + 1)

                print("Val SMAPE: ", epoch_smape.item())
                print("Val MAPE: ", epoch_mape.item())
                print("Val MASE: ", epoch_mase.item())
                print("Val MAE: ", epoch_mae.item())
                print("Val Torch MAE: ", epoch_torch_mae.item())

                total_range_loss = 0  # Total range_loss adds up the range losses across all metrics
                for label in self.labels:
                    label_losses = self.metrics[label].compute()
                    total_range_loss += label_losses['RangeLoss']
                    if self.comet:
                        self.comet_log_torchmetrics(label, label_losses)

                print("Val Range Loss: ", total_range_loss.item())

                if self.comet:
                    self.experiment.log_metric('val_loss', epoch_loss)
                    self.experiment.log_metric('val_mape', epoch_mape)
                    self.experiment.log_metric('val_smape', epoch_smape)
                    self.experiment.log_metric('val_mase', epoch_mase)
                    self.experiment.log_metric('val_mae', epoch_mae)
                    self.experiment.log_metric('total_range_loss', total_range_loss)

        return epoch_loss

    def _train_one_epoch(self):

        self.model.train()

        with self.experiment.train() if self.comet else nullcontext():

            total_loss = 0
            total_mase = 0
            total_mape = 0
            total_smape = 0
            total_mae = 0

            for j, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)  # Bxfreq_range
                targets = targets.to(self.device)  # Bx12x2

                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                predictions = self.model(inputs)
                loss, _ = self.loss_func(predictions, targets)
                total_loss += loss

                # All tracking metrics
                batch_mape, batch_smape, batch_mae, batch_mase = all_metrics(predictions, targets)
                total_mape += batch_mape
                total_mase += batch_mase
                total_smape += batch_smape
                total_mae += batch_mae

                # Propagate gradient
                loss.backward()

                # Update model_weights
                self.optimizer.step()

            epoch_loss = total_loss / (j + 1)
            epoch_smape = total_smape / (j + 1)
            epoch_mape = total_mape / (j + 1)
            epoch_mase = total_mase / (j + 1)
            epoch_mae = total_mae / (j + 1)
            # print("Train SMAPE: ", epoch_smape)
            # print("Train MAPE: ", epoch_mape)
            # print("Train MASE: ", epoch_mase)
            # print("Train MAE: ", epoch_mae)

            if self.comet:
                self.experiment.log_metric('train_loss', epoch_loss)
                self.experiment.log_metric('train_mape', epoch_mape)
                self.experiment.log_metric('train_smape', epoch_smape)
                self.experiment.log_metric('train_mase', epoch_mase)
                self.experiment.log_metric('train_mae', epoch_mae)

        return epoch_loss

    def comet_log_torchmetrics(self, label, results):

        # List of metrics is the losses included in the metric collection
        list_of_metrics = [m for m, _ in self.metrics[label].items()]
        for metric in list_of_metrics:
            self.experiment.log_metric(name=f'{label}_{metric}', value=results[metric])

    def create_sample_outputs(self, predictions, targets, epoch, number=3):
        # Prints sample outputs of all
        pd.set_option('display.max_columns', None)
        labels = ['list_vsini_1', 'list_vsini_2', 'list_m_1', 'list_m_2', 'list_a_1', 'list_a_2', 'list_t_1',
                  'list_t_2', 'list_log_g_1', 'list_log_g_2', 'list_l_1', 'list_l_2']
        for i in range(number):
            sample_prediction = np.round(predictions[i].cpu().numpy(), decimals=3)
            sample_target = np.round(targets[i].cpu().numpy(), decimals=3)
            table = [sample_prediction, sample_target]
            df = pd.DataFrame(table, columns=labels, index=['predicted', 'actual'])
            csv_path = self.sample_output_folder.joinpath(f'epoch_{epoch}_sample_{i}.csv')
            df.to_csv(csv_path)
            self.experiment.log_table(csv_path, tabular_data=table, headers=labels)
            print(df)


def check_exists(*folders):
    for folder in folders:
        if not folder.exists():
            folder.mkdir()

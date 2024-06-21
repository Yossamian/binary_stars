from sys import prefix
import torch
from pathlib import Path
import time
import numpy as np
import pandas as pd
from comet_ml import Experiment
from contextlib import nullcontext
from utils.tmetrics import StarMetrics
from utils import round_to_sig, check_exists, get_dict_value, create_dict_denom, create_output_labels, get_data_indices
from datasets import denormalize
import utils.losses
from torch.optim.lr_scheduler import StepLR
import yaml
from math import log10, floor


# comet_ml.config.save(api_key="8EKfM7gNRhoWg9I2sQz6rcHls")


class Trainer(object):
    def __init__(self,
                 model,
                 parameters,
                 train_loader,
                 val_loader,
                 test_loader=None,
                 normalization_values=None,
                 experiment_name="binary_stars",
                 parallel=True):

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parameters = parameters

        # Folder management - folder for model itself, as well as subfolders for state dictionary and sample outputs
        self.root_dir = Path(f'/media/sam/data/work/stars/configurations/saved_models/{experiment_name}')
        self.folder = self.root_dir.joinpath(parameters["name"])
        self.state_dict_folder = self.folder.joinpath('state_dicts')
        self.sample_output_folder = self.folder.joinpath('sample_outputs')
        self.inference_output_folder = self.folder.joinpath('sample_outputs/inference_outputs')
        check_exists(self.root_dir, self.folder, self.state_dict_folder, self.sample_output_folder, self.inference_output_folder)
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

        # Loss function
        try:
            self.loss_func = getattr(utils.losses, parameters['loss'])(delta=parameters["huber_delta"])
        except KeyError:
            self.loss_func = getattr(utils.losses, parameters['loss'])()

        # Label parameters
        self.target_param = parameters["target_param"]
        self.i1, self.i2 = get_data_indices(self.target_param)
        self.output_labels, self.param_labels = create_output_labels(self.target_param)

        # Normalization details, if required
        if normalization_values is not None:
            self.normalize = parameters["normalize"]
            self.normalization_values = {}
            for key, value in normalization_values.items():
                self.normalization_values[key] = value[self.i1:self.i2]
        else:
            self.normalize = None

        # Dataloaders and training details
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = parameters['epochs']
        self.sets_between_eval = parameters['sets_between_eval']
        self.early_stopping = parameters['early_stopping']

        if experiment_name:
            self.comet = True
            self.experiment = Experiment(
                api_key="8EKfM7gNRhoWg9I2sQz6rcHls",
                project_name=experiment_name)
            self.experiment.log_parameters(parameters)
        else:
            self.comet = False

        # METRICS
        self.metrics = {}
        dict_denom = create_dict_denom(self.target_param)
        for key, range_value in dict_denom.items():
            self.metrics[key] = StarMetrics(range_value).to(self.device)

    def train(self):

        val_loss_best = 1e9
        time_since_improved = 0

        print('****Beginning Training****')
        print(f'Training on GPU:{next(self.model.parameters()).device}')

        for epoch in range(1, self.epochs + 1):

            train_batch_loss = self._train_one_epoch().item()
            print(f"Epoch {epoch} training loss: {train_batch_loss}")

            if epoch % self.sets_between_eval == 0:

                val_loss = self._evaluate_one_epoch(epoch).item()
                print(f"VALIDATION RUN: Epoch {epoch} validation Loss: {val_loss}")
                torch.save(self.model.state_dict(), self.state_dict_folder.joinpath(f'epoch_{epoch}_model.pt'))

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    torch.save(self.model.state_dict(), self.state_dict_folder.joinpath('best_model.pt'))
                    time_since_improved = 0
                    print("~~~~~~~~~~~~~~~SAVING BEST MODEL~~~~~~~~~~~~~~")
                else:
                    time_since_improved += 1

                if time_since_improved == self.early_stopping:
                    print(f"No model improvement in {self.early_stopping} epochs: terminating training")
                    break

                print("\n")

    def test(self):
        print('****Beginning Testing****')
        print(f'Testing on GPU:{next(self.model.parameters()).device}')
        model_state_dict = torch.load(self.state_dict_folder.joinpath('best_model.pt'))
        self.model.load_state_dict(model_state_dict)

        test_loss = self._evaluate_one_epoch(epoch="test_set", test=True).item()
        print(f"Test Loss: {test_loss}")

    def _evaluate_one_epoch(self, epoch, test=False):

        self.model.eval()

        if test:
            set = "test"
            loader = self.test_loader
        else:
            set = "val"
            loader = self.val_loader

        with self.experiment.test() if self.comet else nullcontext():

            with torch.no_grad():

                # Initialize losses
                total_loss = 0
                num_samples = 0

                # For generating output table
                predictions_full = []
                targets_full = []

                for j, (inputs, targets_all_labels) in enumerate(loader):

                    # Get inputs, targets
                    inputs = inputs.to(self.device)
                    targets = targets_all_labels[:, self.i1:self.i2].to(self.device)
                    inputs = inputs.float()
                    targets = targets.float()

                    # Run model, assess loss
                    predictions = self.model(inputs)
                    loss = self.loss_func(predictions, targets)
                    total_loss += (loss * inputs.shape[0])
                    num_samples += inputs.shape[0]

                    # Log losses
                    for label in self.param_labels:
                        self.metrics[label](predictions, targets)

                    # Log results for output table
                    predictions_full.append(np.array(predictions.detach().cpu()))
                    targets_full.append(np.array(targets_all_labels.detach().cpu()))

                # Calculate epoch loss
                epoch_loss = total_loss / num_samples

                # Print losses, send results to comet
                for label in self.param_labels:
                    label_losses = self.metrics[label].compute()
                    print(f"{label} {set} losses: {label_losses}")
                    if self.comet:
                        self.comet_log_torchmetrics(label, label_losses)
                    self.metrics[label].reset()

                # Log total loss to comet
                if self.comet:
                    self.experiment.log_metric(f'{set}_loss', epoch_loss)

                # Create final output arrays
                pred_array = np.row_stack(predictions_full)
                targ_array = np.row_stack(targets_full)
                self.create_output_table(pred_array, targ_array, split=set, epoch=epoch)

        return epoch_loss

    def _train_one_epoch(self):

        self.model.train()

        with self.experiment.train() if self.comet else nullcontext():

            total_loss = 0
            num_samples = 0

            for j, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)  # Bxfreq_range
                targets = targets[:, self.i1:self.i2].to(self.device)  # Bx12x2

                inputs = inputs.float()
                targets = targets.float()

                # clear the gradients
                self.optimizer.zero_grad()
                # compute the model output
                predictions = self.model(inputs)

                loss = self.loss_func(predictions, targets)
                total_loss += (loss * inputs.shape[0])
                num_samples += inputs.shape[0]

                # Propagate gradient
                loss.backward()

                # Update model_weights
                self.optimizer.step()

            epoch_loss = total_loss / num_samples

            if self.comet:
                self.experiment.log_metric('train_loss', epoch_loss)

        return epoch_loss

    def comet_log_torchmetrics(self, label, results):

        # List of metrics is the losses included in the metric collection
        list_of_metrics = [m for m, _ in self.metrics[label].items()]
        for metric in list_of_metrics:
            self.experiment.log_metric(name=f'{label}_{metric}', value=results[metric])

    def create_output_table(self, predictions, targets, split, print_now=False, epoch="", sig_digits=4):
        # Prints sample outputs of all
        pd.set_option('display.max_columns', None)
        print(predictions.shape, targets.shape)

        if self.normalize is not None:
            predictions = denormalize(predictions, mode=self.normalize, norm_values=self.normalization_values)
            targets = denormalize(targets, mode=self.normalize, norm_values=self.normalization_values)

        full_table = np.concatenate([predictions, targets], axis=1)

        # Create prediction, target labels for table
        target_labels = ['target_vsini_1', 'target_vsini_2', 'target_metal_1', 'target_metal_2', 'target_alpha_1', 'target_alpha_2',
                         'target_temp_1', 'target_temp_2', 'target_log_g_1', 'target_log_g_2', 'target_lumin_1', 'target_lumin_2',
                         'target_p_null', 'target_t_null', 'target_v1_null', 'target_v2_null', 'target_snr']
        pred_labels = [f"prediction_{label}" for label in self.output_labels]
        column_labels = [*pred_labels, *target_labels]

        # Create pandas dataframe
        df = pd.DataFrame(full_table, columns=column_labels)
        csv_path = self.inference_output_folder.joinpath(f'{split}_{epoch}_outputs_sample.csv')
        df.to_csv(csv_path)
        if print_now:
            print(df)


if __name__ == "__main__":
    print("Does nothin")
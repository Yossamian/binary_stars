import torch
from torch.utils.data import DataLoader
from datasets import split_dataset
import utils
import models
from pathlib import Path
from comet_ml import Experiment
from contextlib import nullcontext
from utils.metrics_updated import all_metrics
import pandas as pd
import numpy as np


def main(name):

    model_state_dict, parameters = load_model(name)

    print('PARAMETERS:')
    for key, value in parameters.items():
        print(key, value)

    if parameters['target_param'] == "all":
        num_outputs = 12
    else:
        num_outputs = 2

    # Load model
    model = getattr(models, parameters['model'])(num_outputs=num_outputs)
    model.load_state_dict(model_state_dict)
    print('MODEL ARCHITECTURE:')
    print(model)

    # Create datasets and dataloaders with given # of path files
    # num_sets = 2
    # paths = list(range(num_sets))
    paths = list(range(13))
    num_sets = len(paths)
    train_data, val_data, test_data = split_dataset(train=0.8, val=0.1, target_param=parameters['target_param'],
                                                    paths=paths)
    print(
        f'Dataset created with {len(train_data)} training examples, {len(val_data)} val examples, and {len(test_data)} test examples')

    # Create Dataloaders
    test_loader = DataLoader(test_data, shuffle=False, batch_size=256)

    inference = InferenceEngine(
        model=model,
        parameters=parameters,
        test_loader=test_loader,
    )

    inference.infer()


def load_model(experiment_name):
    """
    Load a model based on a specific experiment name
    Will return the best_model and the parameters_path
    :param experiment_name:
    :return:
    """
    folder = Path(f'/media/sam/data/work/stars/configurations/saved_models/{experiment_name}')
    # state_dict_path = Path(f'/media/sam/data/work/stars/configurations/saved_models/{experiment_name}/state_dicts/best_model.pt')
    state_dict_path = folder.joinpath('state_dicts/best_model.pt')
    parameters_path = folder.joinpath('parameters.pt')
    state_dict = torch.load(state_dict_path)
    params = torch.load(parameters_path)

    return state_dict, params


class InferenceEngine(object):

    def __init__(self,
                 model,
                 parameters,
                 test_loader,
                 experiment_name=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.test_loader = test_loader

        self.loss_func = getattr(utils.metrics_updated, parameters['loss'])
        self.torch_MSE = torch.nn.MSELoss()

        # Label parameters
        self.target_param = parameters["target_param"]
        self.output_labels, self.param_labels = self.create_output_labels()
        dict_denom = self.create_dict_denom()

        #Output folder
        self.root_dir = Path('/media/sam/data/work/stars/configurations')
        self.folder = self.root_dir.joinpath(f'saved_models/{parameters["name"]}')
        self.sample_output_folder = self.folder.joinpath('sample_outputs/inference_outputs')
        check_exists(self.folder, self.sample_output_folder)

        # METRICS
        self.metrics = {}
        for key, range_value in dict_denom.items():
            self.metrics[key] = utils.metrics.StarMetrics(range_value).to(self.device)

        if experiment_name:
            self.comet = True
            self.experiment = Experiment(project_name=experiment_name)
            self.experiment.log_parameters(self.hyper_params)
        else:
            self.comet = False

    def infer(self):

        self.model.eval()
        printed = False

        with self.experiment.test() if self.comet else nullcontext():

            with torch.no_grad():
                total_loss = 0
                total_mase = 0
                total_mape = 0
                total_smape = 0
                total_mae = 0
                total_torch_mse = 0

                for j, (inputs, targets) in enumerate(self.test_loader):

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    predictions = self.model(inputs)
                    loss, target_correct = self.loss_func(predictions,
                                                          targets)  # Loss function, and determine best target to evaluate for
                    total_loss += loss

                    ######### test mse
                    l1_loss = self.torch_MSE(predictions, target_correct)
                    total_torch_mse += l1_loss

                    batch_mape, batch_smape, batch_mae, batch_mase = all_metrics(predictions, targets)
                    total_mape += batch_mape
                    total_mase += batch_mase
                    total_smape += batch_smape
                    total_mae += batch_mae

                    # This updates our torchmetrics losses, each batch
                    i = 0
                    for label in self.param_labels:  # Cycle through each label
                        prediction_selection = predictions[:, i:i + 2]
                        target_selection = target_correct[:, i:i + 2]
                        self.metrics[label].update(prediction_selection, target_selection)
                        i += 2

                    ### Print a certain number of output samples for the val set
                    if not printed:
                        print("SAMPLE MODEL OUTPUTS FROM VAL SET:")
                        self.create_sample_outputs(predictions, target_correct, number=100)
                        printed = True

                epoch_loss = total_loss / (j + 1)
                epoch_smape = total_smape / (j + 1)
                epoch_mape = total_mape / (j + 1)
                epoch_mase = total_mase / (j + 1)
                epoch_mae = total_mae / (j + 1)
                epoch_torch_mse = total_torch_mse / (j + 1)

                print("Val SMAPE: ", epoch_smape.item())
                print("Val MAPE: ", epoch_mape.item())
                print("Val MASE: ", epoch_mase.item())
                print("Val MAE: ", epoch_mae.item())
                print("Val Torch MSE: ", epoch_torch_mse.item())

                total_range_loss = 0  # Total range_loss adds up the range losses across all metrics
                for label in self.param_labels:
                    label_losses = self.metrics[label].compute()
                    total_range_loss += label_losses['RangeLoss']
                    if self.comet:
                        self.comet_log_torchmetrics(label, label_losses)

                print("Test Range Loss: ", total_range_loss.item())

                if self.comet:
                    self.experiment.log_metric('val_loss', epoch_loss)
                    self.experiment.log_metric('val_mape', epoch_mape)
                    self.experiment.log_metric('val_smape', epoch_smape)
                    self.experiment.log_metric('val_mase', epoch_mase)
                    self.experiment.log_metric('val_mae', epoch_mae)
                    self.experiment.log_metric('total_range_loss', total_range_loss)

        return epoch_loss

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

    def create_sample_outputs(self, predictions, targets, number=50):
        # Prints sample outputs of all
        pd.set_option('display.max_columns', None)
        table = []
        indices = []
        for i in range(number):
            sample_prediction = np.round(predictions[i].cpu().numpy(), decimals=3)
            sample_target = np.round(targets[i].cpu().numpy(), decimals=3)
            table.append(sample_prediction)
            table.append(sample_target)
            indices.append(f'prediction {i}')
            indices.append(f'target {i}')


        df = pd.DataFrame(table, columns=self.output_labels, index=indices)
        csv_path = self.sample_output_folder.joinpath(f'target_outputs_sample.csv')
        df.to_csv(csv_path)
        if self.comet:
            self.experiment.log_table(csv_path, tabular_data=table, headers=self.output_labels)
        print(df)


def check_exists(*folders):
    for folder in folders:
        if not folder.exists():
            folder.mkdir()


if __name__ == "__main__":
    name = 'InceptionMultiNet_all_MASE_2023_02_21_1563'
    main(name)
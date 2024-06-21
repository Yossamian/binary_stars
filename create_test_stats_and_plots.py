import pandas as pd
from comet_ml import Experiment
from contextlib import nullcontext
from utils.tmetrics import StarMetrics
from utils import create_dict_denom, create_output_labels, get_data_indices
import utils.losses
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from datasets import GaiaDataset, load_data, normalize_labels, reorder_labels, reorder
import models
import yaml
from pathlib import Path
from utils.plotting_24may import create_scatter_avraham
from utils.test_set_analysis_24may import calculate_metrics

# comet_ml.config.save(api_key="8EKfM7gNRhoWg9I2sQz6rcHls")

def run_model(config_loc, dataset, dataset_labels, experiment_name, output_loc):

    print(f'Running main() for {config_loc}')

    # Read in parameters
    with open(config_loc, 'r') as f:
        parameters = yaml.safe_load(f)
    print('PARAMETERS:')
    for key, value in parameters.items():
        print(key, value)

    num_outputs = 12 if parameters['target_param'] == "all" else 2

    # Load model
    model = getattr(models, parameters['model'])(num_outputs=num_outputs)
    state_dict_loc = Path(config_loc).parent.joinpath('state_dicts').joinpath('best_model.pt')

    # Create Datasets, Dataloaders
    reorder_key = parameters["target_param"]
    dataset_labels = reorder(dataset_labels, target=reorder_key)
    # train_labels, val_labels, test_labels = reorder_labels(train_labels, val_labels, test_labels, parameters)
    data = GaiaDataset(dataset=dataset, dataset_labels=dataset_labels)
    data_loader = DataLoader(data, shuffle=True, batch_size=256)
    print(f'Dataset created with {len(data)} examples')

    trainer = ModelRunner(
        model=model,
        parameters=parameters,
        data_loader=data_loader,
        state_dict_loc=state_dict_loc,
        output_loc=output_loc,
        experiment_name=experiment_name,
    )

    trainer.test()

class ModelRunner(object):
    def __init__(self,
                 model,
                 parameters,
                 data_loader,
                 output_loc,
                 state_dict_loc,
                 experiment_name=None,
                 parallel=True):

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parameters = parameters

        # Folder locs
        self.output_loc = output_loc

        # Model Parameters
        if parallel:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        model_state_dict = torch.load(state_dict_loc)
        self.model.load_state_dict(model_state_dict)

        # Loss function
        try:
            self.loss_func = getattr(utils.losses, parameters['loss'])(delta=parameters["huber_delta"])
        except KeyError:
            self.loss_func = getattr(utils.losses, parameters['loss'])()

        # Label parameters
        self.target_param = parameters["target_param"]
        self.i1, self.i2 = get_data_indices(self.target_param)
        self.output_labels, self.param_labels = create_output_labels(self.target_param)

        # Dataloaders and training details
        self.data_loader = data_loader

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

    def test(self):
        print('****Beginning Testing****')
        print(f'Testing on GPU:{next(self.model.parameters()).device}')

        self.model.eval()

        with self.experiment.test() if self.comet else nullcontext():

            with torch.no_grad():

                # Initialize losses
                total_loss = 0
                num_samples = 0

                # For generating output table
                predictions_full = []
                targets_full = []
                targets_df = []

                for j, (inputs, targets_all_labels) in enumerate(self.data_loader):

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
                    targets_df.append(np.array(targets_all_labels.detach().cpu())[:, [self.i1, self.i2-1, -1]])

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
                targ_full_array = np.row_stack(targets_full)
                targ_array = np.row_stack(targets_df)
                self.create_output_table(pred_array, targ_array)

        print(epoch_loss)

    def comet_log_torchmetrics(self, label, results):

        # List of metrics is the losses included in the metric collection
        list_of_metrics = [m for m, _ in self.metrics[label].items()]
        for metric in list_of_metrics:
            self.experiment.log_metric(name=f'{label}_{metric}', value=results[metric])

    def create_output_table(self, predictions, targets, print_now=False, sig_digits=4):
        # Prints sample outputs of all
        pd.set_option('display.max_columns', None)
        print(predictions.shape, targets.shape)

        full_table = np.concatenate([predictions, targets], axis=1)

        # Create prediction, target labels for table
        target_labels = ['target_vsini_1', 'target_vsini_2', 'target_metal_1', 'target_metal_2', 'target_alpha_1', 'target_alpha_2',
                         'target_temp_1', 'target_temp_2', 'target_log_g_1', 'target_log_g_2', 'target_lumin_1', 'target_lumin_2',
                         'target_p_null', 'target_t_null', 'target_v1_null', 'target_v2_null', 'target_snr']

        target_labels = [f"target_{x}" for x in self.output_labels]

        pred_labels = [f"prediction_{label}" for label in self.output_labels]
        column_labels = [*pred_labels, *target_labels]

        column_labels = ["pred_1", "pred_2", "targ_1", "targ_2", "snr"]

        # Create pandas dataframe
        df = pd.DataFrame(full_table, columns=column_labels)
        df.to_csv(self.output_loc)
        if print_now:
            print(df)

if __name__ == "__main__":

    to_use = [
        "DenseNet_alpha_Huber_032_2024_02_11_79654",
        "DenseNet_log_g_Huber_12_2024_02_11_82179",
        "DenseNet_lumin_Huber_1_2024_02_11_14622",
        "DenseNet_metal_Huber_04_2024_02_11_15415",
        "DenseNet_temp_Huber_1200_2024_02_11_51226",
        "DenseNet_vsini_Huber_4_2024_02_11_3750"
    ]

    main_folder = "/media/sam/data/work/stars/configurations/saved_models/11FEB_stars_huber"

    for name in to_use:
        # config_loc = f"/media/sam/data/work/stars/configurations/saved_models/11FEB_stars_huber/{name}/config.yaml"
        # experiment_name = "STARS_FINAL"
        # train, train_labels, val, val_labels, test, test_labels = load_data()
        #
        # dataset = test
        # dataset_labels = test_labels
        output_loc = f"/media/sam/data/work/stars/new_snr_data/22MAYFINAL/{name}.csv"
        # run_model(config_loc, dataset, dataset_labels, experiment_name, output_loc)

        # Save graph
        graph_name = f"/media/sam/data/work/stars/new_snr_data/22MAYFINAL/graphs/{name}.png"
        create_scatter_avraham(graph_name, results_loc=output_loc)

        # Save csv stats
        new_csv_name = f"/media/sam/data/work/stars/new_snr_data/22MAYFINAL/metric_tables/{name}.csv"
        df = calculate_metrics(output_loc)
        df.to_csv(new_csv_name)
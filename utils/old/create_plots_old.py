import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools
from pathlib import Path
import yaml

def get_pd(choice, file_num=4, multi=False):
    """
    Returns the pandas df of results (first 1024 of them) from the test set
    """
    option_dict1 = {'temp': "DenseNet_temp_MSE_2023_04_11_1019",
                   "log_g": "DenseNet_log_g_MSE_2023_04_10_2320",
                   "metal": "DenseNet_metal_MSE_2023_04_10_2024",
                   "alpha": "DenseNet_alpha_MSE_2023_04_10_2230",
                   "vsini": "DenseNet_vsini_MSE_2023_04_11_475",
                   "lumin": "DenseNet_lumin_MSE_2023_04_10_1940",
                   "all": "DenseNet_all_MASE_2023_04_12_1331"
                   }

    option_dict2 = {'temp': "DenseNet_temp_BootlegMSE_2023_05_30_2832",
                    "log_g": "DenseNet_log_g_BootlegMSE_2023_05_30_861",
                    "metal": "DenseNet_metal_BootlegMSE_2023_05_30_2997",
                    "alpha": "DenseNet_alpha_BootlegMSE_2023_05_30_1234",
                    "vsini": "DenseNet_vsini_BootlegMSE_2023_05_31_345",
                    "lumin": "DenseNet_lumin_BootlegMSE_2023_05_30_1276",
                    "all": "DenseNet_all_MASE_2023_04_12_1331"
                    }

    if multi:
        model_name = "DenseNet_all_MASE_2023_04_12_1331"
    else:
        model_name = option_dict2[choice]

    loc = f"/media/sam/data/work/stars/configurations/saved_models/{model_name}/sample_outputs/inference_outputs/target_outputs_full_sample_{file_num}.csv"

    df = pd.read_csv(loc, index_col=0)

    if multi:
        keys = get_labels(choice)
        df = df[keys]

    return df


def get_labels(choice):
    if choice == "temp":
        z = "t"
    elif choice == "log_g":
        z = "log_g"
    elif choice == "metal":
        z = "m"
    elif choice == "alpha":
        z = "a"
    elif choice == "vsini":
        z = "vsini"
    elif choice == "lumin":
        z = "l"

    keys = [f'prediction_list_{z}_1', f'prediction_list_{z}_2',
            f'target_list_{z}_1', f'target_list_{z}_2']

    return keys


def get_samples(df, num_samples=100, seed=42, max_sample=6, round_to=2):
    """
    Returns a list of samples from the given df
    List is in format [pred1, pred2, target1, target2],
    where each item in list is a length-num vector
    """
    return_list = []
    random.seed(seed)

    indices = list(range(len(df)))
    random.shuffle(indices)

    accepted_indices = []
    counts = dict()
    for num in indices:

        include = True
        for key in df.keys():
            if "targ" in key:
                pred_value = round(df[key][num], round_to)
                if pred_value in counts.keys():
                    if counts[pred_value] > max_sample:
                        include = False

        if include:
            accepted_indices.append(num)
            for key in df.keys():
                if "targ" in key:
                    pred_value = round(df[key][num], round_to)
                    if pred_value in counts.keys():
                        counts[pred_value] += 1
                    else:
                        counts[pred_value] = 1





    # indices = random.sample(range(len(df)), num)
    for key in df.keys():
        values = df[key][accepted_indices[:num_samples]].values
        return_list.append(values)

    return return_list


def get_range(var):
    if var == "temp":
        a = 3200
        b = 6800
    elif var == "log_g":
        a = 2.5
        b = 6.5
    elif var == "metal":
        a = -0.25
        b = 1.25
    elif var == "alpha":
        a = -0.4
        b = 0.8
    elif var == "vsini":
        a = -1
        b = 11
    elif var == "lumin":
        a = 9.25
        b = 12.25
    else:
        raise ValueError("incorrect choice")

    return a, b


def create_scatter_avraham(variable,
                           num_samples,
                           multi=False,
                           combine_two_preds=False,
                           name=None,
                           results_loc=f"/media/sam/data/work/stars/configurations/saved_models/DenseNet_temp_MSE_2023_08_06_1454/sample_outputs/inference_outputs/test_target_outputs_full_sample.csv",
                           graph_loc="/media/sam/data/work/stars/graphs/21aug/"
                           ):


    df = pd.read_csv(results_loc, index_col=0)

    # df = get_pd(var, multi=multi, file_num=file_num)

    samples = get_samples(df, num_samples=num_samples)

    pred1 = samples[0]
    target1 = samples[1]
    pred2 = samples[2]
    target2 = samples[3]

    print(f"Langth of set {len(pred2)}")
    range_min, range_max = get_range(variable)
    line_pts = np.array([range_min, range_max])

    if name is None:
        name = variable

    if combine_two_preds:
        fig, ax = plt.subplots()
        fig.suptitle(name)
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.xlim(range_min, range_max)
        plt.ylim(range_min, range_max)
        ax.plot(line_pts, line_pts, color="black", linestyle="dashed", linewidth=0.5)

        ax.scatter(target1, pred1, c='b', marker="o")
        ax.scatter(target2, pred2, c='b', marker="o")
    else:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(variable)
        ax[0].set_xlabel("True Value")
        ax[1].set_xlabel("True Value")
        ax[0].set_ylabel("Predicted Value")
        ax[1].set_ylabel("Predicted Value")
        ax[0].scatter(target1, pred1, c='b', marker="o", label="Predicted1")
        ax[0].title.set_text("Predicted1")
        ax[1].scatter(target2, pred2, c='g', marker="o", label="Predicted2")
        ax[1].title.set_text("Predicted2")

    plt.grid()
    plt.savefig(f"{graph_loc}/{name}.png")
    # plt.show()


def create_scatter_1d(var, num, highlight_errors=True):
    df = get_pd(var)
    samples = get_samples(df, num=num)
    fig, ax = plt.subplots()
    plt.xlabel("Sample Points")
    plt.ylabel(var)

    labels = ["pred_1", "pred_2", "actual_1", "actual_2"]
    shapes = ["^", "x", "^", "x"]
    colors = ["b", "b", "g", "g"]

    for i in range(len(samples)):
        values = samples[i]
        color = colors[i]
        x = np.arange(num)

        if i < 2 and highlight_errors:
            error_indices = get_major_errors(values, samples[i + 2])
            values, errors = split_off_errors(values, error_indices)
            x, x_errors = split_off_errors(x, error_indices)
            ax.scatter(x_errors, errors, c='r', marker=shapes[i], label=f"{labels[i]}_error")

        ax.scatter(x, values, c=color, marker=shapes[i], label=labels[i])

    plt.title(var)
    plt.legend(loc="lower right")
    plt.show()


def create_scatter_2d(var1, var2, num, highlight_errors=True):
    df1 = get_pd(var1)
    df2 = get_pd(var2)
    samples1 = get_samples(df1, num=num)
    samples2 = get_samples(df2, num=num)

    labels = ["pred_1", "pred_2", "actual_1", "actual_2"]
    shapes = ["^", "x", "^", "x"]
    colors = ["b", "b", "g", "g"]

    fig, ax = plt.subplots()
    plt.xlabel(var1)
    plt.ylabel(var2)

    for i in range(len(samples1)):

        color = colors[i]
        values_x, values_y = samples1[i], samples2[i]
        # ax.scatter(samples1[i], samples2[i], c=color, marker=shapes[i], label=labels[i])

        if i < 2 and highlight_errors:
            error_indices_1 = get_major_errors(samples1[i], samples1[i + 2])
            error_indices_2 = get_major_errors(samples2[i], samples2[i + 2])
            error_indices = [*error_indices_1, *error_indices_2]
            values_x, errors_x = split_off_errors(values_x, error_indices)
            values_y, errors_y = split_off_errors(values_y, error_indices)
            ax.scatter(errors_x, errors_y, c="r", marker=shapes[i], label=f"{labels[i]}_error")

        ax.scatter(values_x, values_y, c=color, marker=shapes[i], label=labels[i])

    plt.title(f"{var1} vs {var2}")
    plt.legend(loc="lower right")
    plt.show()


def get_major_errors(pred, target, error=0.25):
    values = np.absolute(pred - target) / (np.absolute(target) + 1)
    major_errors = np.where(values > error)[0]
    return major_errors


def split_off_errors(values, error_indices):
    error_list = values[error_indices]
    good_list = np.delete(values, error_indices)

    return good_list, error_list


def get_max_min(var):
    df = get_pd(var)
    a = get_samples(df, 1000)
    target_array = np.concatenate((a[2], a[3]))
    df_max = np.max(target_array)
    df_min = np.min(target_array)
    var_values = np.unique(target_array)
    return df_min, df_max, var_values


# base_folder = "/media/sam/data/work/stars/configurations/saved_models/aug21/"
base_folder = "/media/sam/data/work/stars/configurations/saved_models/aug_07/"
for folder in Path(base_folder).iterdir():
    name = folder.name
    output_file = str(folder) + "/sample_outputs/inference_outputs/test_target_outputs_full_sample.csv"
    yaml_file = str(folder) + "/config.yaml"
    with open(yaml_file, 'r') as f:
        parameters = yaml.safe_load(f)

    var = parameters["target_param"]
    loss = parameters["loss"]
    norm = parameters["normalize"]

    # if loss == "MAE" and norm == 'z':
    # if var == 'temp' and norm is None:
    if norm is None:
        print(name, loss, norm)
        create_scatter_avraham(variable=var,
                               num_samples=250,
                               results_loc=output_file,
                               name=f'DenseNet {var} model, {loss} loss',
                               combine_two_preds=True,
                               graph_loc="/media/sam/data/work/stars/graphs/09FEB24")

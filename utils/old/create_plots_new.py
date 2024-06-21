import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools
from pathlib import Path
import yaml

def get_samples(df, target_val="target_a", num_samples=100, seed=42, max_sample=6, round_to=2):
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
        for key in df.keys():  # 4 keys: 2 predications and 2 targets
            if target_val in key:
                pred_value = round(df[key][num], round_to) # round value to nearest hundredth
                if pred_value in counts.keys():
                    if counts[pred_value] > max_sample:  # if current target value has already been used more than the max_sample number of times, skip to the next sample
                        include = False

        if include:
            accepted_indices.append(num)
            for key in df.keys():
                if target_val in key:
                    pred_value = round(df[key][num], round_to)
                    if pred_value in counts.keys():  # Count up the times that a certain target value shoes up
                        counts[pred_value] += 1
                    else:
                        counts[pred_value] = 1   # Track if a vlue has not shown up before

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
                           graph_loc="/media/sam/data/work/stars/graphs/21aug/",
                           has_snr=False
                           ):


    df = pd.read_csv(results_loc, index_col=0)

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

if __name__=="__main__":
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

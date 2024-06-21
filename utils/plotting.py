import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


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
                           name=None,
                           results_loc=f"/media/sam/data/work/stars/configurations/saved_models/DenseNet_temp_MSE_2023_08_06_1454/sample_outputs/inference_outputs/test_target_outputs_full_sample.csv",
                           graph_loc="/media/sam/data/work/stars/graphs/21aug/",
                           ):

    # read in the results dataset
    df = pd.read_csv(results_loc, index_col=0)
    if name is None:
        name = variable

    # Stack the two predictions on top of each other
    # resulting pd has four columns: index, pred, target, snr
    one = df[[f"prediction_{variable}_1", f"target_{variable}_1"]]
    two = df[[f"prediction_{variable}_2", f"target_{variable}_2"]]
    one.columns = ["pred", "targ"]
    two.columns = ["pred", "targ"]
    full = pd.concat([one, two], axis=0)
    snr = pd.concat([df["target_snr"], df["target_snr"]], axis=0)
    full = pd.concat([full, snr], axis=1)

    try:
        df = full.groupby("targ").sample(5)
        print("Unique values:", len(full.targ.unique()))
    except ValueError:
        df = full.groupby("targ").sample(3)
        print("Unique values:", len(full.targ.unique()))
        print("^^^ DOES NOT have 4 in each group")
    print(len(df))
    df = df.reindex().sample(300, random_state=42)

    # df = full.sample(500)
    print(f"Length of set {len(full)}, selected {len(df)}")

    # Get the diagonal line for plotting purposes
    range_min, range_max = get_range(variable)
    line_pts = np.array([range_min, range_max])

    # Get alpha value, for display
    max_snr = full.target_snr.max()
    alpha = snr/max_snr

    fig, ax = plt.subplots()
    fig.suptitle(name)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.xlim(range_min, range_max)
    plt.ylim(range_min, range_max)
    ax.plot(line_pts, line_pts, color="black", linestyle="dashed", linewidth=0.5)
    # ax.scatter(df["targ"], df["pred"], c=df["target_snr"], alpha=alpha, marker="o")
    ax.scatter(df["targ"], df["pred"], c="red", alpha=alpha, marker="o")

    plt.grid()
    plt.savefig(f"{graph_loc}/{name}.png")
    # plt.show()

if __name__ == "__main__":

    base_folder = "/media/sam/data/work/stars/configurations/saved_models/11FEB_stars_huber/"
    for folder in Path(base_folder).iterdir():
        name = folder.name
        output_file = str(folder) + "/sample_outputs/inference_outputs/test_target_outputs_sample.csv"
        yaml_file = str(folder) + "/config.yaml"
        with open(yaml_file, 'r') as f:
            parameters = yaml.safe_load(f)

        var = parameters["target_param"]
        loss = parameters["loss"]
        norm = parameters["normalize"]
        delta = parameters["huber_delta"]
        wd = parameters["wd"]
        lr = parameters["lr"]

        # if loss == "MAE" and norm == 'z':
        # if var == 'temp' and norm is None:
        if norm is None:
            print(name, loss, norm)
            create_scatter_avraham(variable=var,
                                   num_samples=250,
                                   results_loc=output_file,
                                   name=f'{name} {var} model, {loss} loss, {delta} {lr} {wd}',
                                   graph_loc="/media/sam/data/work/stars/graphs/snr/15MAR/")

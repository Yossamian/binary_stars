import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import seaborn as sns


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

def get_var(name):
    if "alpha" in name:
        return "alpha"
    if "temp" in name:
        return "temp"
    if "log_g" in name:
        return "log_g"
    if "metal" in name:
        return "metal"
    if "lumin" in name:
        return "lumin"
    if "vsini" in name:
        return "vsini"

def create_scatter_avraham(
        name,
        num_samples=150,
        num_buckets=4,
        num_samples_per_bucket=30,
        results_loc=f"/media/sam/data/work/stars/new_snr_data/experiment_results/DenseNet_alpha_Huber_008_2024_02_11_54985.csv",
):

    # read in the results dataset
    df = pd.read_csv(results_loc, index_col=0)

    # Stack the two predictions on top of each other
    # resulting pd has four columns: index, pred, target, snr
    one = df[[f"pred_1", f"targ_1"]]
    two = df[[f"pred_2", f"targ_2"]]
    one.columns = ["pred", "targ"]
    two.columns = ["pred", "targ"]
    full = pd.concat([one, two], axis=0)
    snr = pd.concat([df["snr"], df["snr"]], axis=0)
    full = pd.concat([full, snr], axis=1)

    try:
        df = full.groupby("targ").sample(5)
        print("Unique values:", len(full.targ.unique()))
    except ValueError:
        df = full.groupby("targ").sample(3)
        print("Unique values:", len(full.targ.unique()))
        print("^^^ DOES NOT have 4 in each group")

    ## Sample based on all SNRS
    # print(len(df))
    # df = df.reindex().sample(num_samples, random_state=42)
    # print(f"Length of set {len(full)}, selected {len(df)}")

    # Create SNR bins, bin labels
    # bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    bins = np.linspace(0, 400, num_buckets+1)
    bin_labels = []
    for i in range(1, len(bins)):
        bin_labels.append(f"[{bins[i - 1]} - {bins[i]})")
    df['snr_group'] = pd.cut(df.snr, bins, labels=bin_labels, right=False)

    df = df.groupby("snr_group").sample(num_samples_per_bucket)

    # Get the diagonal line for plotting purposes
    variable = get_var(results_loc)
    range_min, range_max = get_range(variable)
    line_pts = np.array([range_min, range_max])


    ### PLOT WITH VARYING COLORS FOR SNR GROUP
    # sns.relplot(data=df, x='targ', y='pred', hue='snr_group', aspect=1.61)

    ### PLOT WITH VARYING ALPHA FOR SNR GROUP
    alpha_step = 1/num_buckets
    alpha = 1/num_buckets
    for i in range(len(bin_labels)):
        sel_data = df.loc[df['snr_group'] == bin_labels[i]]
        plt.scatter(sel_data["targ"], sel_data["pred"], color="blue", alpha=alpha)
        # sns.relplot(data=sel_data, x='targ', y='pred', alpha=alpha, aspect=1.61)
        alpha += alpha_step

    plt.plot(line_pts, line_pts, color="black", linestyle="dashed", linewidth=0.5)
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid()
    plt.savefig(name)
    plt.close()

def get_bins(num, bins):
    x = num/bins
    return range(0, num, x)

if __name__ == "__main__":
    create_scatter_avraham("alpha")
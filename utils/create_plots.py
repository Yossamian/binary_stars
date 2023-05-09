import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def get_pd(choice):
    """
    Returns the pandas df of results (first 1024 of them) from the test set
    """
    option_dict = {'temp': "DenseNet_temp_MSE_2023_04_11_1019",
               "log_g": "DenseNet_log_g_MSE_2023_04_10_2320",
               "metal": "DenseNet_metal_MSE_2023_04_10_2024",
               "alpha": "DenseNet_alpha_MSE_2023_04_10_2230",
               "v_sin_i": "DenseNet_vsini_MSE_2023_04_11_475",
               "lumin": "DenseNet_lumin_MSE_2023_04_10_1940",
               "all": "DenseNet_all_MASE_2023_04_12_1331"
               }
    loc = f"/media/sam/data/work/stars/configurations/saved_models/{option_dict[choice]}/sample_outputs/inference_outputs/target_outputs_full_sample_0.csv"

    df = pd.read_csv(loc, index_col=0)

    return df


def get_samples(df, num=20, seed=42):
    """
    Returns a list of samples from the fiven df
    List is in format [pred1, pred2, target1, target2],
    where each item in list is a length-num vector
    """
    return_list = []
    random.seed(seed)
    indices = random.sample(range(len(df)), num)
    for key in df.keys():
        values = df[key][indices].values
        return_list.append(values)

    return return_list


def create_scatter_1d(var, num, highlight_errors=True):
    df = get_pd(var)
    samples = get_samples(df, num=num)
    fig, ax = plt.subplots()

    labels = ["pred_1", "pred_2", "actual_1", "actual_2"]
    shapes = ["^", "x", "^", "x"]
    colors = ["b", "b", "g", "g"]

    for i in range(len(samples)):
        values = samples[i]
        color = colors[i]
        x = np.arange(num)

        if i < 2 and highlight_errors:
            error_indices = get_major_errors(values, samples[i+2])
            values, errors = split_off_errors(values, error_indices)
            x, x_errors = split_off_errors(x, error_indices)
            ax.scatter(x_errors, errors, c='r', marker=shapes[i], label=f"{labels[i]}_error")

        ax.scatter(x, values, c=color, marker=shapes[i], label=labels[i])

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


    fig, ax = plt.subplots(),
    plt.xlabel(var1)
    plt.ylabel(var2)

    for i in range(len(samples1)):

        color = colors[i]
        values_x, values_y = samples1[i], samples2[i]
        # ax.scatter(samples1[i], samples2[i], c=color, marker=shapes[i], label=labels[i])

        if i < 2 and highlight_errors:
            error_indices_1 = get_major_errors(samples1[i], samples1[i+2])
            error_indices_2 = get_major_errors(samples2[i], samples2[i+2])
            error_indices = [*error_indices_1, *error_indices_2]
            values_x, errors_x = split_off_errors(values_x, error_indices)
            values_y, errors_y = split_off_errors(values_y, error_indices)
            ax.scatter(errors_x, errors_y, c="r", marker=shapes[i], label=f"{labels[i]}_error")

        ax.scatter(values_x, values_y, c=color, marker=shapes[i], label=labels[i])

    plt.legend(loc="lower right")
    plt.show()


def get_major_errors(pred, target, error=0.25):
    values = np.absolute(pred-target)/(np.absolute(target)+1)
    major_errors = np.where(values > error)[0]
    return major_errors


def split_off_errors(values, error_indices):

    error_list = values[error_indices]
    good_list = np.delete(values, error_indices)

    return good_list, error_list




create_scatter_2d("temp", "v_sin_i", 40)
# create_scatter_1d("v_sin_i", 50, highlight_errors=False)
# ds = get_pd("temp")
# f = get_samples(ds, 500)
# get_major_errors(f[0], f[2], 0.3)

#
#
#
#
#
# plt.style.use('_mpl-gallery')
#
# # make data:
# np.random.seed(1)
# x = np.arange(0, 10)
# D = np.random.gamma(4, size=(3, 50))
#
# # plot:
# fig, ax = plt.subplots()
#
# ax.eventplot(D, orientation="vertical", lineoffsets=x, linewidth=0.75)
# ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
#
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))
#
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools


def get_pd(choice, file_num=4, multi=False):
    """
    Returns the pandas df of results (first 1024 of them) from the test set
    """
    option_dict = {'temp': "DenseNet_temp_MSE_2023_04_11_1019",
                   "log_g": "DenseNet_log_g_MSE_2023_04_10_2320",
                   "metal": "DenseNet_metal_MSE_2023_04_10_2024",
                   "alpha": "DenseNet_alpha_MSE_2023_04_10_2230",
                   "vsini": "DenseNet_vsini_MSE_2023_04_11_475",
                   "lumin": "DenseNet_lumin_MSE_2023_04_10_1940",
                   "all": "DenseNet_all_MASE_2023_04_12_1331"
                   }

    if multi:
        model_name = "DenseNet_all_MASE_2023_04_12_1331"
    else:
        model_name = option_dict[choice]

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


def get_samples(df, num=20, seed=42):
    """
    Returns a list of samples from the given df
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


def create_scatter_avraham(var, num, multi=False, combined=False, file_num=0):
    df = get_pd(var, multi=multi, file_num=file_num)
    samples = get_samples(df, num=num)

    pred1 = samples[0]
    target1 = samples[2]
    pred2 = samples[1]
    target2 = samples[3]

    range_min, range_max = get_range(var)
    line_pts = np.array([range_min, range_max])

    if combined:
        fig, ax = plt.subplots()
        fig.suptitle(var)
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.xlim(range_min, range_max)
        plt.ylim(range_min, range_max)
        ax.plot(line_pts, line_pts, color="black", linestyle="dashed", linewidth=0.5)

        ax.scatter(target1, pred1, c='b', marker="o")
        ax.scatter(target2, pred2, c='b', marker="o")
    else:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(var)
        ax[0].set_xlabel("True Value")
        ax[1].set_xlabel("True Value")
        ax[0].set_ylabel("Predicted Value")
        ax[1].set_ylabel("Predicted Value")
        ax[0].scatter(target1, pred1, c='b', marker="o", label="Predicted1")
        ax[0].title.set_text("Predicted1")
        ax[1].scatter(target2, pred2, c='g', marker="o", label="Predicted2")
        ax[1].title.set_text("Predicted2")

    plt.grid()
    plt.show()


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


def create_scatter_avraham_full(num=150, multi=False, combined=False, file_num=0):

    x = [0, 1]
    y = [0, 1, 2]
    coords = list(itertools.product(x, y))
    params = ["temp", "log_g", "metal", "alpha", "vsini", "lumin"]

    fig, ax = plt.subplots(nrows=2, ncols=3)

    if combined:
        multi_choices = [False, True]
        color_choices = ['r', 'b']
        shape_choices = ['^', 'o']
        labels = ["multi_param", "single_param"]
    else:
        multi_choices = [multi]
        color_choices = ['b']
        shape_choices = ['o']
        if multi:
            labels = ["multi_param"]
        else:
            labels = ["single_param"]

    for j in range(len(multi_choices)):
        multi_selection = multi_choices[j]
        color_selection = color_choices[j]
        shape_selection = shape_choices[j]
        label = labels[j]

        for i in range(len(params)):
            param = params[i]
            x, y = coords[i]
            df = get_pd(param, multi=multi_selection, file_num=file_num)
            samples = get_samples(df, num=num)

            pred1 = samples[0]
            target1 = samples[2]
            pred2 = samples[1]
            target2 = samples[3]

            range_min, range_max = get_range(param)
            line_pts = np.array([range_min, range_max])

            ax[x, y].set_xlabel("True Value")
            ax[x, y].set_ylabel("Predicted Value")
            ax[x, y].set_title(param)
            ax[x, y].set_xlim(range_min, range_max)
            ax[x, y].set_ylim(range_min, range_max)
            ax[x, y].plot(line_pts, line_pts, color="black", linestyle="dashed", linewidth=0.5)
            ax[x, y].scatter(target1, pred1, c=color_selection, marker=shape_selection, label=label)
            ax[x, y].scatter(target2, pred2, c=color_selection, marker=shape_selection)
            ax[x, y].grid()
            ax[x, y].legend(loc="lower right")

    plt.show()

# create_scatter_2d("temp", "metal", 40)
# for var in ["temp", "log_g", "metal", "alpha", "vsini", "lumin"]:
#     create_scatter_avraham(var=var, num=150, multi=True, combined=True)

for a in [True, False]:
    create_scatter_avraham_full(num=150, multi=a, combined=False, file_num=0)

create_scatter_avraham_full(num=150, multi=False, combined=True, file_num=0)
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
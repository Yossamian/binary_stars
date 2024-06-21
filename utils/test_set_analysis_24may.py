import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(
        results_loc=f"/media/sam/data/work/stars/configurations/saved_models/DenseNet_temp_MSE_2023_08_06_1454/sample_outputs/inference_outputs/test_target_outputs_full_sample.csv",
):
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

    base_metrics = mse_and_mae(full)
    base_metrics = pd.DataFrame([np.array(base_metrics)], columns=["mse", "mae"])
    base_metrics.rename(index={0: 'Full'}, inplace=True)

    # Create SNR bins, bin labels
    bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    bin_labels = []
    for i in range(1, len(bins)):
        bin_labels.append(f"[{bins[i - 1]} - {bins[i]})")
    full['snr_group'] = pd.cut(df.snr, bins, labels=bin_labels, right=False)

    df = full.groupby("snr_group").apply(mse_and_mae)

    df = pd.concat([df, base_metrics], axis=0)

    return df

def mse_and_mae(df):
    mse = mean_squared_error(df['targ'], df['pred'])
    mae = mean_absolute_error(df['targ'], df['pred'])
    return pd.Series(dict(mse=mse, mae=mae))



if __name__ == "__main__":

    main_folder = "/media/sam/data/work/stars/new_snr_data/experiment_results"
    for path in Path(main_folder).iterdir():
        if ".csv" in path.name:
            name = path.stem
            new_csv = f"/media/sam/data/work/stars/new_snr_data/experiment_results/by_snr/{name}.csv"
            df = calculate_metrics(path)
            df.to_csv(new_csv)


    to_use = [
        "DenseNet_alpha_Huber_032_2024_02_11_79654",
        "DenseNet_log_g_Huber_12_2024_02_11_82179",
        "DenseNet_lumin_Huber_1_2024_02_11_14622",
        "DenseNet_metal_Huber_04_2024_02_11_15415",
        "DenseNet_temp_Huber_1200_2024_02_11_51226",
        "DenseNet_vsini_Huber_4_2024_02_11_3750"
    ]
    # results = "/media/sam/data/work/stars/new_snr_data/experiment_results/DenseNet_metal_Huber_01_2024_02_11_74824.csv"
    # df = calculate_metrics(results)
    # print(df)
    # base_folder = "/media/sam/data/work/stars/configurations/saved_models/11FEB_stars_huber/"
    # for folder in Path(base_folder).iterdir():
    #     name = folder.name
    #     output_file = str(folder) + "/sample_outputs/inference_outputs/test_target_outputs_sample.csv"
    #     yaml_file = str(folder) + "/config.yaml"
    #     with open(yaml_file, 'r') as f:
    #         parameters = yaml.safe_load(f)
    #
    #     var = parameters["target_param"]
    #     loss = parameters["loss"]
    #     norm = parameters["normalize"]
    #
    #     print(name, loss, norm)
    #     df = calculate_metrics(variable=var, results_loc=output_file, name=name)
    #     print(df)

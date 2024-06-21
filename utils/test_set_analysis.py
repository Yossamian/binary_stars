import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(
        variable,
        results_loc=f"/media/sam/data/work/stars/configurations/saved_models/DenseNet_temp_MSE_2023_08_06_1454/sample_outputs/inference_outputs/test_target_outputs_full_sample.csv",
        name=""
):
    df = pd.read_csv(results_loc, index_col=0)

    one = df[[f"prediction_{variable}_1", f"target_{variable}_1"]]
    two = df[[f"prediction_{variable}_2", f"target_{variable}_2"]]
    one.columns = ["pred", "targ"]
    two.columns = ["pred", "targ"]
    full = pd.concat([one, two], axis=0)
    snr = pd.concat([df["target_snr"], df["target_snr"]], axis=0)
    full = pd.concat([full, snr], axis=1)

    base_metrics = mse_and_mae(full)
    base_metrics = pd.DataFrame([np.array(base_metrics)], columns=["mse", "mae"])

    df = full.groupby("target_snr").apply(mse_and_mae)

    df = pd.concat([df, base_metrics], axis=0).rename(index={'200.0': '0', '300.0': '1', '0.0':'2'})

    # snr = pd.DataFrame([["200"], ["300"], ["combined"]], columns=["snr_value"]).rename(index={'200.0': '0', '300.0': '1', '0.0':'2'})
    #snr = pd.DataFrame([["200"], ["300"], ["combined"]], columns=["snr_value"])
    #names = pd.DataFrame([[name],[name], [name]], columns=["model_name"])
    #df = df.rename(index={'200.0': '0', '300.0': '1', '0.0':'2'})
    #df = pd.concat([df, snr, names], axis=1)
    #df['model'] = name

    return df

def mse_and_mae(df):
    mse = mean_squared_error(df['targ'], df['pred'])
    mae = mean_absolute_error(df['targ'], df['pred'])
    return pd.Series(dict(mse=mse, mae=mae))



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

        print(name, loss, norm)
        df = calculate_metrics(variable=var, results_loc=output_file, name=name)
        print(df)

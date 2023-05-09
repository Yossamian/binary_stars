
def get_model_loc(choice):
    root_folder = "/media/sam/data/work/stars/configurations/saved_models/"
    option_dict = {'temp': "DenseNet_temp_MSE_2023_04_11_1019",
                   "log_g": "DenseNet_log_g_MSE_2023_04_10_2320",
                   "metal": "DenseNet_metal_MSE_2023_04_10_2024",
                   "alpha": "DenseNet_alpha_MSE_2023_04_10_2230",
                   "v_sin_i": "DenseNet_vsini_MSE_2023_04_11_475",
                   "lumin": "DenseNet_lumin_MSE_2023_04_10_1940",
                   "all": "DenseNet_all_MASE_2023_04_12_1331"
                   }

    loc = f"{root_folder}/{option_dict[choice]}"

    return loc
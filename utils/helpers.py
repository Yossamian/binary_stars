import numpy as np
def check_exists(*folders):
    for folder in folders:
        if not folder.exists():
            folder.mkdir()


def round_to_sig(x, digits=5):
    sig = -np.floor(np.log10(np.abs(x))).astype(int)
    sig = np.sign(sig) * (np.abs(sig) + np.sign(sig)*(digits - 1))
    return np.round(x, sig)


def get_dict_value(dictionary, var, alternate=None):
    try:
        value = dictionary[var]
    except KeyError:
        value = alternate

    return value


def create_output_labels(target_param):

    if target_param == "all":
        labels = ['vsini_1', 'vsini_2', 'metal_1', 'metal_2', 'alpha_1', 'alpha_2', 'temp_1',
                  'temp_2', 'log_g_1', 'log_g_2', 'lumin_1', 'lumin_2']
        param_labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
    else:
        labels = [f"{target_param}_1", f"{target_param}_2"]
        param_labels = [target_param]

    return labels, param_labels


def create_dict_denom(target_param):
    labels = ['vsini', 'metal', 'alpha', 'temp', 'log_g', 'lumin']
    ranges = [10, 1, 0.8, 3000, 3, 1.84186775877]
    dict_denom = {label: num for label, num in iter(zip(labels, ranges))}
    if target_param == 'all':
        dict_denom = dict_denom
    else:
        dict_denom = {target_param: dict_denom[target_param]}

    return dict_denom


def get_data_indices(target_param):
    '''
    Based on the order of the dataset array object
    order = ["vsini", "metal", "alpha", "temp", "log_g", "lumin"]

    :return: i1, i2, the two indices that determine what data is relevant for this model
    '''

    if target_param == "all":
        i1, i2 = [0, 12]
    elif target_param == "vsini":
        i1, i2 = [0, 2]
    elif target_param == "metal":
        i1, i2 = [2, 4]
    elif target_param == "alpha":
        i1, i2 = [4, 6]
    elif target_param == "temp":
        i1, i2 = [6, 8]
    elif target_param == "log_g":
        i1, i2 = [8, 10]
    elif target_param == "lumin":
        i1, i2 = [10, 12]
    else:
        raise ValueError("")

    return i1, i2
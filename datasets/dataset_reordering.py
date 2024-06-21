import numpy as np


def reorder(dataset, target):
    """
    Reorder the dataset to put each two-star pair in min, max order
    """

    if target == "all":
        for j in range(dataset.shape[0]):
            for k in range(0, 12, 2):
                choice = np.argmin(dataset[j, k:k + 2])

                if choice == 0:
                    # If already in min, max order -> do nothing
                    pass
                else:
                    # Flip values if not in correct order
                    val_1 = dataset[j, k]
                    val_2 = dataset[j, k + 1]
                    dataset[j, k] = val_2
                    dataset[j, k + 1] = val_1

    else:
        if target == "vsini":
            i = 0
        elif target == "metal":
            i = 2
        elif target == "alpha":
            i = 4
        elif target == "temp":
            i = 6
        elif target == "log_g":
            i = 8
        elif target == "lumin":
            i = 10
        else:
            raise ValueError("not a correct type of sorting")

        for j in range(dataset.shape[0]):
            choice = np.argmin(dataset[j, i:i+2])

            if choice == 0:
                pass
            else:
                for k in range(0, 12, 2):
                    val1 = dataset[j, k]
                    val2 = dataset[j, k+1]
                    dataset[j, k] = val2
                    dataset[j, k+1] = val1

    return dataset


def reorder_labels(train_labels, val_labels, test_labels, parameters):

    if parameters["reorder"] is not None:
        if parameters["reorder"] == "all":
            reorder_key = "all"
        else:
            reorder_key = parameters["target_param"]

        train_labels = reorder(train_labels, target=reorder_key)
        val_labels = reorder(val_labels, target=reorder_key)
        test_labels = reorder(test_labels, target=reorder_key)

    return train_labels, val_labels, test_labels


if __name__ == "__main__":

    # generate_datasets("06_aug_b", num=15000, seed=42)
    #
    # target_folder ="/media/sam/data/work/stars/test_sets/new_snr_data/npy/"
    # files_list = ["/media/sam/data/work/stars/test_sets/new_snr_data/0.npz", "/media/sam/data/work/stars/test_sets/new_snr_data/1.npz"]
    # spec, labels = gen_snr_data(files_list, append_snr=True)
    # np.save(f"{target_folder}/spectra.npy", spec)
    # np.save(f"{target_folder}/labels.npy", labels)

    pass




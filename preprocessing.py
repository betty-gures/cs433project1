from pathlib import Path
import sys

import numpy as np

sys.path.append("../")
import helpers

base_dir = Path(__file__).parent

print("Loading raw data...")
x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids = helpers.load_csv_data(base_dir / "data/dataset", sub_sample=False)

# load and parse missing values
missing_values = []
with open(base_dir / "data/missing_values.txt", "r") as f:
    for line in f:
        line = line.strip().strip('"')  # remove whitespace and surrounding quotes
        # split by comma and convert to int
        numbers = [int(x.strip()) for x in line.split(",") if x.strip() != ""]
        missing_values.append(numbers)

assert x_train_orig.shape[1] == len(missing_values), "Mismatch between features and missing values"

# load variable types
variable_type = []
with open(base_dir / "data/variable_type.txt", "r") as f:
    for line in f:
        line = line.strip().strip('"')  # remove whitespace and surrounding quotes
        # split by comma and convert to int
        if line == "":
            variable_type.append([])
        else:
            variable_type.append(line)


def impute_missing_values(x):
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])
    return x

def preprocess(replace_nan_codes=True, one_hot_encoding=True, remove_invariant=True, save_dir=None, MAX_ONE_HOT_CATEGORIES=100):
    x_train = x_train_orig.copy()
    x_test = x_test_orig.copy()
    y_train = y_train_orig.copy()

    if replace_nan_codes:
        print("Replacing missing value codes with np.nan...")
        for feature_idx, nan_codes in enumerate(missing_values):
            for nan_code in nan_codes:
                x_train[x_train[:, feature_idx] == nan_code, feature_idx] = np.nan
                x_test[x_test[:, feature_idx] == nan_code, feature_idx] = np.nan

    if one_hot_encoding:
        print("Applying one-hot encoding...")
        nominal_features = np.where(np.array(variable_type)=="nominal")[0]
        one_hot_encoded = []
        for idx in nominal_features:
            unique_vals = np.unique(x_train[:, idx])
            if len(unique_vals) < MAX_ONE_HOT_CATEGORIES:
                train_cols = [(x_train[:, idx] == val).astype(float) for val in unique_vals if val != np.nan]
                test_cols = [(x_test[:, idx] == val).astype(float) for val in unique_vals if val != np.nan]
                x_train = np.column_stack([x_train, *train_cols])
                x_test = np.column_stack([x_test, *test_cols])
                one_hot_encoded.append(idx)
        x_train = np.delete(x_train, one_hot_encoded, axis=1)
        x_test = np.delete(x_test, one_hot_encoded, axis=1)

    print("Imputing missing values...")
    x_train = impute_missing_values(x_train)
    x_test = impute_missing_values(x_test)

    if remove_invariant:
        print("Removing invariant features...")
        stds_train = np.std(x_train, axis=0) # invariant features have zero std
        stds_test = np.std(x_test, axis=0) # invariant features have zero std
        x_train = x_train[:, (stds_train > 0) & (stds_test > 0)] # combine invariance from train and test
        x_test = x_test[:, (stds_train > 0) & (stds_test > 0)]


    # convert target to 0/1
    y_train = ((y_train + 1) / 2).astype(int)

    if save_dir is not None:
        print(f"Saving preprocessed data to {save_dir}...")
        np.savez("../data/dataset_prep/train.npz", x_train=x_train, y_train=y_train)
        np.savez("../data/dataset_prep/test.npz", x_test=x_test, test_ids=test_ids)
    
    return x_train, x_test, y_train, test_ids

def normalize_and_bias_data(x_train, x_test):
    """Standardize data and add bias term.
    Args:
        x_train: np.ndarray of shape (N_train, D)
        x_test: np.ndarray of shape (N_test, D)
    Returns:
        x_train_std: np.ndarray of shape (N_train, D+1)
        x_test_std: np.ndarray of shape (N_test, D+1)
    """

    # feature scaling by standardization
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train - mean) / std
    x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)

    # add bias term
    x_train = np.c_[np.ones((y_train.shape[0], 1)), x_train] 
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

def get_raw_data():
    return x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids
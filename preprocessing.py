import os
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.append("../")
import helpers

base_dir = Path(__file__).parent
dataset_dir = base_dir / "data/dataset"
x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids = None, None, None, None, None

# load and parse missing values
missing_values = []
with open(base_dir / "data/missing_values.txt", "r") as f:
    for line in f:
        line = line.strip().strip('"')  # remove whitespace and surrounding quotes
        # split by comma and convert to int
        numbers = [int(x.strip()) for x in line.split(",") if x.strip() != ""]
        missing_values.append(numbers)

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


def lazy_load_data():
    print("Loading raw data...")
    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(base_dir / "data/dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(base_dir / "data")
    global x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids
    if x_train_orig is None:
        x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids = helpers.load_csv_data(dataset_dir, sub_sample=False)
        assert x_train_orig.shape[1] == len(missing_values), "Mismatch between features and missing values"

def preprocess(replace_nan_codes=True, one_hot_encoding=True, save_dir=None, MAX_ONE_HOT_CATEGORIES=100):
    lazy_load_data()
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

    # convert target to 0/1
    y_train = ((y_train + 1) / 2).astype(int)

    if save_dir is not None:
        print(f"Saving preprocessed data to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        np.savez(f"{save_dir}/train.npz", x_train=x_train, y_train=y_train)
        np.savez(f"{save_dir}/test.npz", x_test=x_test, test_ids=test_ids)

    return x_train, x_test, y_train, test_ids


def impute_missing_values(train, test):
    col_means = np.nanmean(train, axis=0)
    inds = np.where(np.isnan(test))
    test[inds] = np.take(col_means, inds[1])
    return test


def normalize_and_bias_data(x_train, x_test=None):
    """Standardize data and add bias term.
    Args:
        x_train: np.ndarray of shape (N_train, D)
        x_test: np.ndarray of shape (N_test, D)
    Returns:
        x_train_std: np.ndarray of shape (N_train, D+1)
        x_test_std: np.ndarray of shape (N_test, D+1)
    """ 
    # missing data imputation
    x_train = impute_missing_values(x_train, x_train)
    if x_test is not None: x_test = impute_missing_values(x_train, x_test)
    
    # invariant feature removal
    stds_train = np.std(x_train, axis=0)
    x_train = x_train[:, (stds_train > 0)]

    # feature scaling by standardization
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    assert not np.any(std == 0), "At least one feature has zero standard deviation."
    x_train = (x_train - mean) / std
    
    # add bias term
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train] 

    if x_test is None:
        return x_train
    
    
    x_test = x_test[:, (stds_train > 0)]
    x_test = (x_test - mean) / std
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    return x_train, x_test

def pca(x_train, variance_threshold=0.2):
    X_centered = x_train - np.mean(x_train, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    k = np.searchsorted(cumulative_variance, variance_threshold) + 1

    X_pca = X_centered @ eigenvectors[:, :k]

    def apply_pca(X):
        X_test_centered = X - np.mean(x_train, axis=0)
        return X_test_centered @ eigenvectors[:, :k]
    
    return X_pca, apply_pca

def get_raw_data():
    return x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids
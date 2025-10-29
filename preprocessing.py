import hashlib
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
MAX_ONE_HOT_CATEGORIES=100
COL_REMOVE_MANUAL = np.array([0, 2, 8, 36, 38, 40, 45, 52, 60, 87, 90, 93, 104, 222, 225, 244, 246, 249, 251, 274, 282, 283, 288, 307, 308])

# load and parse missing values
missing_values = []
with open(base_dir / "data/metadata/missing_values.txt", "r") as f:
    for line in f:
        line = line.strip().strip('"')  # remove whitespace and surrounding quotes
        # split by comma and convert to int
        numbers = [int(x.strip()) for x in line.split(",") if x.strip() != ""]
        missing_values.append(numbers)

# load variable types
variable_type = []
with open(base_dir / "data/metadata/variable_type.txt", "r") as f:
    for line in f:
        line = line.strip().strip('"')  # remove whitespace and surrounding quotes
        # split by comma and convert to int
        if line == "":
            variable_type.append([])
        else:
            variable_type.append(line)

# load feature names
feature_names = []
with open(base_dir / "data/metadata/feature_names.csv", "r") as f:
    for line in f:
        feature_names.append(line.strip())
feature_names = np.array(feature_names)[1:] # remove ID


def remove_duplicate_columns(X_train, X_test=None):
    """Remove duplicate columns from X_train (and X_test if provided).
    Args:
        X_train: np.ndarray of shape (N_train, D)
        X_test: np.ndarray of shape (N_test, D) or None
    Returns:
        X_train_unique: np.ndarray of shape (N_train, D_unique)
        X_test_unique: np.ndarray of shape (N_test, D_unique) or None
        kept_idx: np.ndarray of shape (D_unique,), indices of kept columns
    """
    hashes = np.array([hashlib.md5(X_train[:, i].tobytes()).hexdigest() for i in range(X_train.shape[1])])
    _, idx = np.unique(hashes, return_index=True)
    kept_idx = np.sort(idx)
    X_train_unique = X_train[:, kept_idx]
    if X_test is not None:
        X_test_unique = X_test[:, kept_idx]
        return X_train_unique, X_test_unique, kept_idx
    return X_train_unique, kept_idx

def lazy_load_data():
    """Lazy load the raw data into memory if not already loaded.
    
    Args:
        None
        
    Returns:
        None
    """
    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(base_dir / "data/dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(base_dir / "data")
    global x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids
    if x_train_orig is None:
        print("Loading raw data...")
        x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids = helpers.load_csv_data(dataset_dir, sub_sample=False)
        assert x_train_orig.shape[1] == len(missing_values), "Mismatch between features and missing values"

def preprocess(replace_nan_codes=True, one_hot_encoding=True, save_dir=None):
    """Preprocess the dataset: replace missing value codes with np.nan, apply one-hot encoding, convert target to 0/1.
    Args:
        replace_nan_codes: whether to replace missing value codes with np.nan
        one_hot_encoding: whether to apply one-hot encoding to nominal features
        save_dir: if not None, directory to save preprocessed data
    Returns:
        x_train: np.ndarray of shape (N_train, D)
        x_test: np.ndarray of shape (N_test, D)
        y_train: np.ndarray of shape (N_train, 1)
        test_ids: np.ndarray of shape (N_test, ), IDs of test samples
        feature_names: np.ndarray of shape (D, ), names of features after preprocessing
    """

    lazy_load_data()
    x_train = x_train_orig.copy()
    x_test = x_test_orig.copy()
    y_train = y_train_orig.copy()
    feature_names_local = feature_names.copy()

    if replace_nan_codes:
        print("Replacing missing value codes with np.nan...")
        for feature_idx, nan_codes in enumerate(missing_values):
            for nan_code in nan_codes:
                x_train[x_train[:, feature_idx] == nan_code, feature_idx] = np.nan
                x_test[x_test[:, feature_idx] == nan_code, feature_idx] = np.nan

    if one_hot_encoding:
        print("Applying one-hot encoding...")
        nominal_features = np.where(np.array(variable_type) == "nominal")[0]

        new_train_cols = []
        new_test_cols = []
        new_feature_names = []

        for idx in nominal_features:
            if idx in COL_REMOVE_MANUAL: # do not one-hot encode manually removed cols
                continue

            unique_vals = np.unique(x_train[:, idx])
            unique_vals = unique_vals[~np.isnan(unique_vals)] # remove NaNs once
            if len(unique_vals) >= MAX_ONE_HOT_CATEGORIES:
                continue

            for val in unique_vals[:-1]: # avoid multicollinearity
                new_train_cols.append((x_train[:, idx] == val).astype(float))
                new_test_cols.append((x_test[:, idx] == val).astype(float))
                new_feature_names.append(f"{feature_names[idx]}_{val}")

        if new_train_cols:
            x_train = np.column_stack([x_train, *new_train_cols])
            x_test = np.column_stack([x_test, *new_test_cols])
            feature_names_local = np.concatenate([feature_names_local, new_feature_names])

    # convert target to 0/1
    y_train = ((y_train + 1) / 2).astype(int)

    if save_dir is not None:
        print(f"Saving preprocessed data to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        np.savez(f"{save_dir}/train.npz", x_train=x_train, y_train=y_train)
        np.savez(f"{save_dir}/test.npz", x_test=x_test, test_ids=test_ids)

    return x_train, x_test, y_train, test_ids, feature_names_local


def impute_missing_values(train, test):
    """Impute missing values in the test set using the mean of the training set.

    Args:
        train: np.ndarray of shape (N_train, D)
        test: np.ndarray of shape (N_test, D)
    Returns:
        test_imputed: np.ndarray of shape (N_test, D)
    """
    
    col_means = np.nanmean(train, axis=0)
    inds = np.where(np.isnan(test))
    test[inds] = np.take(col_means, inds[1])
    return test


def preprocess_splits(x_train, x_test=None, squared_features=True, feature_names=None):
    """Standardize data and add bias term.
    Args:
        x_train: np.ndarray of shape (N_train, D)
        x_test: np.ndarray of shape (N_test, D)
    Returns:
        x_train_std: np.ndarray of shape (N_train, D+1)
        x_test_std: np.ndarray of shape (N_test, D+1)
    """ 
    if feature_names is None: feature_names = np.arange(x_train.shape[1])
    one_hot_encoding = x_train.shape[1] > len(variable_type) # one-hot encoding was applied

    # missing data imputation
    x_train = impute_missing_values(x_train, x_train)
    if x_test is not None: x_test = impute_missing_values(x_train, x_test)
    
    # manual duplicates removal
    x_train = np.delete(x_train, COL_REMOVE_MANUAL, axis=1)
    remaining_var_types = np.delete(variable_type, COL_REMOVE_MANUAL)
    feature_names = np.delete(feature_names, COL_REMOVE_MANUAL)
    assert feature_names.shape[0] == x_train.shape[1], "Feature names and data shape mismatch after manual duplicate removal."

    # invariant feature removal
    stds_train = np.std(x_train, axis=0)
    remove = (stds_train < 1e-10)
    x_train = x_train[:, ~remove]
    remaining_var_types = np.delete(remaining_var_types, remove[:len(remaining_var_types)])
    feature_names = np.delete(feature_names, remove)
    assert feature_names.shape[0] == x_train.shape[1], "Feature names and data shape mismatch after invariant feature removal."

    # feature scaling by standardization
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    assert not np.any(std == 0), "At least one feature has zero standard deviation."
    x_train = (x_train - mean) / std
    
    # squaring continuous features
    if squared_features:
        squared_feature_idx = np.array([i for i, var_type in enumerate(remaining_var_types) if var_type in ["ordinal", "continuous"]])
        x_train = np.column_stack([x_train, x_train[:, squared_feature_idx] ** 2])
        squared_feature_names = [f"{feature_names[i]}_squared" for i in squared_feature_idx]
        feature_names = np.concatenate([feature_names, squared_feature_names])
    assert feature_names.shape[0] == x_train.shape[1], "Feature names and data shape mismatch after squaring features."
    
    if one_hot_encoding: 
        one_hot_encoded = [i for i, t in enumerate(remaining_var_types) if t == "nominal"]
        x_train = np.delete(x_train, one_hot_encoded, axis=1)
        feature_names = np.delete(feature_names, one_hot_encoded)
        assert feature_names.shape[0] == x_train.shape[1], "Feature names and data shape mismatch after removing one-hot encoded features."

    # add bias term
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    feature_names = np.concatenate([np.array(["bias"]), feature_names])

    if x_test is None:
        x_train, kept_idx = remove_duplicate_columns(x_train)
        return x_train, feature_names[kept_idx]

    x_test = np.delete(x_test, COL_REMOVE_MANUAL, axis=1)
    x_test = x_test[:, ~remove]
    x_test = (x_test - mean) / std
    if squared_features:
        x_test = np.column_stack([x_test, x_test[:, squared_feature_idx] ** 2])
    if one_hot_encoding: x_test = np.delete(x_test, one_hot_encoded, axis=1)
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    x_train, x_test, kept_idx = remove_duplicate_columns(x_train, x_test)
    return x_train, x_test, feature_names[kept_idx]

def pca(x_train, variance_threshold=0.2):
    """Apply PCA to reduce dimensionality while retaining specified variance.
    Args:
        x_train: np.ndarray of shape (N, D)
        variance_threshold: float, fraction of variance to retain
    Returns:
        x_train_pca: np.ndarray of shape (N, k) where k is number of components to retain specified variance
        apply_pca: function that can be used to apply the same PCA transformation to new data
    """
    X_mean = np.mean(x_train, axis=0)
    X_centered = x_train - X_mean
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
        X_test_centered = X - X_mean
        X_test_pca = X_test_centered @ eigenvectors[:, :k]
        return X_test_pca
    
    return X_pca, apply_pca

def get_raw_data():
    """Returns the raw (unprocessed) data."""
    lazy_load_data()
    return x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids
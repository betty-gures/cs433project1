
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from preprocessing import preprocess
from models import OrdinaryLeastSquares
from model_selection import cross_validation


ABLATION_MODEL = OrdinaryLeastSquares
NUM_SAMPLES = int(1e6)

def run_preprocessing_ablation(kwarg_list, num_samples = NUM_SAMPLES, num_folds = 5):
    mean_f1 = []

    for cv_kw in kwarg_list:
        print("Running with args:", cv_kw)
        x_train, _, y_train, *_ = preprocess(**cv_kw)
        
        cv_results = cross_validation(x_train[:num_samples], y_train[:num_samples], num_folds=num_folds, verbose=False, model_class=ABLATION_MODEL)
        mean_f1.append(np.mean(cv_results.f1_scores))

    for i, (cv_kw, mean) in enumerate(zip(kwarg_list, mean_f1)):
        print(cv_kw)
        print(f"{mean - mean_f1[0]}" if i > 0 else mean)

# Replace nan codes
kwarg_list = [{"replace_nan_codes": True}, {"replace_nan_codes": False}]
run_preprocessing_ablation(kwarg_list)

# One-hot encoding
kwarg_list = [{"one_hot_encoding": True}, {"one_hot_encoding": False}]
run_preprocessing_ablation(kwarg_list)


x_train, _, y_train, *_ = preprocess() # fixed preprocessed data
def run_model_ablation(kwarg_list, num_samples = NUM_SAMPLES):
    mean_f1 = []

    for cv_kw in kwarg_list:
        print("Running with args:", cv_kw)
        cv_results = cross_validation(x_train[:num_samples], y_train[:num_samples], verbose=False, model_class=ABLATION_MODEL, **cv_kw)
        mean_f1.append(np.mean(cv_results.f1_scores))

    for i, (cv_kw, mean) in enumerate(zip(kwarg_list, mean_f1)):
        print(cv_kw)
        print(f"{mean - mean_f1[0]}" if i > 0 else mean)

# Squared features
kwarg_list = [{"squared_features": True}, {"squared_features": False}]
run_model_ablation(kwarg_list)

# Weighting
kwarg_list = [{"weighting": True}, {"weighting": False}]
run_model_ablation(kwarg_list)

# Threshold tuning
kwarg_list = [{"tune_threshold": True}, {"tune_threshold": False}]
run_model_ablation(kwarg_list)

# Both weighting and threshold tuning
kwarg_list = [{"tune_threshold": True, "weighting": True}, {"tune_threshold": False, "weighting": False}]
run_model_ablation(kwarg_list)

def run_preprocessing_model_ablation(pre_kwarg_list, model_kwarg_list, num_samples = NUM_SAMPLES, num_folds = 5):
    mean_f1 = []

    for pre_kw, model_kw in zip(pre_kwarg_list, model_kwarg_list):
        print("Running with args:", pre_kw, model_kw)
        x_train, _, y_train, *_ = preprocess(**pre_kw)

        cv_results = cross_validation(x_train[:num_samples], y_train[:num_samples], num_folds=num_folds, verbose=False, model_class=ABLATION_MODEL, **model_kw)
        mean_f1.append(np.mean(cv_results.f1_scores))

    for i, (cv_kw, model_kw, mean) in enumerate(zip(pre_kwarg_list, model_kwarg_list, mean_f1)):
        print(cv_kw, model_kw)
        print(f"{mean - mean_f1[0]}" if i > 0 else mean)

# Both one-hot encoding and squared features
pre_kwarg_list = [{"one_hot_encoding": True}, {"one_hot_encoding": False}]
model_kwarg_list = [{"squared_features": True}, {"squared_features": False}]
run_preprocessing_model_ablation(pre_kwarg_list, model_kwarg_list)

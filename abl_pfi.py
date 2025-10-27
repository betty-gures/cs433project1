from collections import defaultdict
import os

import numpy as np

from implementations import sigmoid
from metrics import f_score
from models import OrdinaryLeastSquares, test_val_split
from preprocessing import preprocess, preprocess_splits


os.makedirs("results", exist_ok=True)
out_path = "results/pfi.txt"
num_seeds = 5
initial_seed = 42

def get_f1(model, X, y):
    """Get F1 score of model on data X, y.
    
    Args:
        model: trained model
        X: input data
        y: true labels
    Returns:
        F1 score as float
    """
    preds = sigmoid(X @ model.weights)
    preds = (preds >= model.decision_threshold).astype(int)
    return f_score(preds, y).item()

def get_weight_stats(model, X, y, feature_names, topk=100):
    """Get permutation feature importances for the topk features by absolute weight.
    Args:
        model: trained model
        X: input data
        y: true labels
        feature_names: list of feature names
        topk: number of top features to consider
    Returns:
        dict mapping feature name to (relative delta F1, weight)
    """
    original_f1 = get_f1(model, X, y)
    print(f"Original F1: {original_f1}")
    topk_idx = np.argsort(np.abs(model.weights))[-topk:][::-1]
    deltas = {}
    for idx in topk_idx:
        X_perm = X.copy()
        X_perm[:, idx] = np.random.permutation(X[:, idx])
        rel_delta = (original_f1 - get_f1(model, X_perm, y)) / original_f1
        deltas[feature_names[idx]] = (rel_delta, model.weights[idx])
    return deltas

# preprocessing
x_train, _, y_train, _, feature_names = preprocess()

# drawing seeds from initial seed
np.random.seed(initial_seed)
seeds = list(np.random.randint(0, 2**16 - 1, size=num_seeds))

# training model and getting PFI
all_deltas = []
for seed in seeds:
    print(f"Using seed {seed}...")
    X_train_split, y_train_split, X_val, y_val = test_val_split(np.random.default_rng(seed), x_train, y_train)
    print("Training model...")
    model = OrdinaryLeastSquares()
    model.hyperparameter_tuning(X_train_split, y_train_split, verbose=True)
    model.train(X_train_split, y_train_split)

    print("Normalizing validation data...")
    _, X, feature_names_split = preprocess_splits(X_train_split, X_val, squared_features=model.squared_features, feature_names=feature_names)
    print("Calculating permutation feature importances...")
    all_deltas.append(get_weight_stats(model, X, y_val, feature_names_split))

delta_collect = defaultdict(list)
weight_collect = defaultdict(list)
for d in all_deltas:
    for fname, (rel_delta, w) in d.items():
        delta_collect[fname].append(rel_delta)
        weight_collect[fname].append(w)

# compute mean and std for each feature
deltas = {}
weights = {}
for fname, vals in delta_collect.items():
    deltas[fname] = (float(np.mean(vals)), float(np.std(vals)))
    weights[fname] = (float(np.mean(weight_collect[fname])), float(np.std(weight_collect[fname])))

# sort deltas by delta value descending
deltas = dict(sorted(deltas.items(), key=lambda item: item[1], reverse=True))
print("Feature importances (by decrease in F1 when permuted):")
for fname, (mean, std) in deltas.items():
    print(f"Feature {fname}: ΔF1 = {mean} ± {std}, w: {weights[fname][0]} ± {weights[fname][1]}")

# write results to file
with open(out_path, "w", encoding="utf-8") as f:
    for fname, (mean, std) in deltas.items():
        w_mean, w_std = weights.get(fname, (None, None))
        f.write(f"Feature {fname}: ΔF1 = {mean} ± {std}, w: {w_mean} ± {w_std}\n")
print(f"Wrote {len(deltas)} lines to {out_path}")
from dataclasses import dataclass

from typing import List, Any
import numpy as np
from metrics import f_score, auc_roc

@dataclass
class CVResult:
    f1_scores: List[float]
    f2_scores: List[float]
    auc_rocs: List[float]
    train_result: Any

def cross_validation(model_class, x_train, y_train, num_folds=5, seed=42, verbose=False, **model_args):
    num_samples_total = x_train.shape[0]
    indices = np.arange(num_samples_total)

    np.random.seed(seed)
    np.random.shuffle(indices)
    fold_sizes = np.full(num_folds, num_samples_total // num_folds, dtype=int)
    fold_sizes[:num_samples_total % num_folds] += 1
    current = 0
    f1_scores, f2_scores, aucrocs, train_results = [], [], [], []

    for fold_idx, fold_size in enumerate(fold_sizes):
        print(f"Starting fold {fold_idx + 1}/{num_folds} with {fold_size} samples")
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
 
        model = model_class(**model_args) # initialize a new model for each fold
        train_results.append(model.train(x_train[train_idx], y_train[train_idx], verbose=verbose, metric=f_score)) # train the model
        y_val_pred = model.predict(x_train[val_idx]) # predict on validation set

        y_val = y_train[val_idx]
        f1_scores.append(f_score(y_val_pred, y_val))
        f2_scores.append(f_score(y_val_pred, y_val, beta=2))
        aucrocs.append(auc_roc(model.predict(x_train[val_idx], probability=True), y_val))

        current += fold_size
    return CVResult(f1_scores, f2_scores, aucrocs, train_results)
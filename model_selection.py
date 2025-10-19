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


def find_best_threshold(scores, true, metric, num_thresholds=201, verbose=False):
    """Find the best decision threshold for binary classification based on a given metric.
    Args:
        scores: np.ndarray of shape (N,), predicted probabilities or scores
        true: np.ndarray of shape (N,), true binary labels (0 or 1)
        metric: function(pred, true) -> float, metric to optimize
        num_thresholds: int, number of thresholds to evaluate
        """
    thresholds = np.linspace(0, 1, num_thresholds)
    metric_vals = [] # collects the metric values for each threshold
    
    for t in thresholds:
        pred = (scores >= t).astype(int)
        metric_vals.append(metric(pred, true))
    best_threshold = thresholds[np.argmax(metric_vals)]
    if verbose: print(f"Best threshold: {best_threshold} with score {np.max(metric_vals)}")
    return best_threshold

def test_val_split(rng, X, y, val_ratio=0.2):
    """Split the data into training and validation sets.
     Args:
        rng: np.random.Generator
        X: np.ndarray of shape (N, D)
        y: np.ndarray of shape (N, )
        val_ratio: float, ratio of validation data
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    num_val = int(X.shape[0] * val_ratio)
    indices = rng.permutation(X.shape[0])
    val_idx = indices[:num_val]
    train_idx = indices[num_val:]
    X_val, y_val = X[val_idx], y[val_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    return X_train, y_train, X_val, y_val

def score_by_group(y_true, pred, probs, group_attr):
    """Compute metrics grouped by a specific attribute. 
    Args:
        y_true: np.ndarray of shape (N,), true binary labels (0 or 1
        pred: np.ndarray of shape (N,), predicted binary labels (0 or 1)
        probs: np.ndarray of shape (N,), predicted probabilities
        group_attr: np.ndarray of shape (N,), attribute to group by (can contain NaN values)
    Returns:
        f1_scores: dict mapping group value to F1 score
        f2_scores: dict mapping group value to F2 score
        auc_rocs: dict mapping group value to AUROC
    """
    f1_scores, f2_scores, auc_rocs = {}, {}, {}
    
    for attr_val in np.unique(group_attr):
        mask = group_attr == attr_val if not np.isnan(attr_val) else np.isnan(group_attr)
        group_name = str(attr_val) if not np.isnan(attr_val) else "nan"
        f1_scores[group_name] = f_score(pred[mask], y_true[mask])
        f2_scores[group_name] = f_score(pred[mask], y_true[mask], beta=2)
        auc_rocs[group_name] = auc_roc(probs[mask], y_true[mask])
    return f1_scores, f2_scores, auc_rocs

def cross_validation(x_train, y_train, model_class, num_folds=5, seed=42, verbose=False, scoring_groups=None, **model_args):
    """Perform k-fold cross-validation.
    
    Args:
        x_train: shape=(N, D)
        y_train: shape=(N, 1)
        model_class: class of the model to be trained, must implement 'train' and '
        num_folds: number of folds for cross-validation
        seed: random seed for shuffling data
        verbose: whether to print progress messages
        **model_args: additional arguments to pass to the model constructor
    
    Returns:
        CVResult: dataclass containing lists of f1 scores, f2 scores, auc-rocs, and training results for each fold
    """
    num_samples_total = x_train.shape[0]
    indices = np.arange(num_samples_total)

    np.random.seed(seed)
    np.random.shuffle(indices)
    fold_sizes = np.full(num_folds, num_samples_total // num_folds, dtype=int)
    fold_sizes[:num_samples_total % num_folds] += 1
    current = 0
    f1_scores, f2_scores, aucrocs, train_results = [], [], [], []

    for fold_idx, fold_size in enumerate(fold_sizes):
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        print(f"Starting fold {fold_idx + 1}/{num_folds} with {train_idx.shape[0]} samples")
 
        model = model_class(**model_args) # initialize a new model for each fold
        train_results.append(model.train(x_train[train_idx], y_train[train_idx], verbose=verbose)) # train the model
        y_val_pred = model.predict(x_train[val_idx]) # predict on validation set
        y_probs = model.predict(x_train[val_idx], scores=True)

        y_val = y_train[val_idx]
        if scoring_groups is None:
            f1_scores.append(f_score(y_val_pred, y_val))
            f2_scores.append(f_score(y_val_pred, y_val, beta=2))
            aucrocs.append(auc_roc(y_probs, y_val))
        else:
            f1_dict, f2_dict, aucroc_dict = score_by_group(y_val, y_val_pred, y_probs, scoring_groups[val_idx])
            f1_scores.append(f1_dict)
            f2_scores.append(f2_dict)
            aucrocs.append(aucroc_dict)

        current += fold_size
    return CVResult(f1_scores, f2_scores, aucrocs, train_results)
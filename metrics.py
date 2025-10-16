import numpy as np

def f_score(pred, true, beta=1):
    """
    Compute Fbeta score

    Args:

    """
    assert np.all((pred == 0) | (pred == 1)) and np.all((true == 0) | (true == 1)), "pred and y must be binary (0 or 1)"
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0

def auc_roc(y_score, true):
    """
    Compute AUROC
    
    Args:
    y_score : np.ndarray
        Predicted probabilities.
    true : np.ndarray
        Binary ground truth labels (0 or 1).

    Returns:
    float
        Area under the ROC curve.
    """
    # Sort by predicted scores descending
    desc_sort_indices = np.argsort(-y_score)
    true = true[desc_sort_indices]
    y_score = y_score[desc_sort_indices]

    # Compute true positive and false positive counts
    tps = np.cumsum(true)
    fps = np.cumsum(1 - true)

    # Normalize to get TPR and FPR
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Add (0,0) at the start and (1,1) at the end for integration
    tpr = np.concatenate(([0], tpr, [1]))
    fpr = np.concatenate(([0], fpr, [1]))

    # Compute AUC via trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc

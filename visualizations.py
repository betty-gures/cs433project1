# Plotting functions for model evaluation and fairness analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc(score, y, steps=101):
    """Plots an ROC curve

    Args:
        score: np.ndarray of shape (n_samples,), predicted scores
        y: np.ndarray of shape (n_samples,), true binary labels (0 or 1)
        steps: int, number of threshold steps between 0 and 1
    """
    thresholds = np.linspace(0, 1, steps)
    tpr = []
    fpr = []
    
    for thresh in thresholds:
        y_pred = (score >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))
        TN = np.sum((y_pred == 0) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # recall, sensitivity
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # 1 - specificity
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TPR
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Threshold: {thresh:.2f}, F1 Score: {F1:.4f}")
        tpr.append(TPR)
        fpr.append(FPR)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (NumPy only)")
    plt.show()

def plot_losses(train_losses, val_losses):
    """Plots training and validation losses over iterations.
    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
    """
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_class_distribution_by_group(y, group_attr, annotate=True, save_dir=None):
    """
    Plots the class distribution of y segmented by group_attr.
    Args:
        y: np.ndarray of shape (n_samples,), binary class labels (0 or 1)
        group_attr: np.ndarray of shape (n_samples,), categorical group attribute
        annotate: bool, whether to annotate bars with relative frequencies
        save_dir: str or None, if provided, saves the plot to this directory
    """
    mask = ~np.isnan(group_attr)
    group_clean = group_attr[mask]
    y_clean = y[mask]
    plt.figure()
    g = sns.histplot(x=group_clean, hue=y_clean, multiple="stack", binwidth=0.5)

    if annotate:
        totals = np.array([bar.get_height() for container in g.containers for bar in container if bar.get_height() > 0]).reshape(2, -1).sum(axis=0)
        for container in g.containers:
            i = 0
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    g.annotate(
                        f'{height*100/totals[i]:.1f}%',  # relative frequency
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2),
                        ha='left',
                        va='center',
                        color='black',
                        fontsize=9
                    )
                    i+=1

    if save_dir:
        plt.savefig(save_dir)
    plt.show()
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc(score, y, steps=101):
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
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_class_distribution_by_group(y, group_attr, annotate=True):
    mask = ~np.isnan(group_attr)
    group_clean = group_attr[mask]
    y_clean = y[mask]
    g = sns.histplot(x=group_clean, hue=y_clean, multiple="stack", stat="probability")

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


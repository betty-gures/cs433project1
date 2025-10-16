import numpy as np
import matplotlib.pyplot as plt

def plot_roc(score, y, steps=21):
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
print("Starting model comparisons...")
import os

import numpy as np

from models import OrdinaryLeastSquares, LogisticRegression, LinearSVM, KNearestNeighbors
from model_selection import cross_validation

train = np.load("data/dataset_prep/train.npz")
x_train, y_train = train["x_train"], train["y_train"]
    
    
model_settings = [
    {"model_class": OrdinaryLeastSquares},
    #{"model_class": LogisticRegression},
    #{"model_class": LinearSVM},
    #{"model_class": KNearestNeighbors, "use_pca": True},
]
for model in model_settings:
    print(f"Cross-validating model: {model['model_class'].__name__}")
    num_samples = int(1e6) if model['model_class'] != KNearestNeighbors else int(1e6)
    cv_results = cross_validation(x_train[:num_samples], y_train[:num_samples], verbose=True, max_test=20000, **model)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model['model_class'].__name__}.txt")

    with open(out_path, "w") as f:
        f.write(f"Model: {model['model_class'].__name__}\n\n")
        f.write("cv_results repr:\n")
        f.write(repr(cv_results))
        f.write("\n\n")
        f.write("Aggregated metrics:\n")
        f.write(f"Average F1-score: {np.mean(cv_results.f1_scores)*100:.1f}% ± {np.std(cv_results.f1_scores)*100:.1f}\n")
        f.write(f"Average F2-score: {np.mean(cv_results.f2_scores)*100:.1f}% ± {np.std(cv_results.f2_scores)*100:.1f}\n")
        f.write(f"Average AUC-ROC: {np.mean(cv_results.auc_rocs)*100:.1f}% ± {np.std(cv_results.auc_rocs)*100:.1f}\n")
    
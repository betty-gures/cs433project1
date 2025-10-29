"""Compare models via cross-validation.

This script provides a small CLI to run one or more models on the
preprocessed training split and write aggregated metrics to `results/`.

Usage examples:
- Run all models (default):
    python compare_models.py
- Run a subset of models:
    python compare_models.py --model linear_svm logistic_regression
- Limit the number of samples used:
    python compare_models.py --num-samples 100000
"""

import argparse
import os

import numpy as np

from models import OrdinaryLeastSquares, LogisticRegression, LinearSVM, KNearestNeighbors, DecisionTree
from model_selection import cross_validation
from preprocessing import preprocess


# Map short, CLI-friendly names to concrete model classes.
MODEL_REGISTRY = {
    "ols": OrdinaryLeastSquares,
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVM,
    "knn": KNearestNeighbors,
    "decision_tree": DecisionTree,
}


def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including: `models` (list[str]) and `num_samples` (int).
    """
    parser = argparse.ArgumentParser(description="Compare different models with cross-validation.")
    parser.add_argument(
        "--model",
        dest="models",
        choices=list(MODEL_REGISTRY.keys()),
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="One or more models to run. Defaults to all available models.",
    )
    parser.add_argument(
        "--no_one_hot_encoding",
        action="store_true",
        help="If set, disables one-hot encoding for categorical features.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(1e6),
        help="Maximum number of samples to use from the training split.",
    )
    return parser.parse_args()


def main():
    """Entry point for running comparisons and writing results."""
    args = parse_args()

    print("Starting model comparisons...")
    out_dir = "results"  # Output directory for result text files
    num_samples = args.num_samples

    # Build model configurations from CLI selection(s).
    model_settings = [{"model_class": MODEL_REGISTRY[name]} for name in args.models]

    # Loading data
    x_train, _, y_train, *_ = preprocess(one_hot_encoding=not args.no_one_hot_encoding)

    for model in model_settings:
        print(f"Cross-validating model: {model['model_class'].__name__}")

        # kNN can be very slow on large test sets; cap its `max_test` separately.
        cv_results = cross_validation(
            x_train[:num_samples],
            y_train[:num_samples],
            verbose=True,
            max_test=20000 if model["model_class"] == KNearestNeighbors else num_samples,
            **model,
        )

        # Persist a short textual report with basic aggregated metrics.
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


if __name__ == "__main__":
    main()
    

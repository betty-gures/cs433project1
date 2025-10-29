"""Train model, make predictions on test set and create submission file.

This script allows training a specified model on the training data, making predictions on the test data, and creating a submission file in the required format.

Usage examples:
- Train and evaluate using Ordinary Least Squares (default):
    python run.py
- Train and evaluate using Logistic Regression:
    python run.py --model logistic_regression
- Disable one-hot encoding for categorical features:
    python run.py --no_one_hot_encoding
"""

import argparse
import os

import numpy as np

from helpers import create_csv_submission
from models import OrdinaryLeastSquares, LogisticRegression, LinearSVM, KNearestNeighbors, DecisionTree
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
    Args:
        None
    
    Returns:
        argparse.Namespace, Parsed arguments including: `model` (str), `no_one_hot_encoding` (bool), `submission_file_name` (str), and `verbose` (bool).
    """
    parser = argparse.ArgumentParser(description="Train and evaluate prediction model.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="ols",
        help="The model to use for training and evaluation (ols, logistic_regression, linear_svm, knn)."
    )
    parser.add_argument(
        "--no_one_hot_encoding",
        action="store_true",
        help="If set, disables one-hot encoding for categorical features.",
    )
    parser.add_argument("--submission_file_name", type=str, default="ordinary_least_squares", help="Name of the submission file (without dir and file extension).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints detailed logs during training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Entry point for training and model."""
    args = parse_args()

    # Preprocess the data
    x_train, x_test, y_train, test_ids, _ = preprocess(one_hot_encoding=not args.no_one_hot_encoding)

    # Initialize and train the model
    model_class = MODEL_REGISTRY[args.model]
    print("Training model...")
    model = model_class()
    model.hyperparameter_tuning(x_train, y_train, verbose=args.verbose)
    model.train(X=x_train, y=y_train)

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Convert to -1 and 1
    predictions = np.where(predictions == 1, 1, -1)

    # Create submission file
    os.makedirs("data/submissions", exist_ok=True)
    create_csv_submission(test_ids, predictions, f"data/submissions/{args.submission_file_name}.csv")
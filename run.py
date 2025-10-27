import argparse
import os

import numpy as np

from helpers import create_csv_submission
from models import OrdinaryLeastSquares, LogisticRegression, LinearSVM, KNearestNeighbors
from preprocessing import preprocess

# Map short, CLI-friendly names to concrete model classes.
MODEL_REGISTRY = {
    "ols": OrdinaryLeastSquares,
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVM,
    "knn": KNearestNeighbors,
}

if __name__ == "__main__":
    # Command line arguments
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
    args = parser.parse_args()
    model_class = {
        "ordinary_least_squares": OrdinaryLeastSquares,
        "logistic_regression": LogisticRegression,
        "linear_svm": LinearSVM,
        "k_nearest_neighbors": KNearestNeighbors
    }[args.model]


    # Preprocess the data
    x_train, x_test, y_train, test_ids, _ = preprocess(one_hot_encoding=not args.no_one_hot_encoding)

    # Initialize and train the model
    model = model_class()
    print("Training model...")
    model = OrdinaryLeastSquares()
    model.hyperparameter_tuning(x_train, y_train, verbose=args.verbose)
    model.train(X=x_train, y=y_train)

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Convert to -1 and 1
    predictions = np.where(predictions == 1, 1, -1)

    # Create submission file
    os.makedirs("data/submissions", exist_ok=True)
    create_csv_submission(test_ids, predictions, f"data/submissions/{args.submission_file_name}.csv")
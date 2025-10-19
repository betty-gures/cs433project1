# Classes for LogReg, SVM, Decision Tree, etc
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np

from implementations import sigmoid
from model_selection import test_val_split, find_best_threshold

### LOSS FUNCTIONS

def binary_cross_entropy_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Numerically stable version, operating directly on the logits, does not require clipping of y_pred

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    logits = tx @ w
    loss = np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits)))
    return np.mean(loss).item()

def get_sample_weights(y, weighting=False):
    if weighting:
        return y.shape[0] / 2 * np.where(y==1, 1/np.sum(y), 1/np.sum(1-y))
    else:
        return np.ones(y.shape[0])


def validate_data(X, y):
    assert set(np.unique(y)).issubset({0, 1}), "y must be in {0, 1}"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"


### MODELS

class OrdinaryLeastSquares():
    """(Weighted) Ordinary Least Squares Regression for binary classification (0/1 labels) using closed-form solution.
    """
    def __init__(self, seed = 42, weighting = True, metric = None):
        self.weights = None
        self.decision_threshold = 0.5
        self.metric = metric
        self.weighting = weighting
        self.seed = seed
        self.rng = np.random.default_rng(seed) # local random generator

    def train(self, X, y, verbose=False):
        """Fit the model to data using ordinary least squares.

        Args:
            X (np.array): training data
            y (np.array): labels for training data in format (0,1)
        """
        validate_data(X, y)

        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.2)

        self.sample_weights = get_sample_weights(y_train, weighting=self.weighting)

        # closed-form solution
        w_sqrt = np.sqrt(self.sample_weights)[:, np.newaxis]  # shape (N,1)
        Xw = X_train * w_sqrt  # multiply each row by sqrt(weight)
        yw = y_train * np.sqrt(self.sample_weights)  # shape (N,)
        self.weights = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)

        if self.metric is not None:
            scores = self.predict(X_val, scores=True)
            self.decision_threshold = find_best_threshold(scores, y_val, self.metric, verbose=verbose)

    def predict(self, X, scores=False):
        """
        Predict the probablity of Y=1

        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions
        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        preds = X @ self.weights
        if scores: return sigmoid(preds) # to stay in the range from 0 to 1
        return (preds >= self.decision_threshold).astype(int)

@dataclass
class LogRegTrainResult:
    iterations: int
    train_losses: List[float]
    val_losses: List[float]
    decision_threshold: float
    weighting: bool
    gamma: float
    seed: int

class LogisticRegression():
    """Logistic Regression using Gradient Descent for binary classification (0/1 labels).
    """
    def __init__(self, max_iter = 1000, gamma = 5e-1, weighting = True, seed = 42, metric = None, patience=10, stopping_threshold=1e-6):
        # hyperparameters
        self.decision_threshold = 0.5
        self.gamma = gamma
        self.max_iter = max_iter
        self.metric = metric
        self.patience = patience
        self.rng = np.random.default_rng(seed) # local random generator
        self.sample_weights = None
        self.seed = seed
        self.stopping_threshold = stopping_threshold
        self.weighting = weighting
        
        # model parameters
        self.weights = None
        
    def gradient_step(self, y, tx):
        """Compute the gradient of loss and take a gradient step.

        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)

        Returns:
            a vector of shape (D, 1)
        """
        pred = sigmoid(tx @ self.weights) # σ(x^T w), numerically stable
        gradient = tx.T @ (self.sample_weights * (pred - y)) / np.sum(self.sample_weights) # weighted gradient
        assert gradient.shape == self.weights.shape
        self.weights -= self.gamma * gradient

    def train(self, X, y, verbose=False):
        """Train the model using gradient descent
    
        Args:
            X (np.array): training data
            y (np.array): labels for training data in format (0,1)
            val_ratio (float, optional): ratio of validation data. Defaults to 0.2
            verbose (bool, optional): If True, print out information along the training. Defaults to True.

        Returns:
            LogRegTrainResult: dataclass containing training information
        """

        validate_data(X, y)

        train_losses, val_losses = [], []
        if self.metric is not None:
            # hyperparameter tuning
            X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)

            self.weights = np.zeros(X_train.shape[1])
            self.sample_weights = get_sample_weights(y_train, weighting=self.weighting)

            
            best_val_loss = np.inf
            patience_counter = 0

            for iter in range(self.max_iter):
                # keep track of losses
                train_losses.append(binary_cross_entropy_loss(y_train, X_train, self.weights))
                val_loss = binary_cross_entropy_loss(y_val, X_val, self.weights)
                val_losses.append(val_loss)

                # early stopping check
                if val_loss < best_val_loss - self.stopping_threshold:
                    best_val_loss = val_loss
                    best_w = self.weights.copy() # remember best weights
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose: print(f"Early stopping at iteration {iter}")
                        self.weights = best_w
                        break

                # gradient step
                self.gradient_step(y_train, X_train)

                # log info
                if verbose:
                    print(f"Current iteration={iter}, loss={val_losses[-1]}")
                    print(f"L1 norm of w: {np.sum(np.abs(self.weights))}")
            if verbose:
                print(f"train loss={binary_cross_entropy_loss(y_train, X_train, self.weights)}")
                print(f"val loss={binary_cross_entropy_loss(y_val, X_val, self.weights)}")

            
            scores = self.predict(X_val, scores=True)
            self.decision_threshold = find_best_threshold(scores, y_val, self.metric, verbose=verbose)
            self.max_iter = iter  # update max_iter to actual number of iterations run

        self.weights = np.zeros(X.shape[1])
        self.sample_weights = get_sample_weights(y, weighting=self.weighting)
        for _ in range(self.max_iter):
            self.gradient_step(y, X)  # retrain on full data

        return LogRegTrainResult(self.max_iter, train_losses, val_losses, self.decision_threshold, self.weighting, self.gamma, self.seed)

    def predict(self, X, scores=False):
        """
        Predict the probablity of Y=1

        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions

        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        probs = sigmoid(X @ self.weights)
        if scores: return probs
        return (probs >= self.decision_threshold).astype(int)

### SVM
def hinge_loss(y, X, w, _lambda=1.0, include_reg=True):
    """
    Compute SVM hinge loss.
    """
    margins = y * (X @ w)
    hinge = np.maximum(0, 1 - margins)
    if include_reg:
        return _lambda / 2 * np.sum(w[1:] ** 2) + np.mean(hinge) # exclude bias term from regularization
    else:
        return np.mean(hinge)

class LinearSVM:
    def __init__(self, _lambda=1.0, lr=0.1, metric=None, max_iters=1000, patience=10, seed=42):
        # hyperparameters
        self._lambda = _lambda
        self.lr = lr
        self.metric = metric
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)
        self.patience = patience
        # model parameters
        self.w = None

    def gradient_step(self, X, y):
        margins = y * (X @ self.w) # compute margins
        mask = margins < 1 # find misclassified samples
        dw = self._lambda * self.w - np.mean((mask[:, None] * y[:, None]) * X, axis=0) # gradient calculation
        self.w -= self.lr * dw # gradient descent update

    def train(self, X, y, verbose=False):
        validate_data(X, y)
        y = np.where(y <= 0, -1, 1)  # ensure labels are -1 or 1

        if self.metric is not None:
            # hyperparameter tuning
            X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)

            lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
            lrs = [0.01, 0.1]
            metric_scores = {}
            num_iters = {}
            train_losses, val_losses = {}, {}

            for _lambda, lr in product(lambdas, lrs):
                train_losses[(_lambda, lr)] = []
                val_losses[(_lambda, lr)] = []
                self.w = np.zeros(X.shape[1]) # initialize weights
                
                self._lambda = _lambda
                self.lr = lr

                best_val_loss = np.inf
                patience_counter = 0

                for iter in range(self.max_iters):
                    train_loss = hinge_loss(y, X, self.w, _lambda=_lambda, include_reg=False)
                    if verbose: print(f"Iteration {iter}, Training Loss: {train_loss}")
                    train_losses[(_lambda, lr)].append(train_loss)
                    val_loss = hinge_loss(y_val, X_val, self.w, _lambda=_lambda, include_reg=False)
                    val_losses[(_lambda, lr)].append(val_loss)

                    # early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_w = self.w.copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if verbose: print(f"Early stopping at iteration {iter}")
                            self.w = best_w
                            break

                    self.gradient_step(X_train, y_train)

                metric_scores[(_lambda, lr)] = self.metric(self.predict(X_val), y_val.astype(int).clip(min=0))
                num_iters[(_lambda, lr)] = iter + 1

            # select best lambda based on metric
            self._lambda, self.lr = max(metric_scores, key=metric_scores.get)
            self.max_iters = num_iters[(self._lambda, self.lr)]
        
        # train model on full data
        self.w = np.zeros(X.shape[1])  # initialize weights
        for iter in range(self.max_iters):
            self.gradient_step(X, y)  # retrain on full data

        return train_losses[(self._lambda, self.lr)], val_losses[(self._lambda, self.lr)]

    def predict(self, X, scores=False):
        preds = X @ self.w
        if scores:
            return sigmoid(preds) # to stay in the range from 0 to 1
        return np.sign(preds).astype(int).clip(min=0)  # convert -1/1 to 0/1
    
# kNN
class KNearestNeighbors:
    def __init__(self, k=100, metric=None, seed=42):
        self.k = k
        self.metric = metric
        self.decision_threshold = 0.5
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def train(self, X, y, verbose=False):
        """
        Store the training data.
        X: training features, shape (num_samples, num_features)
        y: training labels, shape (num_samples,)
        """

        validate_data(X, y)

        if self.metric is not None:
            # hyperparameter tuning 

            X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.2)
            self.X_train = X_train
            self.y_train = y_train
            # find optimal K and threshold on validation set
            base_k = min(0.1 * X_train.shape[0], np.sqrt(X_train.shape[0]))
            k_values = list([int(base_k * factor) | 1 for factor in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]) # make sure k is odd with bitwise OR
            print(f"Trying k values: {k_values}")
            metric_scores = {}
            thresholds = {}
            for k in k_values:
                if verbose: print(f"Evaluating k={k}")
                self.k = k
                scores = self.predict(X_val, scores=True)
                if verbose: print("Finding best threshold...")
                thresholds[k] = find_best_threshold(scores, y_val, self.metric, verbose=verbose)
                predictions = (scores >= thresholds[k]).astype(int)
                metric_scores[k] = self.metric(predictions, y_val)
            self.k = max(metric_scores, key=metric_scores.get)
            self.decision_threshold = thresholds[self.k]
            if verbose: print(f"Best k found: {self.k} with metric score: {metric_scores[self.k]}")

        # "retrain" on full data
        self.X_train = X
        self.y_train = y


    def predict(self, X, scores=False):
        """
        Predict the label for each point in X.
        Args:
            X: test features, shape (num_test_samples, num_features)
            scores: test scores, shape (num_test_samples,)
        Returns:
            predicted labels or probabilities, shape (num_test_samples,)
        """
        probabilities = []
        for x in X:
            # Compute Euclidean distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Find the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get the labels of the k nearest neighbors
            k_labels = self.y_train[k_indices]
            # Compute ratio of positive labels
            probs = np.mean(k_labels)
            probabilities.append(probs)
        if scores: 
            return np.array(probabilities)
        return (np.array(probabilities) >= self.decision_threshold).astype(int)
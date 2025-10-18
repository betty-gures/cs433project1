# Classes for LogReg, SVM, Decision Tree, etc
from dataclasses import dataclass
from typing import List, Any
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
    def __init__(self, max_iter = 1000, stopping_threshold = 1e-6, gamma = 5e-1, weighting = True, seed = 42, metric = None):
        self.max_iter = max_iter
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.weighting = weighting
        self.sample_weights = None
        self.weights = None
        self.metric = metric
        self.decision_threshold = 0.5
        self.seed = seed
        self.rng = np.random.default_rng(seed) # local random generator
        
    def compute_gradient(self, y, tx, w):
        """compute the gradient of loss.

        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)

        Returns:
            a vector of shape (D, 1)
        """
        pred = sigmoid(tx @ w) # σ(x^T w), numerically stable
        # weighted gradient
        gradient = tx.T @ (self.sample_weights * (pred - y)) / np.sum(self.sample_weights)
        assert gradient.shape == w.shape
        return gradient

    def train(self, X, y, metric=None, verbose=False):
        """Train the model using gradient descent
    
        Args:
            X (np.array): training data
            y (np.array): labels for training data in format (0,1)
            val_ratio (float, optional): ratio of validation data. Defaults to 0.2
            verbose (bool, optional): If True, print out information along the training. Defaults to True.
            metric (callable, optional): metric function to evaluate performance. Defaults to None.

        Returns:
            LogRegTrainResult: dataclass containing training information
        """

        validate_data(X, y)

        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)

        self.sample_weights = get_sample_weights(y_train, weighting=self.weighting)

        train_losses, val_losses, iter = [], [], 0
        self.weights = np.zeros((X_train.shape[1]))

        while iter < self.max_iter:
            # compute losses
            train_losses.append(binary_cross_entropy_loss(y_train, X_train, self.weights))
            val_losses.append(binary_cross_entropy_loss(y_val, X_val, self.weights))

            # early stopping criterion
            if len(val_losses) > 1 and np.abs(val_losses[-1] - val_losses[-2]) < self.stopping_threshold:
                break

            # gradient step
            self.weights -= self.gamma * self.compute_gradient(y_train, X_train, self.weights)
            iter += 1

            # log info
            if verbose and iter % 1 == 0:
                print(f"Current iteration={iter}, loss={val_losses[-1]}")
                print(f"L1 norm of w: {np.sum(np.abs(self.weights))}")
        if verbose:
            print(f"train loss={binary_cross_entropy_loss(y_train, X_train, self.weights)}")
            print(f"val loss={binary_cross_entropy_loss(y_val, X_val, self.weights)}")

        if self.metric is not None:
            scores = self.predict(X_val, scores=True)
            self.decision_threshold = find_best_threshold(scores, y_val, self.metric, verbose=verbose)
            

        return LogRegTrainResult(iter, train_losses, val_losses, self.decision_threshold, self.weighting, self.gamma, self.seed)

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
def hinge_loss(y, X, w, C=1.0, include_reg=True):
    """
    Compute SVM hinge loss.
    """
    margins = y * (X @ w)
    hinge = np.maximum(0, 1 - margins)
    if include_reg:
        return 0.5 * np.sum(w[1:] ** 2) + C / y.shape[0] * np.sum(hinge)
    else:
        return C / y.shape[0] * np.sum(hinge)
    
class LinearSVM:
    def __init__(self, C=1.0, lr=0.1, max_iters=1000, seed=42, patience=10):
        self.C = C
        self.lr = lr
        self.max_iters = max_iters
        self.w = None
        self.rng = np.random.default_rng(seed)
        self.patience = patience

    def train(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)  # ensure labels are -1 or 1

        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.2)

        # initialize weights
        self.w = np.zeros(n_features)
        train_losses, val_losses = [], []
        
        best_val_loss = np.inf
        patience_counter = 0

        for iter in range(self.max_iters):
            train_loss = hinge_loss(y_train, X_train, self.w, C=self.C, include_reg=False)
            if verbose: print(f"Iteration {iter}, Training Loss: {train_loss}")
            train_losses.append(train_loss)
            val_loss = hinge_loss(y_val, X_val, self.w, C=self.C, include_reg=False)
            val_losses.append(val_loss)

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_w = self.w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at iteration {iter}")
                    self.w = best_w
                    break

            # compute margins
            margins = y_train * (X_train @ self.w)
            # find misclassified samples
            mask = margins < 1

            # gradient calculation
            dw = self.w - self.C / y_train.shape[0] * np.sum((mask[:, None] * y_train[:, None]) * X_train, axis=0)

            # gradient descent update
            self.w -= self.lr / self.C * dw

        return train_losses, val_losses

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

        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.2)

        self.X_train = X_train
        self.y_train = y_train

        if self.metric is not None:
            scores = self.predict(X_val, scores=True)
            self.decision_threshold = find_best_threshold(scores, y_val, self.metric, verbose=verbose)



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
# Classes for LogReg, SVM, Decision Tree, etc
from dataclasses import dataclass
from typing import List, Any
import numpy as np

from implementations import sigmoid

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
    def __init__(self, max_iter = 1000, stopping_threshold = 1e-6, gamma = 5e-1, weighting = False, seed = 42):
        self.max_iter = max_iter
        self.stopping_threshold = stopping_threshold
        self.gamma = gamma
        self.weighting = weighting
        self.sample_weights = None
        self.weights = None
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

    def train(self, X_train, y_train, val_ratio=0.2, verbose=False, metric=None, NUM_THRESHOLDS=100):
        """Train the model using gradient descent
    
        Args:
            X_train (np.array): training data
            y_train (np.array): labels for training data in format (0,1)
            val_ratio (float, optional): ratio of validation data. Defaults to 0.2
            verbose (bool, optional): If True, print out information along the training. Defaults to True.
            metric (callable, optional): metric function to evaluate performance. Defaults to None.
        """

        assert set(np.unique(y_train)).issubset({0, 1}), "y_train must be in {0, 1}"
        
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train must have the same number of samples"

        # split train/val for early stopping and threshold tuning
        num_val = int(X_train.shape[0] * val_ratio)
        indices = np.random.permutation(X_train.shape[0])
        val_idx = indices[:num_val]
        train_idx = indices[num_val:]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        X_train, y_train = X_train[train_idx], y_train[train_idx]

        if self.weighting:
            pos_weight = y_train.shape[0] / (2 * np.sum(y_train))
            neg_weight = y_train.shape[0] / (2 * np.sum(1 - y_train))
            self.sample_weights = np.where(y_train == 1, pos_weight, neg_weight)
        else:
            self.sample_weights = np.ones(y_train.shape[0])          

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

        # find optimal threshold on validation set
        thresholds = np.linspace(0, 1, NUM_THRESHOLDS)
        scores = []
        y_score = self.predict(X_val, probability=True)
        if metric is not None:
            for t in thresholds:
                y_val_pred = (y_score >= t).astype(int)
                scores.append(metric(y_val_pred, y_val))
            self.decision_threshold = thresholds[np.argmax(scores)]
            if verbose: print(f"Best threshold: {self.decision_threshold} with score {np.max(scores)}")

        return LogRegTrainResult(iter, train_losses, val_losses, self.decision_threshold, self.weighting, self.gamma, self.seed)

    def predict(self, X, probability=False):
        """
        Predict the probablity of Y=1

        """
        probs = sigmoid(X @ self.weights)
        if probability:
            return probs
        return (probs >= self.decision_threshold).astype(int)


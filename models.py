# Classes for LogReg, SVM, Decision Tree, etc
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np

from implementations import sigmoid
from metrics import f_score
from preprocessing import normalize_and_bias_data, pca

### HYPERPARAMETER TUNING HELPER FUNCTIONS

def find_best_threshold(scores, true, metric, num_thresholds=201, verbose=False):
    """Find the best decision threshold for binary classification based on a given metric.
    Args:
        scores: np.ndarray of shape (N,), predicted probabilities or scores
        true: np.ndarray of shape (N,), true binary labels (0 or 1)
        metric: function(pred, true) -> float, metric to optimize
        num_thresholds: int, number of thresholds to evaluate
        """
    thresholds = np.linspace(0, 1, num_thresholds)
    metric_vals = [] # collects the metric values for each threshold
    
    for t in thresholds:
        pred = (scores >= t).astype(int)
        metric_vals.append(metric(pred, true))
    best_threshold = thresholds[np.argmax(metric_vals)]
    if verbose: print(f"Best threshold: {best_threshold} with score {np.max(metric_vals)}")
    return best_threshold

def test_val_split(rng, X, y, val_ratio=0.2):
    """Split the data into training and validation sets.
     Args:
        rng: np.random.Generator
        X: np.ndarray of shape (N, D)
        y: np.ndarray of shape (N, )
        val_ratio: float, ratio of validation data
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    num_val = int(X.shape[0] * val_ratio)
    indices = rng.permutation(X.shape[0])
    val_idx = indices[:num_val]
    train_idx = indices[num_val:]
    X_val, y_val = X[val_idx], y[val_idx]
    X_train, y_train = X[train_idx], y[train_idx]
    return X_train, y_train, X_val, y_val

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
    assert not np.any(np.isnan(tx)), "Input features contain NaN values"
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
    def __init__(self, seed = 42, weighting = True):
        self.weights = None
        self.decision_threshold = 0.5
        self.weighting = weighting
        self.seed = seed
        self.rng = np.random.default_rng(seed) # local random generator

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        """Tune the decision threshold with a validation set"""
        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)
        
        self.train(X_train, y_train)

        scores = self.predict(X_val, scores=True)
        self.decision_threshold = find_best_threshold(scores, y_val, metric, verbose=verbose)

    def train(self, X, y):
        """Fit the model to data using ordinary least squares.

        Args:
            X (np.array): training data
            y (np.array): labels for training data in format (0,1)
        """
        validate_data(X, y)
        self.X = X
        X = normalize_and_bias_data(X)
        
        self.sample_weights = get_sample_weights(y, weighting=self.weighting)
        w_sqrt = np.sqrt(self.sample_weights)[:, np.newaxis]  # shape (N,1)
        Xw = X * w_sqrt  # multiply each row by sqrt(weight)
        yw = y * np.sqrt(self.sample_weights)  # shape (N,)
        #self.weights = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
        self.weights, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    def predict(self, X, scores=False, save_scores=False, precomputed_scores=None):
        """
        Predict the probablity of Y=1

        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions
        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        if precomputed_scores is not None:
            preds = precomputed_scores
        else:
            _, X = normalize_and_bias_data(self.X, X)
            preds = X @ self.weights
        if save_scores: self.scores = preds
        if scores: return sigmoid(preds) # non-parametric Platt scaling
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
    def __init__(self, max_iter = 1000, gamma = 1e-3, _lambda = 0.0, weighting = True, seed = 42, patience=10, stopping_threshold=1e-4, rho=0.95, eps=1e-8, use_pca=False, variance=0.999):
        # hyperparameters
        self.decision_threshold = 0.5
        self.gamma = gamma
        self._lambda = _lambda
        self.max_iter = max_iter
        self.patience = patience
        self.rng = np.random.default_rng(seed) # local random generator
        self.sample_weights = None
        self.seed = seed
        self.stopping_threshold = stopping_threshold
        self.weighting = weighting
        # --- Optimizer Parameters ---
        self.rho = rho
        self.eps = eps
        self.use_pca = use_pca
        self.variance = 0.999

        # model parameters
        self.weights = None
        self.X = None
        
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
        gradient = tx.T @ (self.sample_weights * (pred - y)) / np.sum(self.sample_weights) + 2 * self._lambda * np.r_[0, self.weights[1:]] # weighted gradient, not penalizing bias
        self.gradient = gradient
        assert gradient.shape == self.weights.shape

        # --- Adam optimizer ---
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m = beta1 * self.m + (1 - beta1) * gradient
        self.v = beta2 * self.v + (1 - beta2) * (gradient ** 2)
        m_hat = self.m / (1 - beta1 ** self.t)
        v_hat = self.v / (1 - beta2 ** self.t)

        self.weights -= self.gamma * m_hat / (np.sqrt(v_hat) + eps)

        #self.weights -= self.gamma * gradient

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        """
        """
        train_losses, val_losses = [], []

        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)
        self.X = X_train
        X_train, X_val_biased = normalize_and_bias_data(X_train, X_val)
        if self.use_pca:
            X_train, self.apply_pca = pca(X_train, self.variance)
            if verbose: print(f"Reduced to {X_train.shape[1]} features")
            X_val_biased = self.apply_pca(X_val_biased)
        
        self.sample_weights = get_sample_weights(y_train, weighting=self.weighting)

        lambdas = [0] #, 1e-4, 1e-3, 1e-2, 1e-1] #4

        metric_scores, num_iters, thresholds = {}, {}, {}
        train_losses, val_losses = {}, {}

        for _lambda in lambdas:
            self._lambda = _lambda
            if verbose: print(f"Evaluating lambda={self._lambda}")
            self.weights = np.zeros(X_train.shape[1])
            self.m = np.zeros_like(self.weights)
            self.v = np.zeros_like(self.weights)
            self.t = 0

            best_val_loss = np.inf
            patience_counter = 0
            train_losses[_lambda] = []
            val_losses[_lambda] = []
            best_w = self.weights.copy()
            for iter in range(self.max_iter):
                # keep track of losses
                train_losses[_lambda].append(binary_cross_entropy_loss(y_train, X_train, self.weights))
                val_loss = binary_cross_entropy_loss(y_val, X_val_biased, self.weights)
                val_losses[_lambda].append(val_loss)
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

                if verbose and iter % 10 == 0:  # every 10 iterations
                    print(f"Iter {iter:4d}: "
                          f"loss={val_loss:.4f}")
               


            if verbose:
                print(f"train loss={binary_cross_entropy_loss(y_train, X_train, self.weights)}")
                print(f"val loss={binary_cross_entropy_loss(y_val, X_val_biased, self.weights)}")
            
            scores = self.predict(X_val, scores=True)
            thresholds[_lambda] = find_best_threshold(scores, y_val, metric, verbose=verbose)
            
            metric_scores[_lambda] = metric(self.predict(X_val), y_val)
            num_iters[_lambda] = iter + 1

        self._lambda = max(metric_scores, key=metric_scores.get)
        self.max_iter = num_iters[self._lambda]
        self.decision_threshold = thresholds[self._lambda]
        return train_losses[self._lambda], val_losses[self._lambda]

    def train(self, X, y):
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

        self.X = X # remember X for normalization during prediction
        X = normalize_and_bias_data(X)
        if self.use_pca:
            X, self.apply_pca = pca(X, self.variance)

        self.weights = np.zeros(X.shape[1])
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0
        self.sample_weights = get_sample_weights(y, weighting=self.weighting)
        
        for _ in range(self.max_iter):
            self.gradient_step(y, X)

    def predict(self, X, scores=False, save_scores=False, precomputed_scores=None):
        """
        Predict the probablity of Y=1

        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions

        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        if precomputed_scores is not None:
            probs = precomputed_scores
        else:
            _, X = normalize_and_bias_data(self.X, X)
            if self.use_pca:
                X = self.apply_pca(X)
            probs = sigmoid(X @ self.weights)
        if save_scores: self.scores = probs
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

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        
        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)
        self.X = X_train
        X_train, X_val_biased = normalize_and_bias_data(X_train, X_val)

        lambdas = [0.25, 0.5, 1.0, 2.0, 4.0]
        lrs = [0.01, 0.1]
        metric_scores = {}
        num_iters = {}
        train_losses, val_losses = {}, {}

        for _lambda, lr in product(lambdas, lrs):
            train_losses[(_lambda, lr)] = []
            val_losses[(_lambda, lr)] = []
            
            
            self._lambda = _lambda
            self.lr = lr

            best_val_loss = np.inf
            patience_counter = 0
            self.w = np.zeros(X_train.shape[1]) # initialize weights
            for iter in range(self.max_iters):
                train_loss = hinge_loss(y_train, X_train, self.w, _lambda=_lambda, include_reg=False)
                #if verbose: print(f"Iteration {iter}, Training Loss: {train_loss}")
                train_losses[(_lambda, lr)].append(train_loss)
                val_loss = hinge_loss(y_val, X_val_biased, self.w, _lambda=_lambda, include_reg=False)
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

            metric_scores[(_lambda, lr)] = metric(self.predict(X_val), y_val.astype(int).clip(min=0))
            num_iters[(_lambda, lr)] = iter + 1

        # select best lambda based on metric
        self._lambda, self.lr = max(metric_scores, key=metric_scores.get)
        self.max_iters = num_iters[(self._lambda, self.lr)]
        return train_losses, val_losses

    def train(self, X, y):
        validate_data(X, y)

        self.X = X # remember X for normalization during prediction
        X = normalize_and_bias_data(X)
        y = np.where(y <= 0, -1, 1)  # ensure labels are -1 or 1
        
        self.w = np.zeros(X.shape[1])  # initialize weights
        for _ in range(self.max_iters):
            self.gradient_step(X, y)  # retrain on full data

    def predict(self, X, scores=False, save_scores=False, precomputed_scores=None):
        if precomputed_scores is not None:
            preds = precomputed_scores
        else:
            _, X = normalize_and_bias_data(self.X, X)
            preds = X @ self.w
        if save_scores: self.scores = preds
        if scores: return sigmoid(preds) # to stay in the range from 0 to 1
        return np.sign(preds).astype(int).clip(min=0)  # convert -1/1 to 0/1
    
# kNN
class KNearestNeighbors:
    def __init__(self, k=100, metric=None, seed=42, use_pca=True):
        self.k = k
        self.metric = metric
        self.decision_threshold = 0.5
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_pca = use_pca
        self.variance = 1.0

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.05)
        print(X_val.shape)
        self.X = X_train
        self.y_train = y_train
        X_train = normalize_and_bias_data(X_train)
        
        # find optimal K and threshold on validation set
        base_k = min(0.1 * X_train.shape[0], np.sqrt(X_train.shape[0]))
        k_values = list([int(base_k * factor) | 1 for factor in [1/64, 1/32, 1/16, 1/8, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]]) # make sure k is odd with bitwise OR
        variances = [ 1/20, 1/10, 1/6, 1/4, 1/2, 0.75]#, 0.9, 1.0 ] if self.use_pca else [1.0]

        metric_scores = {}
        thresholds = {}

        for variance in variances:
            if verbose: print(f"Evaluating variance={variance}")
            if self.use_pca and variance < 1.0:
                X, apply_pca = pca(X_train, variance)
                self.apply_pca = apply_pca
            else:
                X = X_train
            if verbose: print(f"Reduced to {X.shape[1]} features")
            self.X_train = X
            self.variance = variance
            all_scores = self.predict(X_val, return_max_k=max(k_values))

            for k in k_values:
                if verbose: print(f"Evaluating k={k}")
                scores = all_scores[:, k-1]
                thresholds[(k, variance)] = find_best_threshold(scores, y_val, metric, verbose=verbose)
                predictions = (scores >= thresholds[(k, variance)]).astype(int)
                metric_scores[(k, variance)] = metric(predictions, y_val)

        self.k, self.variance = max(metric_scores, key=metric_scores.get)
        self.decision_threshold = thresholds[(self.k, self.variance)]
        if verbose: print(f"Hyperparams found: {self.k}, {self.variance} with metric score: {metric_scores[(self.k, self.variance)]} at threshold {self.decision_threshold}")

    def train(self, X, y):
        """
        Store the training data.
        X: training features, shape (num_samples, num_features)
        y: training labels, shape (num_samples,)
        """

        validate_data(X, y)
        self.X = X
        X_train = normalize_and_bias_data(X)
        if self.use_pca and self.variance < 1.0:
            X_train, apply_pca = pca(X_train, self.variance)
            self.apply_pca = apply_pca
        self.X_train = X_train
        self.y_train = y


    def predict(self, X, scores=False, return_max_k=None, save_scores=False, precomputed_scores=None):
        """
        Predict the label for each point in X.
        Args:
            X: test features, shape (num_test_samples, num_features)
            scores: test scores, shape (num_test_samples,)
        Returns:
            predicted labels or probabilities, shape (num_test_samples,)
        """
        if precomputed_scores is not None:
            probs = precomputed_scores
        else:
            _, X = normalize_and_bias_data(self.X, X)
            if self.use_pca and self.variance < 1.0:
                X = self.apply_pca(X)

            max_k = self.k  if return_max_k is None else return_max_k

            predictions = np.zeros((X.shape[0], max_k))
            n = X.shape[0]
            for i, x in enumerate(X):
                if i % 1000 == 0: print(f"Predicting sample {i}/{n}")
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1)) # Compute Euclidean distances to all training points
                k_indices = np.argsort(distances)
                sorted_labels = self.y_train[k_indices]
                predictions[i, :] = np.cumsum(sorted_labels[:max_k]) / np.arange(1, max_k + 1)
            if return_max_k is not None:
                return predictions
        
            probs = predictions[:, self.k-1]
        if save_scores: self.scores = probs
        if scores: return probs
        return (probs >= self.decision_threshold).astype(int)
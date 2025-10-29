# Classes for LogReg, SVM, Decision Tree, etc
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np

from implementations import sigmoid
from metrics import f_score
from preprocessing import preprocess_splits, pca

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
    return best_threshold, np.max(metric_vals)

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
    def __init__(self, _lambda=0, seed = 42, weighting = True, squared_features = True, tune_threshold=True):
        # Model parameters
        self.weights = None
        # Hyperparameters
        self.decision_threshold = sigmoid(0.5)
        self._lambda = _lambda
        self.rng = np.random.default_rng(seed) # local random generator
        self.seed = seed
        self.squared_features = squared_features
        self.tune_threshold = tune_threshold
        self.weighting = weighting

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        """Tune the decision threshold with a validation set
        Args:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N, )
        
        Returns:
            None
        """
        if self.tune_threshold:
            X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)
            
            self.train(X_train, y_train)

            scores = self.predict(X_val, scores=True)
            self.decision_threshold, _ = find_best_threshold(scores, y_val, metric, verbose=verbose)

        return self.decision_threshold

    def train(self, X, y):
        """Fit the model to data using ordinary least squares.

        Args:
            X (np.array): training data
            y (np.array): labels for training data in format (0,1)

        Returns:
            None
        """
        validate_data(X, y)
        self.X = X
        X, _ = preprocess_splits(X, squared_features=self.squared_features)

        # Sample weights, are multiplied from the left and right
        self.sample_weights = get_sample_weights(y, weighting=self.weighting)
        w_sqrt = np.sqrt(self.sample_weights)[:, np.newaxis]
        Xw = X * w_sqrt
        yw = y * np.sqrt(self.sample_weights)

        # Regularization matrix (no penalty on bias term)
        D = np.eye(Xw.shape[1])
        D[0, 0] = 0

        # Ridge-augmented least squares: We create "new datapoints" that just reflected 
        X_aug = np.vstack([Xw, self._lambda * D])
        y_aug = np.concatenate([yw, np.zeros(Xw.shape[1])])

        # Weight update
        self.weights, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)

    def predict(self, X, scores=False, save_scores=False, use_scores=False):
        """
        Predict the probablity of Y=1

        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions
        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        if use_scores:
            preds = self.scores
        else:
            _, X, _ = preprocess_splits(self.X, X, squared_features=self.squared_features)
            preds = X @ self.weights
        if save_scores: self.scores = preds
        if scores: return sigmoid(preds) # non-parametric Platt scaling
        return (sigmoid(preds) >= self.decision_threshold).astype(int)


class LogisticRegression():
    """Logistic Regression using Gradient Descent for binary classification (0/1 labels).
    """
    def __init__(self, max_iter = 1000, gamma = 1e-1, _lambda = 0.0, weighting = True, seed = 42, patience=25, squared_features=False, stopping_threshold=1e-4):
        # hyperparameters
        self.decision_threshold = 0.5
        self.gamma = gamma
        self._lambda = _lambda
        self.max_iter = max_iter
        self.patience = patience
        self.rng = np.random.default_rng(seed) # local random generator
        self.sample_weights = None
        self.seed = seed
        self.squared_features = squared_features
        self.stopping_threshold = stopping_threshold
        self.weighting = weighting       

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

        self.weights -= self.gamma * gradient

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        """
        Tune hyperparameters using a validation set.
        Args:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N, ), with values in {0, 1}
            metric: function(pred, true) -> float, metric to optimize
            verbose: bool, if True print progress messages
        Returns:
            train_losses: list of training losses
            val_losses: list of validation losses
        """
        validate_data(X, y)

        train_losses, val_losses = [], []

        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)
        self.X = X_train
        X_train, X_val_biased, _ = preprocess_splits(X_train, X_val, squared_features=self.squared_features)
       
        
        self.sample_weights = get_sample_weights(y_train, weighting=self.weighting)

        lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1] 

        metric_scores, num_iters, thresholds = {}, {}, {}
        train_losses, val_losses = {}, {}

        for _lambda in lambdas:
            self._lambda = _lambda
            if verbose: print(f"Evaluating lambda={self._lambda}")
            self.weights = np.zeros(X_train.shape[1])

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
            thresholds[_lambda], _ = find_best_threshold(scores, y_val, metric, verbose=verbose)
            
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
            None
        """

        validate_data(X, y)

        self.X = X # remember X for normalization during prediction
        X, _ = preprocess_splits(X, squared_features=self.squared_features)

        self.weights = np.zeros(X.shape[1])
        self.sample_weights = get_sample_weights(y, weighting=self.weighting)
        
        for _ in range(self.max_iter):
            self.gradient_step(y, X)

    def predict(self, X, scores=False, save_scores=False, use_scores=False):
        """
        Predict the probablity of Y=1

        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions

        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        if use_scores:
            probs = self.scores
        else:
            _, X, _ = preprocess_splits(self.X, X, squared_features=self.squared_features)
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
    """Linear Support Vector Machine for binary classification (0/1 labels) using Gradient Descent on the hinge loss.
    """
    def __init__(self, _lambda=1.0, lr=0.1, metric=None, max_iters=1000, patience=10, seed=42, squared_features=False):
        # hyperparameters
        self._lambda = _lambda
        self.lr = lr
        self.metric = metric
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)
        self.patience = patience
        self.squared_features = squared_features
        # model parameters
        self.w = None

    def gradient_step(self, X, y):
        """Perform a single gradient step on the hinge loss.
        Args:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N, ), with values in {-1, 1}

        Returns:
            None
        """

        margins = y * (X @ self.w) # compute margins
        mask = margins < 1 # find misclassified samples
        dw = self._lambda * self.w - np.mean((mask[:, None] * y[:, None]) * X, axis=0) # gradient calculation
        self.w -= self.lr * dw # gradient descent update

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        """Tune hyperparameters using a validation set.
        
        Args:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N, ), with values in {0, 1}
            metric: function(pred, true) -> float, metric to optimize
            verbose: bool, if True print progress messages

        Returns:
            train_losses: dict mapping (_lambda, lr) to list of training losses"""
        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y)
        self.X = X_train
        X_train, X_val_biased, _ = preprocess_splits(X_train, X_val, squared_features=self.squared_features)

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
                train_losses[(_lambda, lr)].append(train_loss)
                val_loss = hinge_loss(y_val, X_val_biased, self.w, _lambda=_lambda, include_reg=False)
                val_losses[(_lambda, lr)].append(val_loss)
                if verbose and iter % 100 == 0: print(f"Iteration {iter}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

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
        """Train the SVM using gradient descent.
        Args:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N, ), with values in {0, 1}
        
        Returns:
            None
        """
        validate_data(X, y)

        self.X = X # remember X for normalization during prediction
        X, _ = preprocess_splits(X, squared_features=self.squared_features)
        y = np.where(y <= 0, -1, 1)  # ensure labels are -1 or 1
        
        self.w = np.zeros(X.shape[1])  # initialize weights
        for _ in range(self.max_iters):
            self.gradient_step(X, y)  # retrain on full data

    def predict(self, X, scores=False, save_scores=False, use_scores=False):
        """
        Predict the labels for given data points.
        Args:
            X: np.ndarray of shape (N,D)
            scores: bool, if True return raw scores instead of binary predictions
            save_scores: bool, if True save the raw scores
            use_scores: bool, if True use saved scores instead of recomputing
        Returns:
            np.ndarray of shape (N, ) with predicted labels (0 or 1) or scores
        """
        if use_scores:
            preds = self.scores
        else:
            _, X, _ = preprocess_splits(self.X, X, squared_features=self.squared_features)
            preds = X @ self.w
        if save_scores: self.scores = preds
        if scores: return sigmoid(preds) # non-parametric Platt scaling
        return np.sign(preds).astype(int).clip(min=0)  # convert -1/1 to 0/1
    
# kNN
class KNearestNeighbors:
    """k-Nearest Neighbors classifier for binary classification (0/1 labels).
    """
    def __init__(self, k=100, metric=None, seed=42, squared_features=False, use_pca=True):
        self.decision_threshold = 0.5
        self.k = k
        self.metric = metric
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.squared_features = squared_features
        self.use_pca = use_pca
        self.variance = 1.0

    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        """Tune hyperparameters k and explained variance using a validation set.
        Args:
            X: np.ndarray of shape (N, D)
            y: np.ndarray of shape (N, ), with values in {0, 1}
            metric: function, the metric to optimize
            verbose: bool, if True prints detailed logs
        Returns:
            None
        """
        X_train, y_train, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.05)
        print(X_val.shape)
        self.X = X_train
        self.y_train = y_train
        X_train, _ = preprocess_splits(X_train, squared_features=self.squared_features)
        
        # find optimal K and threshold on validation set
        base_k = min(0.1 * X_train.shape[0], np.sqrt(X_train.shape[0]))
        k_values = list([int(base_k * factor) | 1 for factor in [1/64, 1/32, 1/16, 1/8, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]]) # make sure k is odd with bitwise OR
        variances = [ 1/20, 1/10, 1/6, 1/4, 1/2, 0.75, 0.9, 1.0 ] if self.use_pca else [1.0]

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
                thresholds[(k, variance)], _ = find_best_threshold(scores, y_val, metric, verbose=verbose)
                predictions = (scores >= thresholds[(k, variance)]).astype(int)
                metric_scores[(k, variance)] = metric(predictions, y_val)

        self.k, self.variance = max(metric_scores, key=metric_scores.get)
        self.decision_threshold = thresholds[(self.k, self.variance)]
        if verbose: print(f"Hyperparams found: {self.k}, {self.variance} with metric score: {metric_scores[(self.k, self.variance)]} at threshold {self.decision_threshold}")

    def train(self, X, y):
        """
        Lazy function. Only store the preprocessed training data.
        X: training features, shape (num_samples, num_features)
        y: training labels, shape (num_samples,)
        """

        validate_data(X, y)
        self.X = X
        X_train, _ = preprocess_splits(X, squared_features=self.squared_features)
        if self.use_pca and self.variance < 1.0:
            X_train, apply_pca = pca(X_train, self.variance)
            self.apply_pca = apply_pca
        self.X_train = X_train
        self.y_train = y


    def predict(self, X, scores=False, return_max_k=None, save_scores=False, use_scores=False):
        """
        Predict the label for each point in X.
        Args:
            X: test features, shape (num_test_samples, num_features)
            scores: test scores, shape (num_test_samples,)
        Returns:
            predicted labels or probabilities, shape (num_test_samples,)
        """
        if use_scores:
            probs = self.scores
        else:
            _, X, _ = preprocess_splits(self.X, X, squared_features=self.squared_features)
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
    

class DecisionTree:
    """
    Lightweight implementation of a binary decision tree (CART) using Gini impurity.
    Fully NumPy-based and compatible with the existing project pipeline.
    """
    def __init__(self, max_depth=6, min_samples_leaf=50, max_features=None, seed=42, squared_features=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # if None -> use all features
        self.rng = np.random.default_rng(seed)
        self.squared_features = squared_features
        self.tree_ = None  # list of nodes

    # ----- utilities -----
    @staticmethod
    def _gini(y):
        if y.size == 0: return 0.0
        p1 = np.mean(y == 1)
        return 2 * p1 * (1 - p1)  # 1 - (p1^2 + p0^2)

    @staticmethod
    def _leaf_value(y):
        # return class-1 probability at leaf
        return np.mean(y == 1).item() if y.size else 0.0

    def _best_split(self, X, y, feat_idxes):
        # Returns (best_feature, best_threshold, best_gini, left_mask) or (None, None, None, None)
        n, d = X.shape
        base_gini = self._gini(y)
        best_gain = 0.0
        best = (None, None, None, None)

        for j in feat_idxes:
            col = X[:, j]
            # candidate thresholds: midpoints between sorted unique values (downsample if many)
            uniq = np.unique(col)
            if uniq.size < 2: 
                continue
            # downsample candidate thresholds for speed
            if uniq.size > 64:
                idx = np.linspace(0, uniq.size - 1, 64, dtype=int)
                uniq = uniq[idx]
            thr_candidates = (uniq[:-1] + uniq[1:]) * 0.5

            for thr in thr_candidates:
                left_mask = col <= thr
                nL = np.sum(left_mask)
                nR = n - nL
                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue
                giniL = self._gini(y[left_mask])
                giniR = self._gini(y[~left_mask])
                gini_split = (nL * giniL + nR * giniR) / n
                gain = base_gini - gini_split
                if gain > best_gain:
                    best_gain = gain
                    best = (j, thr, gini_split, left_mask)
        return best

    def _build(self, X, y, depth):
        node = {}
        # stopping criteria
        if (depth >= self.max_depth) or (y.size <= 2 * self.min_samples_leaf) or (np.all(y == y[0])):
            node["type"] = "leaf"
            node["p1"] = self._leaf_value(y)
            return node

        d = X.shape[1]
        feat_idxes = np.arange(d)
        if self.max_features is not None and self.max_features < d:
            feat_idxes = self.rng.choice(d, size=self.max_features, replace=False)

        j, thr, _, left_mask = self._best_split(X, y, feat_idxes)
        if j is None:
            node["type"] = "leaf"
            node["p1"] = self._leaf_value(y)
            return node

        node["type"] = "split"
        node["j"] = int(j)
        node["thr"] = float(thr)
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # ----- public API -----
    def hyperparameter_tuning(self, X, y, metric=f_score, verbose=False):
        # simple grid on depth / min_leaf; pick threshold later as usual
        X_tr, y_tr, X_val, y_val = test_val_split(self.rng, X, y, val_ratio=0.2)
        self.X = X_tr
        X_tr_p, X_val_p, _ = preprocess_splits(X_tr, X_val, squared_features=self.squared_features)

        grids = []
        for md in [3, 4, 5, 6, 8, 10]:
            for ml in [20, 50, 100, 200]:
                grids.append((md, ml))

        best_score = -1
        best_params = (self.max_depth, self.min_samples_leaf)
        best_thr = 0.5

        for md, ml in grids:
            self.max_depth, self.min_samples_leaf = md, ml
            # train
            self.tree_ = self._build(X_tr_p, y_tr, depth=0)
            # val probs
            probs = self.predict(X_val, scores=True)
            thr, _ = find_best_threshold(probs, y_val, metric, verbose=False)
            preds = (probs >= thr).astype(int)
            score = metric(preds, y_val)
            if verbose:
                print(f"depth={md}, min_leaf={ml} -> score={score:.4f} thr={thr:.2f}")
            if score > best_score:
                best_score, best_params, best_thr = score, (md, ml), thr

        self.max_depth, self.min_samples_leaf = best_params
        self.decision_threshold = best_thr
        if verbose:
            print(f"Best: depth={self.max_depth}, min_leaf={self.min_samples_leaf}, thr={self.decision_threshold:.2f}, score={best_score:.4f}")
        return [], []  # keep API symmetry with others

    def train(self, X, y):
        validate_data(X, y)
        self.X = X
        Xp, _ = preprocess_splits(X, squared_features=self.squared_features)
        self.tree_ = self._build(Xp, y, depth=0)

    def _predict_row(self, x, node):
        while node["type"] != "leaf":
            if x[node["j"]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["p1"]  # probability of class 1

    def predict(self, X, scores=False, save_scores=False, use_scores=False):
        _, Xp, _ = preprocess_splits(self.X, X, squared_features=self.squared_features)
        probs = np.array([self._predict_row(row, self.tree_) for row in Xp])
        if save_scores: self.scores = probs
        if scores: return probs
        thr = getattr(self, "decision_threshold", 0.5)
        return (probs >= thr).astype(int)

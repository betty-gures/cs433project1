# Predicting Cardiovascular Diseases from Clinical and Lifestyle Data

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies](https://img.shields.io/badge/Dependencies-NumPy%2C%20Matplotlib%2C%20Seaborn-lightgrey)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![License](https://img.shields.io/github/license/betty-gures/cs433project1)

This is the code repository for our Project 1 in the EPFL Course CS-433 "Machine Learning".

# Authors
- Abhinand Shibu
- Dominik Glandorf
- F. Betül Güres

# Summary
The project task was to implement and test machine learning methods to predict cardiovascular diseases. We found a weighted least squares model with non-parametric Platt scaling best performing, fast and stable to estimate (F1 score: 42.8%). Our project report is inside the repository: [Report](https://github.com/CS-433/project-1-ed4ml/blob/main/report.pdf). More information about the challenge can be found on [AICrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

# File structure

```
.
|-- data
|   |-- metadata
|   |   |-- feature_names.csv
|   |   |-- missing_percentages.csv
|   |   |-- missing_values.txt
|   |   `-- variable_type.txt
|   |-- submissions
|   |-- dataset.zip
|   `-- sample-submission.csv
|-- results
|-- LICENSE
|-- README.md
|-- abl_dataset_size.py
|-- abl_factors.py
|-- abl_fairness.py
|-- abl_pfi.py
|-- compare_models.py
|-- environment.yml
|-- helpers.py
|-- implementations.py
|-- metrics.py
|-- model_selection.py
|-- models.py
|-- preprocessing.py
|-- report.pdf
|-- run.py
`-- visualizations.py
```

## File description
- `data/metadata/` contains metadata about the dataset (feature names, missing values, variable types, etc.)
- `data/submissions`  contains all the submissions we made to AICrowd
- `data/dataset.zip` is the dataset provided by AICrowd in .zip format
- `results/` contains results from the experiments (plots, model performances, etc.)
- `abl_dataset_size.py` script to run dataset size experiment
- `abl_factors.py` script to run factor ablation experiments
- `abl_fairness.py` script to run fairness experiment across demographic groups
- `abl_pfi.py` script to run explainability experiment (permutation feature importance)
- `compare_models.py` script to compare different models using 5-fold cross-validation
- `environment.yml` conda environment file for development (same as grading environment but with ipykernel)
- `helpers.py` helper functions for data loading, preprocessing, etc.
- `implementations.py` implementations of the algorithms asked for in the project description
- `metrics.py` implementations of evaluation metrics: Fbeta score and AUROC
- `model_selection.py` functions cross-validation and group-based scoring
- `models.py` model implementations: OLS, Logistic Regression, Linear SVM, KNN, Decision Tree 
- `preprocessing.py` functions for data preprocessing, details see below
- `run.py` script to run the entire pipeline and produce the final AICrowd submission
- `visualizations.py` functions for visualizations (ROC curve, loss curves, etc)


# Requirements
- Python 3.9
- Libraries
    - `numpy=1.23.1`
    - `matplotlib=3.5.2`
    - `seaborn=0.13.2`

# Setup
Setup conda environment:
```bash
conda env create -f environment.yml
conda activate project1-dev
```

# Usage

The following sections describe how to reproduce our results.

## Final predictions
Model predictions using `python run.py`, reported in Table 1 under F1 AI Crowd. Due to private testset, results have to be manually submitted to the above mentioned AI Crowd Challenge.

Command line arguments:
```
 --model {ols,logistic_regression,linear_svm,knn,decision_tree}
                        The model to use for training and evaluation.
  --no_one_hot_encoding     If set, disables one-hot encoding for categorical features.
  --submission_file_name SUBMISSION_FILE_NAME
                        Name of the submission file (without dir and file name).
  --verbose             If set, prints detailed logs during training.
```

## Dataset Exploration
Reproduce results from the dataset exploration (Figure 1) using `python dataset_exploration.py` and see output in `results/feature_statistics_overview.pdf` .

## Model comparisons
Reproduce Table 1. Compare different models using `python compare_models.py`. Has the same arguments as `run.py` without `--submission_file_name`, but allows to run multiple models in one go and performs 5-fold cross-validation.

## Model ablations
Reproduces Table 2. Run preprocessing and modeling ablations using `python abl_factors.py` and see the output in the stdout.

## Dataset size and regularization ablation
Reproduces Figure 2. Run dataset size experiment using `python abl_dataset_size.py` and see output plot in `results/ablations_sample_size.pdf`.

## Fairness
Reproduces Table 3. Run fairness experiments using `python abl_fairness.py` and see output in `results/fairness_{demographic}.txt`.

## Explainability
Reproduces Results Section 3) Explainability. Run permutation feature importance using `python abl_pfi.py` and see output in `results/pfi.txt`.

# Preprocessing steps

Details of the preprocessing in `preprocessing.py`.

Split-independent preprocessing in `preprocess()`:
1. Replace codes that represent missingness with `np.nan` (values in `data/metadata/missing_values.txt`)
2. One-hot encode nominal features (in `data/metadata/variable_type.txt`)

Split-dependent preprocessing in `preprocess_splits()`:
1. Impute missing values using mean imputation
2. Remove handpicked redundant features as configured in `COL_REMOVE_MANUAL`
3. Remove invariant features
4. Standardize features to zero mean and unit variance
5. Squaring ordinal and continuous features (in `data/metadata/variable_type.txt`)
6. Remove original one hot encoded features to prevent multicollinearity
7. Add a bias term (column of ones)
8. Remove duplicate features

If the function is a called with a train and test set, the same preprocessing steps are applied to the test set using statistics (mean, std) computed on the training set to remove test data leakage.

# Models
Explanation of the specifics and tested hyperparameters of the following models in `models.py`:

General structure of all models:
- `__init__`: initializes the model with hyperparameters
- `hyperparameter_tuning(X, y, metric, verbose)`: tunes hyperparameters using a validation set split from the training data
- `train(X, y)`: fits the model to the training data
- `predict(X)`: predicts class labels for the input data, you can cache scores using the parameter `save_scores=True` for quicker hyperparameter tuning, especially import for kNN

## Weighted Ordinary Least Squares (OLS)

Our custom implementation of linear regression. We apply non-parametric Platt scaling (i.e. the sigmoid function) to convert the regression outputs to class probabilities. If weighting is set, the model multiplies the datapoints with their inverse class frequency weights in the loss function to handle class imbalance. The weights are estimated using `np.linalg.lstsq` for numerical stability in the call of ill-conditioned gram matrices X^TX.

### Hyperparameters:
- `decision_threshold` [0,1]: threshold to convert least squares output to class labels, if not tuned defaults to 0.5
- `_lambda` {0, 1e1, 1e2, 1e3, 1e4}: L2 regularization strength, excluding bias term
- `squared_features` {False, True}: whether to include squared ordinal and continuous features
- `weighting` {False, True}: use inverse class frequency weights to handle class imbalance

## Logistic Regression

Our custom implementation of logistic regression using full-batch gradient descent optimization. We intentionally did not estimate it using stochastic gradient descent or different optimizers because of the convexity of the loss and the dataset size that allowed for fast estimation on CPU with standard methods. If weighting is set, the model multiplies the datapoints with their inverse class frequency weights in the loss function to handle class imbalance.

### Hyperparameters:

- `decision_threshold` [0,1]: threshold to convert least squares output to class labels, if not tuned defaults to 0.5
- `gamma` [0, inf]: step size for gradient descent, defaults to 0.1, after we observed the smooth training loss using `plot_losses` from `visualizations.py`
- `_lambda` {0, 1e-4, 1e-3, 1e-2, 1e-1}: L2 regularization strength, excluding bias term
- `max_iters` [0-1000]: maximum number of iterations for gradient descent, set to the number of iterations after which the validation loss converged, defaults to 1000
- `patience` [0-inf]: number of iterations to wait for improvement in validation loss before early stopping, defaults to 25
- `squared_features` {False, True}: whether to include squared ordinal and continuous features
- `stopping_threshold` [0, inf]: threshold for early stopping based on the change in loss between iterations, defaults to 1e-4
- `weighting` {False, True}: use inverse class frequency weights to handle class imbalance

## Linear Support Vector Machine (SVM)

Support Vector Machine estimated using the primal form (Hinge loss) and optimized using full-batch gradient descent.

### Hyperparameters:
- `_lambda` {0.25, 0.5, 1.0, 2.0, 4.0}: L2 regularization strength, excluding bias term
- `lr`{0.01, 0.1}: learning rate for gradient descent
- `squared_features` {False, True}: whether to include squared ordinal and continuous features

## K-Nearest Neighbors (KNN)

KNN classifier using Euclidean distance. We apply PCA for dimensionality reduction before fitting the model to speed up predictions and reduce noise. The number of components is chosen is a hyperparameter, implicitly determined by the explained variance.

### Hyperparameters:
- `decision_threshold` [0,1]: threshold to convert least squares output to class labels, if not tuned defaults to 0.5
- `k`: number of neighbors, uses the square root of the number of samples as base and then uses the following fractions and multiples: {1/64, 1/32, 1/16, 1/8, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0}
- `squared_features` {False, True}: whether to include squared ordinal and continuous features
- `variance` {1/20, 1/10, 1/6, 1/4, 1/2, 0.75, 0.9, 1.0}: cumulative explained variance to determine the number of PCA components

## Decision Tree

Our Decision Tree splits based on Gini impurity and stops splitting when a maximum depth or minimum number of samples per leaf node is reached.

### Hyperparameters:
- `max_depth` {3, 4, 5, 6, 8, 10}: maximum depth of the decision tree
- `min_samples_split` {20, 50, 100, 200}: minimum number of leaf nodes after a split
- `squared_features` {False, True}: whether to include squared ordinal and continuous features

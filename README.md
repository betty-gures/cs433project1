# Predicting Cardiovascular Diseases from Clinical and Lifestyle Data

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the code repository for our Project 1 in the EPFL Course CS-433 "Machine Learning".

# Authors
- Abhinand Shibu
- Dominik Glandorf
- F. Betül Güres

# Summary
The project task was to implement and test machine learning methods to predict cardiovascular diseases. We found a weighted least squares model with non-parametric Platt scaling best performing, fast and stable to estimate (F1 score: 42.9%). Our project report is inside the repository: [Report](https://github.com/betty-gures/cs433project1/blob/main/report.pdf). More information about the challenge can be found on [AICrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

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
|-- grading_tests
|-- notebooks
|-- results
|   |-- KNearestNeighbors.txt
|   |-- LinearSVM.txt
|   |-- LogisticRegression.txt
|   |-- OrdinaryLeastSquares.txt
|   |-- ablations_sample_size.pdf
|   |-- feature_statistics_overview.pdf
|   `-- pfi.txt
|-- LICENSE
|-- README.md
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
- `grading_tests/` contains the tests and conda environment for grading the project
- `notebooks/` contains Jupyter notebooks for exploration and quick experimentation (not relevant for grading)
- `results/` contains results from the experiments (plots, model performances, etc.)
- `abl_pfi.py` script to run explainability experiment (permutation feature importance)
- `compare_models.py` script to compare different models using 5-fold cross-validation
- `environment.yml` conda environment file for development (same as grading environment but with ipykernel)
- `helpers.py` helper functions for data loading, preprocessing, etc.
- `implementations.py` implementations of the algorithms asked for in the project description
- `metrics.py` implementations of evaluation metrics: Fbeta score and AUROC
- `model_selection.py` functions cross-validation and group-based scoring
- `preprocessing.py` functions for data preprocessing (loading data, handling missing values, removing duplicate columns, etc.)
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

## Final predictions
Model predictions using `python run.py`.

Command line arguments:
```
 --model {ols,logistic_regression,linear_svm,knn}
                        The model to use for training and evaluation.
  --no_one_hot_encoding     If set, disables one-hot encoding for categorical features.
  --submission_file_name SUBMISSION_FILE_NAME
                        Name of the submission file (without dir and file name).
  --verbose             If set, prints detailed logs during training.
```

## Dataset Exploration
Run dataset exploration using `notebooks/002_dataset_exploration.ipynb` and see output in `results/feature_statistics_overview.pdf`

## Model comparisons
Compare different models using `compare_models.py`. Has the same arguments as `run.py` without `--submission_file_name`, but allows to run multiple models in one go and performs 5-fold cross-validation.

## Model ablations
Run preprocessing and modeling ablations using `notebooks/006_ablations.ipynb` and see output in `results/ablations_sample_size.pdf`

## Fairness
Run fairness experiments using `notebooks/008_fairness.ipynb` and see output in `results/fairness.txt`

## Explainability
Run permutation feature importance using `python abl_pfi.py` and see output in `results/pfi.txt`
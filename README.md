# Predicting Cardiovascular Diseases from Clinical and Lifestyle Data
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
|   |-- INSTRUCTIONS.md
|   |-- conftest.py
|   |-- environment.yml
|   `-- test_project1_public.py
|-- notebooks
|   |-- 001_implementations.ipynb
|   |-- 002_dataset_exploration.ipynb
|   |-- 003_preprocessing.ipynb
|   |-- 004_logistic_regression.ipynb
|   |-- 005_model_selection.ipynb
|   |-- 006_ablations.ipynb
|   |-- 007_neural_networks.ipynb
|   |-- 008_fairness.ipynb
|   |-- 009_least_squares.ipynb
|   |-- 010_SVM.ipynb
|   |-- 011_kNN.ipynb
|   |-- 012_PCA.ipynb
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


data 
    The files in the data folder, represent the columns from some of the manual processes we did for feature selection in google docs 
    - missing_values.txt
    - variable_type.txt
    - feature_names.csv
    - missing_percentages.csv

notebooks 
    These are messy development spaces, after which we move the code out into files for reuse

models.py
    Made of classes that represent the different models.

## File description
- `data/metadata/` contains metadata about the dataset (feature names, missing values, variable types, etc.)
- `data/submissions`  contains all the submissions we made to AICrowd
- `data/dataset.zip` is the dataset provided by AICrowd in .zip format
- `grading_tests/` contains the tests for grading the project
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


# Requirements
- Python 3.9
- Libraries
    - `numpy=1.23.1`
    - `matplotlib=3.5.2`
    - `seaborn=0.13.2`

# Setup
1. Setup conda environment:
```bash
conda env create -f environment.yml
conda activate cs433project1
```

# Usage

## Final predictions
Model predictions using `run.py`

## Dataset Exploration
Run dataset exploration using `notebooks/002_dataset_exploration.ipynb` and see output in `results/feature_statistics_overview.pdf`

## Model comparisons
Compare different models using `compare_models.py`

## Model ablations
Run preprocessing and modeling ablations using `notebooks/006_ablations.ipynb` and see output in `results/ablations_sample_size.pdf`

## Fairness
Run fairness experiments using `notebooks/008_fairness.ipynb` and see output in `results/fairness.txt`

## Explainability
Run permutation feature importance using `python abl_pfi.py` and see output in `results/pfi.txt`
# Predicting Cardiovascular Diseases from Clinical and Lifestyle Data
This is the code repository for our Project 1 in the EPFL Course CS-433 "Machine Learning".

# Authors
- Abhinand Shibu
- Dominik Glandorf
- F. Betül Güres

# Summary
The project task was to develop and test Machine Learning methods to predict cardiovascular diseases. We found a weighted least squares model with non-parametric Platt scaling best performing, fast and stable to estimate (F1 score: 42.9%). Our project report is inside the repository: [Report](https://github.com/betty-gures/cs433project1/blob/main/report.pdf). More information about the challenge can be found on [AICrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

# File structure

```
.
|-- data
|   |-- dataset
|   |-- submissions
|   |-- dataset.zip
|   |-- feature_names.csv
|   |-- missing_percentages.csv
|   |-- missing_values.txt
|   |-- sample-submission.csv
|   `-- variable_type.txt
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
|   `-- 012_PCA.ipynb
|-- results
|   |-- KNearestNeighbors.txt
|   |-- LogisticRegression.txt
|   |-- OrdinaryLeastSquares.txt
|   |-- ablations_sample_size.pdf
|   `-- feature_statistics_overview.pdf
|-- LICENSE
|-- README.md
|-- compare_models.py
|-- environment.yml
|-- helpers.py
|-- implementations.py
|-- metrics.py
|-- model_selection.py
|-- models.py
|-- preprocessing.py
|-- project1_description.pdf
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

# Requirements
- Python 3.9
- Libraries
    - `numpy`

# Setup


# Usage

Model predictions using `run.py`
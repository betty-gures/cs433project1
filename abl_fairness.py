"""Experiment script for evaluating fairness across demographic groups.

This script runs cross-validation experiments to evaluate model performance across different demographic groups defined by protected attributes such as race, sex, and age. It generates LaTeX-formatted tables summarizing the results and saves visualizations of class distributions by group.
"""

import os

import numpy as np

from model_selection import cross_validation
from helpers import MODEL_REGISTRY
from preprocessing import preprocess, get_raw_data
from visualizations import plot_class_distribution_by_group

# Configurations
NUM_SAMPLES = int(1e6)
DEMOGRAPHICS = {# taken from the dataset documentation
    "race": {"COL_IDX": 245, "VALUE_NAMES": {"1.0": "White", "2.0": "Black", "3.0": "Hispanic", "4.0": "Other", "5.0": "Multiracial", "nan": "Unknown"}},
    "sex": {"COL_IDX": 50, "VALUE_NAMES": {"1.0": "Male", "2.0": "Female"}},
    "age": {"COL_IDX": 247, "VALUE_NAMES": {"1.0": "Under 65", "2.0": "Above 65", "3.0": "Unknown"}},
}
MODEL = MODEL_REGISTRY["ols"]


def run_experiment(x, y, attr, num_samples = NUM_SAMPLES):
    """Runs cross-validation experiment on specified protected attribute.
    Args:
        attr: np.ndarray of shape (n_samples,), protected attribute
        num_samples: int, number of samples to use for the experiment

    Returns:
        results: results from cross-validation
    """
    return cross_validation(x[:num_samples], y[:num_samples], MODEL, scoring_groups=attr[:num_samples])

def format_results(results, value_names):
    """Formats results into LaTeX table.

    Args:
        results: results from cross-validation
        value_names: dict mapping attribute values to names
    Returns:
        latex_table: str, LaTeX formatted table
    """
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Estimated performance metrics (\\%±STD) by demographic group.}")
    latex.append("\\label{tab:results}")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lc}") # cc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Attribute} & \\textbf{F1-score (\\%)}\\")
    latex.append("\\midrule")
    for group in results.f1_scores[0].keys():
        f1_scores = [fold_result[group] for fold_result in results.f1_scores]
        f1_mean, f1_std = np.mean(f1_scores) * 100, np.std(f1_scores) * 100

        latex.append(f"{value_names[group]} & {f1_mean:.1f}±{f1_std:.1f}\\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex_table = "\n".join(latex)
    return latex_table

# Load raw data for demographic attributes
x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids = get_raw_data()

# Preprocess data
x_train, _, y_train, *_ = preprocess(one_hot_encoding=False)

os.makedirs("results", exist_ok=True)
for name, config in DEMOGRAPHICS.items():
    attr = x_train_orig[:, config["COL_IDX"]]
    # Create plot for analysis
    plot_class_distribution_by_group(y_train_orig, attr, save_dir=f"results/dist_{name}.pdf")

    results = run_experiment(x_train, y_train, attr)

    with open(f"results/fairness_{name}.txt", "w", encoding="utf-8") as f:
        f.write(format_results(results, config["VALUE_NAMES"]))
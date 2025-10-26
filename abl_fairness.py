import os

import numpy as np

from model_selection import cross_validation
from models import OrdinaryLeastSquares
from preprocessing import preprocess, get_raw_data
from visualizations import plot_class_distribution_by_group

# Configurations
NUM_SAMPLES = int(1e6)

x_train_orig, x_test_orig, y_train_orig, train_ids, test_ids = get_raw_data()
race = x_train_orig[:, 245]
sex = x_train_orig[:, 50]
age = x_train_orig[:, 247]

# Create distribution plots
plot_class_distribution_by_group(y_train_orig, race, save_dir="results/dist_race.pdf")

plot_class_distribution_by_group(y_train_orig, sex, save_dir="results/dist_sex.pdf")

plot_class_distribution_by_group(y_train_orig, age, save_dir="results/dist_age.pdf")

# Preprocess data
x_train, _, y_train, *_ = preprocess()

def run_experiment(attr, num_samples = NUM_SAMPLES):
    """Runs cross-validation experiment on specified protected attribute.
    Args:
        attr: np.ndarray of shape (n_samples,), protected attribute
        num_samples: int, number of samples to use for the experiment
    
    Returns:
        results: results from cross-validation
    """
    return cross_validation(x_train[:num_samples], y_train[:num_samples], OrdinaryLeastSquares, scoring_groups=attr[:num_samples])

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
    latex.append("\\caption{Estimated performance metrics (\\%±STD) by protected attribute.}")
    latex.append("\\label{tab:results}")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lc}") # cc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Attribute} & \\textbf{F1-score (\\%)}\\")# & \\textbf{F2-score (\\%)} & \\textbf{AUC-ROC (\\%)} \\\\")
    latex.append("\\midrule")
    for group in results.f1_scores[0].keys():
        f1_scores = [fold_result[group] for fold_result in results.f1_scores]
        f1_mean, f1_std = np.mean(f1_scores) * 100, np.std(f1_scores) * 100

        latex.append(f"{value_names[group]} & {f1_mean:.1f}±{f1_std:.1f}\\\\")#  & {f2_mean:.1f}±{f2_std:.1f} & {auc_mean:.1f}±{auc_std:.1f} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex_table = "\n".join(latex)
    return latex_table


results = run_experiment(race)
value_names = {"1.0": "White", "2.0": "Black", "3.0": "Hispanic", "4.0": "Other", "5.0": "Multiracial", "nan": "Unknown"}
with open("results/fairness_race.txt", "w", encoding="utf-8") as f:
    f.write(format_results(results, value_names))

results = run_experiment(sex)
value_names = {"1.0": "Male", "2.0": "Female"}
with open("results/fairness_sex.txt", "w", encoding="utf-8") as f:
    f.write(format_results(results, value_names))

results = run_experiment(age)
value_names = {"1.0": "Under 65", "2.0": "Above 65", "3.0": "Unknown"}
with open("results/fairness_age.txt", "w", encoding="utf-8") as f:
    f.write(format_results(results, value_names))
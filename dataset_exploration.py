from collections import Counter
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.patches import Patch
import seaborn as sns


import helpers
from preprocessing import preprocess, impute_missing_values

# Export feature names for reference and use in dataset preprocessing
with open("data/dataset/x_train.csv", "r") as f:
    feature_names = f.readline().strip().split(",")
with open("data/metadata/feature_names.csv", "w") as out_f:
    for name in feature_names:
        out_f.write(f"{name}\n")

x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data("data/dataset")

print(f"y_train.shape: {y_train.shape}")
print(f"The target contains these values: {np.unique(y_train)}")
unique, counts = np.unique(y_train, return_counts=True)
percentages = counts / counts.sum() * 100
for label, pct in zip(unique, percentages):
    print(f"Label {label}: {pct:.2f}%")


print(f"x_train.shape: {x_train.shape}")
print(f"x_test.shape: {x_test.shape}")
missing_values = (np.isnan(x_train).mean(axis=0))
print(f"Features with missing values: {(missing_values>0).sum()} / {x_train.shape[1]} ({(missing_values > 0.00).mean()*100:.1f}%)")
print(f"Features with more than 90% missing data: {(missing_values > 0.9).mean()*100:.1f}")
print(f"Recalculated features: {np.sum(['_' in name for name in feature_names])}")



# Calculate percentage of missing data for each feature (excluding the ID column)
missing_percentages = np.isnan(x_train).sum(axis=0) / x_train.shape[0] * 100

with open("data/metadata/missing_percentages.csv", "w") as out_f:
    for name, pct in zip(feature_names, missing_percentages):
        out_f.write(f"{pct:.1f}\n")


x_train, _, y_train, _ = preprocess(one_hot_encoding=False) # replace missing values with NaN

# Missingness
missing_perc = np.isnan(x_train).mean(axis=0)
sorted_missing = missing_perc[np.argsort(missing_perc)]

# Explained Variance
X = impute_missing_values(x_train, x_train) # PCA needs complete data
mean = X.mean(axis=0)
std = X.std(axis=0)
X[:,std>0] = (X[:,std>0] - mean[std>0]) / std[std>0]
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)  # shape (n_features, n_features)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Number of components without added variance: {np.sum(cumulative_variance > 1-1e-16)}")

# Variable types
# Number of unique values per feature and variable types
variable_type = []
with open("data/metadata/variable_type.txt", "r") as f:
    for line in f:
        line = line.strip().strip('"')  # remove whitespace and surrounding quotes
        # split by comma and convert to int
        if line == "":
            variable_type.append([])
        else:
            variable_type.append(line)
num_unique = []
for i in range(x_train.shape[1]):  # Skip ID column
        col = x_train[:, i] # skip ID column
        unique_vals = np.unique(col[~np.isnan(col)])
        n_unique = len(unique_vals)
        num_unique.append(n_unique)

unique_types = ["binary","nominal","ordinal","continuous"]  # preserve order
# Map variable_type strings to numeric codes following unique_types order
type_to_idx = {t: i for i, t in enumerate(unique_types)}
variable_idx = np.array([type_to_idx.get(t, len(unique_types)) for t in variable_type])

# Ensure num_unique is an array for lexsort
num_unique = np.array(num_unique)
order = np.lexsort((num_unique, variable_idx))
sorted_unique_vals = np.array(num_unique)[order]
sorted_var_types = np.array(variable_type)[order]
type_counts = Counter(sorted_var_types)

sns.set(style="whitegrid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True)

# --- Subplot 1: cumulative variance & missing values (percentage) ---
sns.lineplot(x=range(len(cumulative_variance)), y=cumulative_variance, ax=ax1, label="Cum. variance explained")
sns.lineplot(x=range(len(sorted_missing)), y=sorted_missing, ax=ax1, label="Missing Values")

ax1.set_ylabel("Percentage")
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax1.legend(loc="lower right")
ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))

# --- Subplot 2: unique values (log scale) ---
sns.lineplot(x=range(len(sorted_unique_vals)), y=sorted_unique_vals, ax=ax2, color="black", label="# Unique values")

ax2.set_ylabel("")
yticks = np.logspace(
    np.log10(1),
    np.log10(100000),
    6
)
ax2.set_yscale("log")
ax2.set_yticks(yticks)

colors = plt.cm.Pastel1.colors  # use soft, distinct colors
color_map = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}

for i, t in enumerate(sorted_var_types):
    if i == 0 or sorted_var_types[i] != sorted_var_types[i - 1]:
        start = i
        # Find where this region ends
        end = next((j for j in range(i + 1, len(sorted_var_types)) if sorted_var_types[j] != t), len(sorted_var_types))
        ax2.axvspan(start, end, color=color_map[t], alpha=0.75, zorder=0)
        
ax2.set_xlabel("Feature index sorted by statistic")
ax2.set_ylabel("Unique values (log)")
patches = [Patch(facecolor=color_map[t], alpha=0.75, label=f"{t} ({type_counts[t]})") for t in unique_types]
ax2.legend(handles=patches, loc="upper left", frameon=True)

ax2.grid(axis='x')
plt.xlim(-3, x_train.shape[1]+3)
plt.tight_layout()
plt.savefig("results/feature_statistics_overview.pdf")


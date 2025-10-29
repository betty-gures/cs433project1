import matplotlib.pyplot as plt
import numpy as np

from models import OrdinaryLeastSquares
from metrics import f_score
from preprocessing import preprocess


def mean_ci_normal(values, z = 1.96):
    """Return mean and approximate CI half-width using normal distribution (z=1.96).
    
    Args:
        values: list or np.ndarray of values
        z: float, z-score for confidence interval (default: 1.96)
    Returns:
        mean: float, mean of values
        h: float, half-width of 95% confidence interval
    """
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    n = len(arr)
    h = z * std / np.sqrt(n)
    return mean, h


ABLATION_MODEL = OrdinaryLeastSquares
num_samples = [int(1e2), int(5e2), int(1e3), int(5e3), int(1e4), int(2e4), int(5e4), int(1e5), int(2e5)]
NUM_SEEDS = 5
REG_STRENGTHS = [10e5,0]

x_train, _, y_train, _ = preprocess()

f1_train_scores, f1_val_scores = {}, {}
seeds = list(np.random.randint(0, 2**16 - 1, size=NUM_SEEDS))
test_size = 100000
for _lambda in REG_STRENGTHS:
    f1_train_scores[_lambda] = {}
    f1_val_scores[_lambda] = {}
    for n in num_samples:
        print(f"Running sample size: {n}")
        f1_train_scores[_lambda][n] = []
        f1_val_scores[_lambda][n] = []
        for seed in seeds:
            print(f" Seed: {seed}")
            rng = np.random.default_rng(seed)
            idxs = rng.choice(x_train.shape[0], size=n+test_size, replace=False)
            x_sample = x_train[idxs[:n]]
            y_sample = y_train[idxs[:n]]
            x_val_sample = x_train[idxs[n:]]
            y_val_sample = y_train[idxs[n:]]

            model = ABLATION_MODEL(_lambda=_lambda)
            model.train(x_sample, y_sample)

            y_pred_train = model.predict(x_sample)
            train_score = f_score(y_pred_train, y_sample)
            f1_train_scores[_lambda][n].append(train_score)

            y_pred_val = model.predict(x_val_sample)
            test_score = f_score(y_pred_val, y_val_sample)

            f1_val_scores[_lambda][n].append(test_score)
            print(train_score, test_score)

print([np.mean(f1_train_scores[_lambda][n]) for _lambda in f1_train_scores for n in num_samples])
print([np.mean(f1_val_scores[_lambda][n]) for _lambda in f1_val_scores for n in num_samples])


# Compute mean + CI per λ
results = {}
for _lambda in reversed(f1_train_scores.keys()):
    train_means, train_cis = [], []
    val_means, val_cis = [], []
    
    for n in num_samples:
        m, h = mean_ci_normal(f1_train_scores[_lambda][n])
        train_means.append(m)
        train_cis.append(h)
        m, h = mean_ci_normal(f1_val_scores[_lambda][n])
        val_means.append(m)
        val_cis.append(h)
    
    results[_lambda] = {
        "train_means": np.array(train_means),
        "train_cis": np.array(train_cis),
        "val_means": np.array(val_means),
        "val_cis": np.array(val_cis),
    }


# Plotting
plt.figure(figsize=(7, 2))

for _lambda, res in results.items():  # reverse order for legend
    label_suffix = f" (reg.)" if _lambda != 0 else ""
    plt.plot(num_samples, res["train_means"], '-o', label='Train' + label_suffix)
    plt.plot(num_samples, res["val_means"], '-o', label='Val.' + label_suffix)

    plt.fill_between(num_samples,
                     res["train_means"] - res["train_cis"],
                     res["train_means"] + res["train_cis"], alpha=0.2)
    plt.fill_between(num_samples,
                     res["val_means"] - res["val_cis"],
                     res["val_means"] + res["val_cis"], alpha=0.2)

plt.xscale('log')
plt.xticks(
    num_samples,
    [f"$10^2$", f"$5\\cdot10^2$", f"$10^3$", f"$5\\cdot10^3$", f"$10^4$", f"$2\\cdot10^4$", f"$5\\cdot10^4$", f"$10^5$", f"$2\\cdot10^5$"],
    rotation=0,
    
)

plt.xlabel("Sample size")
plt.ylabel("F1 score")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("results/ablations_sample_size.pdf")

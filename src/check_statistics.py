import json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy import stats

records = json.loads(Path("results/raw_results.json").read_text(encoding="utf-8"))
models = ["CNN5", "CNN5Wide", "CNN7"]
by_model = {}

for model in models:
    by_model[model] = {}
    for r in records:
        if r["model"] == model:
            by_model[model][r["seed"]] = r["best_val_accuracy"]

print("TOTAL RECORDS:", len(records))
for model in models:
    print(model, "n =", len(by_model[model]), "seeds =", sorted(by_model[model]))

if len(records) != 15 or any(len(by_model[m]) != 5 for m in models):
    raise ValueError("Expected 15 records and 5 runs per model.")

comparisons = []
print("\nPAIRED COMPARISONS")

for a, b in combinations(models, 2):
    seeds = sorted(set(by_model[a]) & set(by_model[b]))
    if len(seeds) != 5:
        raise ValueError(f"Seed mismatch: {a} vs {b}")

    x = np.array([by_model[a][s] for s in seeds])
    y = np.array([by_model[b][s] for s in seeds])
    d = x - y
    t = stats.ttest_rel(x, y)
    w = stats.wilcoxon(d, alternative="two-sided", method="exact")
    comparisons.append((a, b, float(t.pvalue)))

    print(f"\n{a} - {b}")
    print(f"Mean difference: {d.mean() * 100:.4f} percentage points")
    print(f"SD of differences: {d.std(ddof=1) * 100:.4f} percentage points")
    print(f"Paired t-test: t={t.statistic:.4f}, p={t.pvalue:.6f}")
    print(f"Wilcoxon exact: W={w.statistic}, p={w.pvalue:.6f}")

ordered = sorted(comparisons, key=lambda item: item[2])
adjusted = {}
running_max = 0.0

for rank, (a, b, p) in enumerate(ordered):
    running_max = max(running_max, min(1.0, (len(ordered) - rank) * p))
    adjusted[(a, b)] = running_max

print("\nHOLM-ADJUSTED P-VALUES FOR PAIRED T-TESTS")
for a, b, p in comparisons:
    print(f"{a} vs {b}: raw p={p:.6f}, Holm p={adjusted[(a, b)]:.6f}")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path("results/raw_results.json")
OUTPUT_DIR = Path("results")

MODEL_ORDER = ["CNN5", "CNN5Wide", "CNN7"]
EXPECTED_SEEDS = {42, 43, 44, 45, 46}


def load_results():
    """Load and validate the recorded experiment results."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list) or not results:
        raise ValueError("The results file must contain a non-empty list.")

    required_fields = {
        "model",
        "seed",
        "parameters",
        "best_epoch",
        "best_val_loss",
        "best_val_accuracy",
        "epochs_completed",
        "val_loss",
        "val_accuracy",
    }

    seen = set()

    for result in results:
        missing = required_fields - result.keys()
        if missing:
            raise ValueError(
                f"Record is missing fields: {sorted(missing)}"
            )

        key = (result["model"], result["seed"])

        if key in seen:
            raise ValueError(f"Duplicate experiment record: {key}")

        seen.add(key)

        if result["model"] not in MODEL_ORDER:
            raise ValueError(f"Unexpected model: {result['model']}")

        if not np.isfinite(result["best_val_accuracy"]):
            raise ValueError(f"Invalid validation accuracy: {key}")

        if not 0 <= result["best_val_accuracy"] <= 1:
            raise ValueError(f"Accuracy outside [0, 1]: {key}")

        val_loss = result["val_loss"]
        val_accuracy = result["val_accuracy"]

        if not val_loss or len(val_loss) != len(val_accuracy):
            raise ValueError(f"Invalid validation history: {key}")

        if len(val_loss) != result["epochs_completed"]:
            raise ValueError(f"Epoch count mismatch: {key}")

        if not np.all(np.isfinite(val_loss)):
            raise ValueError(f"Non-finite validation loss: {key}")

        best_index = int(np.argmin(val_loss))

        if result["best_epoch"] != best_index + 1:
            raise ValueError(
                f"Best epoch does not match minimum validation loss: {key}"
            )

        if not np.isclose(
            result["best_val_loss"], val_loss[best_index]
        ):
            raise ValueError(f"Best validation loss mismatch: {key}")

        if not np.isclose(
            result["best_val_accuracy"], val_accuracy[best_index]
        ):
            raise ValueError(
                f"Best validation accuracy mismatch: {key}"
            )

    expected = {
        (model, seed)
        for model in MODEL_ORDER
        for seed in EXPECTED_SEEDS
    }

    if seen != expected:
        missing = expected - seen
        unexpected = seen - expected
        raise ValueError(
            f"Experiment records do not match the expected design.\n"
            f"Missing: {sorted(missing)}\n"
            f"Unexpected: {sorted(unexpected)}"
        )

    print(
        f"Integrity check passed: {len(results)} records, "
        "3 architectures, 5 matched seeds."
    )

    return results


def create_dataframe(results):
    """Create a tidy table containing one row per experimental run."""
    df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "seed": r["seed"],
                "parameters": r["parameters"],
                "best_epoch": r["best_epoch"],
                "best_val_loss": r["best_val_loss"],
                "best_val_accuracy": r["best_val_accuracy"],
                "epochs_completed": r["epochs_completed"],
            }
            for r in results
        ]
    )

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return df.sort_values(["model", "seed"]).reset_index(drop=True)


def create_summary(df):
    """Calculate descriptive statistics across the five runs per model."""
    summary = (
        df.groupby("model", observed=True)
        .agg(
            n=("seed", "count"),
            parameters=("parameters", "first"),
            mean_val_accuracy=("best_val_accuracy", "mean"),
            std_val_accuracy=("best_val_accuracy", "std"),
            min_val_accuracy=("best_val_accuracy", "min"),
            max_val_accuracy=("best_val_accuracy", "max"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_epochs_completed=("epochs_completed", "mean"),
        )
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    for column in [
        "mean_val_accuracy",
        "std_val_accuracy",
        "min_val_accuracy",
        "max_val_accuracy",
    ]:
        summary[f"{column}_percent"] = summary[column] * 100

    return summary


def print_summary(summary):
    """Print readable summary statistics."""
    print("\n" + "=" * 72)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 72)

    for _, row in summary.iterrows():
        print(f"\n{row['model']}")
        print(f"  Runs: {int(row['n'])}")
        print(f"  Parameters: {int(row['parameters']):,}")
        print(
            f"  Mean validation accuracy: "
            f"{row['mean_val_accuracy_percent']:.3f}%"
        )
        print(
            f"  Standard deviation: "
            f"{row['std_val_accuracy_percent']:.3f} percentage points"
        )
        print(
            f"  Range: {row['min_val_accuracy_percent']:.3f}% - "
            f"{row['max_val_accuracy_percent']:.3f}%"
        )
        print(f"  Mean best epoch: {row['mean_best_epoch']:.2f}")
        print(
            f"  Mean epochs completed: "
            f"{row['mean_epochs_completed']:.2f}"
        )


def plot_accuracy_comparison(summary):
    """Plot mean accuracy with standard-deviation error bars."""
    means = summary["mean_val_accuracy_percent"].to_numpy()
    stds = summary["std_val_accuracy_percent"].to_numpy()
    models = summary["model"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        models,
        means,
        yerr=stds,
        capsize=6,
        error_kw={"elinewidth": 1.4},
    )

    ax.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in means],
        padding=7,
        fontsize=10,
    )

    # A zoomed scale makes small differences visible.
    # The subtitle explicitly identifies the truncated scale.
    lower = max(0, float(np.min(means - stds)) - 0.6)
    upper = min(100, float(np.max(means + stds)) + 0.6)
    ax.set_ylim(lower, upper)

    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_xlabel("CNN architecture")
    ax.set_title(
        "Validation Accuracy Across CNN Architectures\n"
        "Bars show mean ± 1 SD across five seeds; y-axis is truncated"
    )

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "validation_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_seed_results(df):
    """Plot individual run accuracies in matched seed order."""
    fig, ax = plt.subplots(figsize=(9, 6))

    markers = {
        "CNN5": "o",
        "CNN5Wide": "s",
        "CNN7": "^",
    }

    for model in MODEL_ORDER:
        model_data = (
            df[df["model"] == model]
            .sort_values("seed")
        )

        ax.plot(
            model_data["seed"],
            model_data["best_val_accuracy"] * 100,
            marker=markers[model],
            linewidth=1.8,
            markersize=7,
            label=model,
        )

    ax.set_xticks(sorted(EXPECTED_SEEDS))
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title(
        "Validation Accuracy by Matched Random Seed\n"
        "Each point represents one run"
    )

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Architecture")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "validation_accuracy_by_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    df = create_dataframe(results)
    summary = create_summary(df)

    summary.to_csv(
        OUTPUT_DIR / "summary_statistics.csv",
        index=False,
        float_format="%.8f",
    )

    # Export the run-level table for reproducibility.
    df.to_csv(
        OUTPUT_DIR / "run_level_results.csv",
        index=False,
        float_format="%.8f",
    )

    print_summary(summary)

    plot_accuracy_comparison(summary)
    plot_seed_results(df)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"- {OUTPUT_DIR / 'summary_statistics.csv'}")
    print(f"- {OUTPUT_DIR / 'run_level_results.csv'}")
    print(f"- {OUTPUT_DIR / 'validation_accuracy_comparison.png'}")
    print(f"- {OUTPUT_DIR / 'validation_accuracy_by_seed.png'}")
    print("\nNo model training was performed.")


if __name__ == "__main__":
    main()
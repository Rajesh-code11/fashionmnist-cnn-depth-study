
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results_test_comparison"
INPUT_CSV = RESULTS_DIR / "all_test_results.csv"
IMAGE_DIR = PROJECT_ROOT / "images"

MODEL_ORDER = ["CNN5", "CNN5Wide", "CNN7"]


def load_results():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        row["seed"] = int(row["seed"])
        row["test_accuracy"] = float(row["test_accuracy"])
        row["test_loss"] = float(row["test_loss"])

    return rows


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()

    summaries = {}

    for model in MODEL_ORDER:
        model_rows = [
            row for row in rows if row["model"] == model
        ]

        if len(model_rows) != 5:
            raise ValueError(
                f"Expected five runs for {model}; "
                f"found {len(model_rows)}."
            )

        accuracies = np.array(
            [r["test_accuracy"] * 100 for r in model_rows]
        )
        losses = np.array(
            [r["test_loss"] for r in model_rows]
        )

        summaries[model] = {
            "accuracy_mean": accuracies.mean(),
            "accuracy_sd": accuracies.std(ddof=1),
            "loss_mean": losses.mean(),
            "loss_sd": losses.std(ddof=1),
        }

    labels = MODEL_ORDER
    positions = np.arange(len(labels))

    # Figure 1: Mean test accuracy with SD error bars.
    means = [
        summaries[m]["accuracy_mean"] for m in labels
    ]
    errors = [
        summaries[m]["accuracy_sd"] for m in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        positions,
        means,
        yerr=errors,
        capsize=6,
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(
        "FashionMNIST Test Accuracy Across Five Seeds"
    )
    ax.set_ylim(min(means) - 0.8, 100)
    ax.bar_label(
        bars,
        labels=[f"{value:.2f}%" for value in means],
        padding=4,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        IMAGE_DIR / "test_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 2: Accuracy for every seed.
    fig, ax = plt.subplots(figsize=(9, 5))

    for model in MODEL_ORDER:
        model_rows = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: r["seed"],
        )
        seeds = [r["seed"] for r in model_rows]
        accuracies = [
            r["test_accuracy"] * 100 for r in model_rows
        ]

        ax.plot(
            seeds,
            accuracies,
            marker="o",
            linewidth=2,
            label=model,
        )

    ax.set_xlabel("Random seed")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Test Accuracy Across Random Seeds")
    ax.set_xticks([42, 43, 44, 45, 46])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        IMAGE_DIR / "test_accuracy_by_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Figure 3: Mean test loss with SD error bars.
    means = [
        summaries[m]["loss_mean"] for m in labels
    ]
    errors = [
        summaries[m]["loss_sd"] for m in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        positions,
        means,
        yerr=errors,
        capsize=6,
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Cross-entropy test loss")
    ax.set_title(
        "FashionMNIST Test Loss Across Five Seeds"
    )
    ax.bar_label(
        bars,
        labels=[f"{value:.4f}" for value in means],
        padding=4,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        IMAGE_DIR / "test_loss_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("PLOTS CREATED SUCCESSFULLY")
    print(f"Saved to: {IMAGE_DIR}")
    print(" - test_accuracy_comparison.png")
    print(" - test_accuracy_by_seed.png")
    print(" - test_loss_comparison.png")


if __name__ == "__main__":
    main()

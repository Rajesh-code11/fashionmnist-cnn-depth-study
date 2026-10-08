import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "results/raw_results.json"
OUTPUT_DIR = "results"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, "r") as file:
        results = json.load(file)

    rows = []

    for result in results:
        rows.append({
            "model": result["model"],
            "seed": result["seed"],
            "parameters": result["parameters"],
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "best_val_accuracy": result["best_val_accuracy"],
            "epochs_completed": result["epochs_completed"],
        })

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    summary = (
        df.groupby("model")
        .agg(
            parameters=("parameters", "first"),
            mean_val_accuracy=("best_val_accuracy", "mean"),
            std_val_accuracy=("best_val_accuracy", "std"),
            min_val_accuracy=("best_val_accuracy", "min"),
            max_val_accuracy=("best_val_accuracy", "max"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_epochs_completed=("epochs_completed", "mean"),
        )
        .reset_index()
    )

    summary["mean_val_accuracy_percent"] = (
        summary["mean_val_accuracy"] * 100
    )

    summary["std_val_accuracy_percent"] = (
        summary["std_val_accuracy"] * 100
    )

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "summary_statistics.csv"),
        index=False,
    )

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for _, row in summary.iterrows():

        print(
            f"\n{row['model']}"
        )

        print(
            f"Parameters: {int(row['parameters']):,}"
        )

        print(
            f"Mean validation accuracy: "
            f"{row['mean_val_accuracy']:.4f}"
        )

        print(
            f"Std validation accuracy: "
            f"{row['std_val_accuracy']:.4f}"
        )

        print(
            f"Range: "
            f"{row['min_val_accuracy']:.4f} - "
            f"{row['max_val_accuracy']:.4f}"
        )

        print(
            f"Mean best epoch: "
            f"{row['mean_best_epoch']:.2f}"
        )

    # ---------------------------------------------------------
    # Accuracy comparison plot
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    models = summary["model"]
    means = summary["mean_val_accuracy"] * 100
    stds = summary["std_val_accuracy"] * 100

    plt.bar(
        models,
        means,
        yerr=stds,
        capsize=5,
    )

    plt.ylabel("Validation Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Validation Accuracy Across CNN Architectures")

    plt.ylim(90, 95)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "validation_accuracy_comparison.png",
        ),
        dpi=300,
    )

    plt.close()

    # ---------------------------------------------------------
    # Individual seed results
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for model in df["model"].unique():

        model_data = df[df["model"] == model]

        plt.plot(
            model_data["seed"],
            model_data["best_val_accuracy"] * 100,
            marker="o",
            label=model,
        )

    plt.xlabel("Random Seed")
    plt.ylabel("Best Validation Accuracy (%)")
    plt.title("Validation Accuracy Across Random Seeds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "validation_accuracy_by_seed.png",
        ),
        dpi=300,
    )

    plt.close()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved:\n"
        f"- {OUTPUT_DIR}/summary_statistics.csv\n"
        f"- {OUTPUT_DIR}/validation_accuracy_comparison.png\n"
        f"- {OUTPUT_DIR}/validation_accuracy_by_seed.png"
    )


if __name__ == "__main__":
    main()
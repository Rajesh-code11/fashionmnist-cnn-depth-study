
import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from models import CNN5, CNN7, CNN5Wide
from utils import get_data_loaders


# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIGS = {
    "CNN5": {
        "class": CNN5,
        "checkpoint_dir": (
            PROJECT_ROOT / "results_cpu_cnn5" / "checkpoints"
        ),
    },
    "CNN7": {
        "class": CNN7,
        "checkpoint_dir": (
            PROJECT_ROOT / "results_cpu_additional" / "checkpoints"
        ),
    },
    "CNN5Wide": {
        "class": CNN5Wide,
        "checkpoint_dir": (
            PROJECT_ROOT / "results_cpu_additional" / "checkpoints"
        ),
    },
}

OUTPUT_DIR = PROJECT_ROOT / "results_test_comparison"

BATCH_SIZE = 128
SEEDS = [42, 43, 44, 45, 46]
DEVICE = torch.device("cpu")


def evaluate_checkpoint(model_class, checkpoint_path, test_loader):
    """Evaluate one saved checkpoint on the official test set."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = model_class().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("The test loader returned no examples.")

    metadata = checkpoint.get("metadata", {})

    return {
        "test_examples": total_examples,
        "test_loss": total_loss / total_examples,
        "test_accuracy": total_correct / total_examples,
        "best_epoch": metadata.get("best_epoch"),
        "best_validation_accuracy": metadata.get(
            "best_validation_accuracy"
        ),
    }


def main():
    print("=" * 72)
    print("FASHIONMNIST: OFFICIAL TEST-SET EVALUATION")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Check that every required checkpoint exists before evaluating.
    for model_name, config in MODEL_CONFIGS.items():
        checkpoint_dir = config["checkpoint_dir"]

        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory missing for {model_name}: "
                f"{checkpoint_dir}"
            )

        for seed in SEEDS:
            checkpoint_path = (
                checkpoint_dir / f"{model_name}_seed_{seed}.pt"
            )

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint_path}"
                )

    # Load the official test set once, using the existing pipeline.
    _, _, test_loader = get_data_loaders(
        data_dir=str(PROJECT_ROOT / "data"),
        batch_size=BATCH_SIZE,
        validation_ratio=0.1,
        seed=42,
        split_seed=2026,
        num_workers=0,
    )

    all_results = []

    for model_name, config in MODEL_CONFIGS.items():
        print("\n" + "-" * 72)
        print(f"Evaluating {model_name}")
        print("-" * 72)

        model_results = []

        for seed in SEEDS:
            checkpoint_path = (
                config["checkpoint_dir"]
                / f"{model_name}_seed_{seed}.pt"
            )

            print(f"\n{model_name} | Seed {seed}")

            metrics = evaluate_checkpoint(
                model_class=config["class"],
                checkpoint_path=checkpoint_path,
                test_loader=test_loader,
            )

            result = {
                "model": model_name,
                "seed": seed,
                **metrics,
            }

            model_results.append(result)
            all_results.append(result)

            print(
                f"Test examples: {metrics['test_examples']}"
            )
            print(
                f"Test loss: {metrics['test_loss']:.4f}"
            )
            print(
                f"Test accuracy: "
                f"{metrics['test_accuracy'] * 100:.2f}%"
            )

        accuracies = [
            result["test_accuracy"]
            for result in model_results
        ]
        losses = [
            result["test_loss"]
            for result in model_results
        ]

        summary = {
            "model": model_name,
            "number_of_seeds": len(model_results),
            "test_examples_per_run": (
                model_results[0]["test_examples"]
            ),
            "mean_test_accuracy": statistics.mean(accuracies),
            "sample_sd_test_accuracy": statistics.stdev(
                accuracies
            ),
            "min_test_accuracy": min(accuracies),
            "max_test_accuracy": max(accuracies),
            "mean_test_loss": statistics.mean(losses),
            "sample_sd_test_loss": statistics.stdev(losses),
            "seeds": SEEDS,
        }

        print(f"\n{model_name} summary")
        print(
            f"Mean test accuracy: "
            f"{summary['mean_test_accuracy'] * 100:.2f}%"
        )
        print(
            f"Sample SD: "
            f"{summary['sample_sd_test_accuracy'] * 100:.2f} "
            "percentage points"
        )
        print(
            f"Mean test loss: {summary['mean_test_loss']:.4f}"
        )

    # Build combined summary for all three architectures.
    summaries = []

    for model_name in MODEL_CONFIGS:
        rows = [
            result for result in all_results
            if result["model"] == model_name
        ]

        accuracies = [row["test_accuracy"] for row in rows]
        losses = [row["test_loss"] for row in rows]

        summaries.append({
            "model": model_name,
            "number_of_seeds": len(rows),
            "test_examples_per_run": rows[0]["test_examples"],
            "mean_test_accuracy": statistics.mean(accuracies),
            "sample_sd_test_accuracy": statistics.stdev(
                accuracies
            ),
            "min_test_accuracy": min(accuracies),
            "max_test_accuracy": max(accuracies),
            "mean_test_loss": statistics.mean(losses),
            "sample_sd_test_loss": statistics.stdev(losses),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save all 15 individual runs.
    results_csv = OUTPUT_DIR / "all_test_results.csv"
    with results_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Save individual results and architecture-level summaries.
    results_json = OUTPUT_DIR / "all_test_results.json"
    with results_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "evaluation": {
                    "device": str(DEVICE),
                    "batch_size": BATCH_SIZE,
                    "seeds": SEEDS,
                    "test_split": "Official FashionMNIST test set",
                    "checkpoint_selection": (
                        "Saved checkpoint selected by lowest "
                        "validation loss"
                    ),
                },
                "individual_results": all_results,
                "architecture_summaries": summaries,
            },
            file,
            indent=4,
        )

    summary_csv = OUTPUT_DIR / "architecture_summary.csv"
    with summary_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print("\n" + "=" * 72)
    print("FINAL TEST-SET COMPARISON")
    print("=" * 72)
    print(
        f"{'Model':<12} {'Mean accuracy':>15} "
        f"{'SD (pp)':>10} {'Mean loss':>12}"
    )
    print("-" * 55)

    for summary in summaries:
        print(
            f"{summary['model']:<12} "
            f"{summary['mean_test_accuracy'] * 100:>14.2f}% "
            f"{summary['sample_sd_test_accuracy'] * 100:>9.2f} "
            f"{summary['mean_test_loss']:>12.4f}"
        )

    print("\nEVALUATION COMPLETE")
    print(f"Individual results: {results_csv}")
    print(f"Detailed JSON:      {results_json}")
    print(f"Architecture summary: {summary_csv}")
    print("\nNo models were retrained.")


if __name__ == "__main__":
    main()

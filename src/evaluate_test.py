
import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn

from models import CNN5
from utils import get_data_loaders


# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results_cpu_cnn5"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

BATCH_SIZE = 128
SEEDS = [42, 43, 44, 45, 46]
DEVICE = torch.device("cpu")


def main():
    print("=" * 65)
    print("FASHIONMNIST CNN5 TEST-SET EVALUATION")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {CHECKPOINT_DIR}"
        )

    # Use the project's existing data-loading pipeline so the
    # official test set receives the same preprocessing.
    _, _, test_loader = get_data_loaders(
        data_dir=str(PROJECT_ROOT / "data"),
        batch_size=BATCH_SIZE,
        validation_ratio=0.1,
        seed=42,
        split_seed=2026,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    results = []

    for seed in SEEDS:
        checkpoint_path = (
            CHECKPOINT_DIR / f"CNN5_seed_{seed}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint_path}"
            )

        print(f"\nEvaluating CNN5 | Seed {seed}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model = CNN5().to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

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

        test_loss = total_loss / total_examples
        test_accuracy = total_correct / total_examples

        metadata = checkpoint.get("metadata", {})

        result = {
            "model": "CNN5",
            "seed": seed,
            "test_examples": total_examples,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_epoch": metadata.get("best_epoch"),
            "best_validation_accuracy": metadata.get(
                "best_validation_accuracy"
            ),
        }

        results.append(result)

        print(f"Test examples: {total_examples}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Summarize results across the five independently trained models.
    accuracies = [r["test_accuracy"] for r in results]
    losses = [r["test_loss"] for r in results]

    summary = {
        "model": "CNN5",
        "number_of_seeds": len(results),
        "test_examples_per_run": results[0]["test_examples"],
        "mean_test_accuracy": statistics.mean(accuracies),
        "sample_sd_test_accuracy": statistics.stdev(accuracies),
        "mean_test_loss": statistics.mean(losses),
        "sample_sd_test_loss": statistics.stdev(losses),
        "seeds": SEEDS,
    }

    csv_path = RESULTS_DIR / "test_results.csv"
    json_path = RESULTS_DIR / "test_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "individual_results": results,
                "summary": summary,
            },
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("TEST EVALUATION COMPLETE")
    print("=" * 65)
    print(
        f"Mean test accuracy: "
        f"{summary['mean_test_accuracy'] * 100:.2f}%"
    )
    print(
        f"Sample SD: "
        f"{summary['sample_sd_test_accuracy'] * 100:.2f} "
        "percentage points"
    )
    print(f"Mean test loss: {summary['mean_test_loss']:.4f}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
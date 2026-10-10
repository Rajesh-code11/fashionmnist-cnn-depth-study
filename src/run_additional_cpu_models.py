
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from models import CNN7, CNN5Wide, count_parameters
from train import train_model, set_seed
from utils import get_data_loaders, get_device


# =========================================================
# ADDITIONAL CPU EXPERIMENT: CNN7 AND CNN5WIDE
# =========================================================

MODELS = {
    "CNN7": CNN7,
    "CNN5Wide": CNN5Wide,
}

SEEDS = [42, 43, 44, 45, 46]

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
SPLIT_SEED = 2026
CPU_THREADS = 6

# Keep all new results separate from previous experiments.
OUTPUT_DIR = Path("results_cpu_additional")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_PATH = OUTPUT_DIR / "raw_results.json"


def atomic_json_save(data, path):
    """Write JSON atomically to avoid incomplete result files."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temp_file:
        json.dump(data, temp_file, indent=4)
        temp_path = Path(temp_file.name)

    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def checkpoint_path(model_name, seed):
    return CHECKPOINT_DIR / f"{model_name}_seed_{seed}.pt"


def save_checkpoint(model, metadata, path):
    """Save the trained model weights and experiment metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_completed_results():
    """Resume safely without repeating completed runs."""
    if not RESULTS_PATH.exists():
        return []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    completed = []
    seen = set()

    for result in results:
        model_name = result["model"]
        seed = int(result["seed"])
        key = (model_name, seed)

        if key in seen:
            raise ValueError(f"Duplicate run found: {key}")

        seen.add(key)

        if model_name not in MODELS or seed not in SEEDS:
            raise ValueError(f"Unexpected run in results: {key}")

        # A run is complete only when its checkpoint exists.
        if checkpoint_path(model_name, seed).exists():
            completed.append(result)
        else:
            print(
                f"Missing checkpoint for {key}; "
                "this run will be repeated."
            )

    return completed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(CPU_THREADS)
    device = get_device()

    print("=" * 70)
    print("CNN7 AND CNN5WIDE CPU EXPERIMENT")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Seeds: {SEEDS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Early-stopping patience: {PATIENCE}")
    print(f"Validation split seed: {SPLIT_SEED}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    if device.type != "cpu":
        raise RuntimeError(
            "This script is intended for CPU training. "
            "No runs were started."
        )

    results = load_completed_results()

    completed_runs = {
        (item["model"], int(item["seed"]))
        for item in results
    }

    atomic_json_save(results, RESULTS_PATH)

    for model_name, model_class in MODELS.items():
        for seed in SEEDS:
            key = (model_name, seed)

            if key in completed_runs:
                print(f"\nSkipping completed run: {key}")
                continue

            print("\n" + "=" * 70)
            print(f"Starting {model_name} | Seed {seed}")
            print("=" * 70)

            set_seed(seed)

            train_loader, val_loader, _ = get_data_loaders(
                batch_size=BATCH_SIZE,
                seed=seed,
                split_seed=SPLIT_SEED,
                num_workers=0,
            )

            model = model_class()
            parameters = count_parameters(model)

            print(f"Trainable parameters: {parameters:,}")

            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                patience=PATIENCE,
                seed=seed,
            )

            if not history["val_loss"]:
                raise RuntimeError(
                    f"No validation history for {key}"
                )

            best_epoch = int(np.argmin(history["val_loss"])) + 1
            best_val_loss = float(
                history["val_loss"][best_epoch - 1]
            )
            best_val_accuracy = float(
                history["val_accuracy"][best_epoch - 1]
            )

            record = {
                "model": model_name,
                "seed": seed,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "epochs_completed": len(history["train_loss"]),
                "train_loss": history["train_loss"],
                "train_accuracy": history["train_accuracy"],
                "val_loss": history["val_loss"],
                "val_accuracy": history["val_accuracy"],
            }

            metadata = {
                "model": model_name,
                "seed": seed,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": EPOCHS,
                "patience": PATIENCE,
                "split_seed": SPLIT_SEED,
                "selection_metric": "minimum validation loss",
                "device": "cpu",
                "test_set_evaluated": False,
            }

            # Save the checkpoint before recording the run as complete.
            save_checkpoint(
                model,
                metadata,
                checkpoint_path(model_name, seed),
            )

            results.append(record)
            atomic_json_save(results, RESULTS_PATH)
            completed_runs.add(key)

            print(f"Checkpoint saved: {checkpoint_path(model_name, seed)}")
            print(f"Best epoch: {best_epoch}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Best validation accuracy: {best_val_accuracy:.4%}")
            print(f"Completed additional runs: {len(completed_runs)}/10")

    print("\n" + "=" * 70)
    print("ADDITIONAL CPU TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results: {RESULTS_PATH.resolve()}")
    print(f"Checkpoints: {CHECKPOINT_DIR.resolve()}")
    print("The official test set has not been evaluated by this script.")


if __name__ == "__main__":
    main()

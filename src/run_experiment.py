
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from models import CNN5, count_parameters
from train import train_model, set_seed
from utils import get_data_loaders, get_device


# ---------------------------------------------------------
# CPU EXPERIMENT: CNN5, FIVE SEEDS
# ---------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
SPLIT_SEED = 2026

# Keep this separate from the original experiment outputs.
OUTPUT_DIR = Path("results_cpu_cnn5")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_PATH = OUTPUT_DIR / "raw_results.json"

# Avoid oversubscribing the CPU.
CPU_THREADS = 6


def atomic_json_save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temp_file:
        json.dump(data, temp_file, indent=4)
        temp_path = Path(temp_file.name)

    os.replace(temp_path, path)


def checkpoint_path(seed):
    return CHECKPOINT_DIR / f"CNN5_seed_{seed}.pt"


def save_checkpoint(model, metadata, path):
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
    if not RESULTS_PATH.exists():
        return []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    completed = []
    seen = set()

    for result in results:
        seed = int(result["seed"])

        if seed in seen:
            raise ValueError(f"Duplicate CNN5 seed: {seed}")

        seen.add(seed)

        # A run is complete only if its checkpoint also exists.
        if checkpoint_path(seed).exists():
            completed.append(result)
        else:
            print(
                f"Seed {seed} has metrics but no checkpoint; "
                "it will be rerun."
            )

    return completed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(CPU_THREADS)
    device = get_device()

    print("=" * 65)
    print("CNN5 CPU EXPERIMENT")
    print("=" * 65)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print()

    if device.type != "cpu":
        print("CUDA is available; this runner is intended for CPU.")
        print("Stopping to avoid an unintended device change.")
        return

    results = load_completed_results()
    completed_seeds = {int(item["seed"]) for item in results}

    atomic_json_save(results, RESULTS_PATH)

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"Skipping completed seed {seed}")
            continue

        print("\n" + "=" * 65)
        print(f"Starting CNN5 | Seed {seed}")
        print("=" * 65)

        set_seed(seed)

        train_loader, val_loader, _ = get_data_loaders(
            batch_size=BATCH_SIZE,
            seed=seed,
            split_seed=SPLIT_SEED,
            num_workers=0,
        )

        model = CNN5()
        parameters = count_parameters(model)

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

        best_epoch = int(np.argmin(history["val_loss"])) + 1
        best_val_loss = float(history["val_loss"][best_epoch - 1])
        best_val_accuracy = float(
            history["val_accuracy"][best_epoch - 1]
        )

        record = {
            "model": "CNN5",
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
            "model": "CNN5",
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
            "test_set_evaluated": False,
        }

        # Save weights before marking the run as complete.
        save_checkpoint(model, metadata, checkpoint_path(seed))

        results.append(record)
        atomic_json_save(results, RESULTS_PATH)

        print(f"Saved checkpoint for seed {seed}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {best_val_accuracy:.4%}")
        print(f"Completed runs: {len(results)}/{len(SEEDS)}")

    print("\n" + "=" * 65)
    print("CNN5 CPU TRAINING COMPLETE")
    print("=" * 65)
    print(f"Results: {RESULTS_PATH.resolve()}")
    print(f"Checkpoints: {CHECKPOINT_DIR.resolve()}")
    print("The test set has not been evaluated.")


if __name__ == "__main__":
    main()
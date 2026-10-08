import json
import os

from models import CNN5, CNN7, CNN5Wide, count_parameters
from train import train_model, set_seed
from utils import get_data_loaders, get_device


SEEDS = [42, 43, 44, 45, 46]

MODELS = {
    "CNN5": CNN5,
    "CNN7": CNN7,
    "CNN5Wide": CNN5Wide,
}

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5


def run_single_experiment(model_name, model_class, seed, device):

    print("=" * 70)
    print(f"Model: {model_name} | Seed: {seed}")
    print("=" * 70)

    set_seed(seed)

    train_loader, val_loader, _ = get_data_loaders(
        batch_size=BATCH_SIZE,
        seed=seed,
    )

    model = model_class()

    parameter_count = count_parameters(model)

    print(f"Parameters: {parameter_count:,}")
    print(f"Device: {device}")
    print()

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

    best_epoch = history["val_loss"].index(
        min(history["val_loss"])
    ) + 1

    best_val_loss = min(history["val_loss"])

    best_val_accuracy = history["val_accuracy"][
        best_epoch - 1
    ]

    return {
        "model": model_name,
        "seed": seed,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "epochs_completed": len(history["train_loss"]),
        "train_loss": history["train_loss"],
        "train_accuracy": history["train_accuracy"],
        "val_loss": history["val_loss"],
        "val_accuracy": history["val_accuracy"],
    }


def main():

    device = get_device()

    os.makedirs("results", exist_ok=True)

    output_path = "results/raw_results.json"

    all_results = []

    for model_name, model_class in MODELS.items():

        for seed in SEEDS:

            result = run_single_experiment(
                model_name=model_name,
                model_class=model_class,
                seed=seed,
                device=device,
            )

            all_results.append(result)

            # Save after every completed run
            with open(output_path, "w") as file:
                json.dump(
                    all_results,
                    file,
                    indent=4,
                )

            print()
            print(
                f"Best validation accuracy: "
                f"{result['best_val_accuracy']:.4f}"
            )

            print(
                f"Best epoch: "
                f"{result['best_epoch']}"
            )

            print(
                f"Completed runs: "
                f"{len(all_results)}/15"
            )

            print()

    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
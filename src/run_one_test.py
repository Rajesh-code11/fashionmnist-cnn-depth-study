import csv
import os

import torch
import torch.nn as nn

from models import CNN5, CNN5Wide, CNN7, count_parameters
from train import train_model, set_seed, evaluate
from utils import get_data_loaders, get_device


def main():
    seed = 42
    batch_size = 128
    learning_rate = 1e-3
    epochs = 10
    patience = 5

    set_seed(seed)

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size,
        seed=seed,
    )

    device = get_device()
    criterion = nn.CrossEntropyLoss()

    models = {
        "CNN5": CNN5(),
        "CNN5Wide": CNN5Wide(),
        "CNN7": CNN7(),
    }

    results = []

    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print()

    for model_name, model in models.items():

        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        print(f"Parameters: {count_parameters(model):,}")

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            seed=seed,
        )

        best_epoch = history["val_loss"].index(
            min(history["val_loss"])
        ) + 1

        best_val_loss = min(history["val_loss"])

        best_val_accuracy = history["val_accuracy"][
            best_epoch - 1
        ]

        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print()
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Validation accuracy: {best_val_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print()

        results.append({
            "model": model_name,
            "seed": seed,
            "parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "validation_loss": best_val_loss,
            "validation_accuracy": best_val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        })

    os.makedirs("results", exist_ok=True)

    output_path = "results/test_results.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("=" * 70)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print()

    for result in results:
        print(
            f"{result['model']}: "
            f"Validation = {result['validation_accuracy']:.4f}, "
            f"Test = {result['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
import torch

from models import CNN5, count_parameters
from train import train_model
from utils import get_data_loaders, get_device


def main():
    seed = 42
    batch_size = 128

    train_loader, val_loader, _ = get_data_loaders(
        batch_size=batch_size,
        seed=seed,
    )

    device = get_device()

    model = CNN5()

    print(f"Device: {device}")
    print(f"Model: CNN5")
    print(f"Parameters: {count_parameters(model):,}")
    print()

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=2,
        learning_rate=1e-3,
        patience=5,
        seed=seed,
    )

    print()
    print("Sanity check completed.")
    print(f"Epochs completed: {len(history['train_loss'])}")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
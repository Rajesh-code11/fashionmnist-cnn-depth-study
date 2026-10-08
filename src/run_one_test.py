from models import CNN5, count_parameters
from train import train_model, set_seed
from utils import get_data_loaders, get_device


def main():
    seed = 42
    set_seed(seed)

    train_loader, val_loader, _ = get_data_loaders(
        batch_size=128,
        seed=seed,
    )

    device = get_device()

    model = CNN5()

    print(f"Device: {device}")
    print(f"Model: CNN5")
    print(f"Seed: {seed}")
    print(f"Parameters: {count_parameters(model):,}")
    print()

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=3,
        learning_rate=1e-3,
        patience=5,
        seed=seed,
    )

    best_epoch = history["val_loss"].index(
        min(history["val_loss"])
    ) + 1

    best_val_loss = min(history["val_loss"])

    best_val_accuracy = history["val_accuracy"][
        best_epoch - 1
    ]

    print()
    print("=" * 60)
    print("ONE-RUN TEST COMPLETE")
    print("=" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
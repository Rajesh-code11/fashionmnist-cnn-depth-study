import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(
    data_dir="data",
    batch_size=128,
    validation_ratio=0.1,
    seed=42,
    split_seed=2026,
    num_workers=0,
):
    """
    Create FashionMNIST train, validation, and test loaders.

    The train/validation split is fixed using split_seed.
    The seed controls training/shuffling reproducibility.

    Images remain at native 28x28 resolution.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    validation_size = int(
        len(train_dataset) * validation_ratio
    )

    train_size = len(train_dataset) - validation_size

    split_generator = torch.Generator().manual_seed(split_seed)

    train_dataset, validation_dataset = random_split(
        train_dataset,
        [train_size, validation_size],
        generator=split_generator,
    )

    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader


def get_device():
    """Return CUDA if available, otherwise CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
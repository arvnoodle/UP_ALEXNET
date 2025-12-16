"""
Data Loading and Preprocessing
Extracted from the original alexnetv3.py by JM
Modular, configurable data pipeline for production use
Modified for Anime Classification (3D, 90s, Modern)
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_data_loaders(config):
    """
    Create train, calibration, and test data loaders based on configuration.

    Args:
        config (dict): Configuration dictionary containing data settings

    Returns:
        tuple: (train_loader, calib_loader, test_loader)
    """
    print("✅ USING UPDATED get_data_loaders() FROM:", __file__)
    

    # Extract config
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["training"]

    batch_size = train_config["batch_size"]
    num_workers = data_config["num_workers"]
    pin_memory = data_config["pin_memory"]
    drop_last = data_config["drop_last"]
    data_dir = data_config["data_dir"]
    image_dim = model_config["image_dim"]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Directories
    train_dir = os.path.join(data_dir, "train")
    calib_dir = os.path.join(data_dir, "calib")
    test_dir  = os.path.join(data_dir, "test")
    

    # Basic existence checks
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if not os.path.exists(calib_dir):
        raise FileNotFoundError(
            f"Calibration directory not found: {calib_dir}\n"
            "Create it as data/calib/<class_name>/... or update config['data']['data_dir']."
        )

    if not os.path.exists(test_dir):
        print("⚠️ Warning: Test directory not found. Using calibration directory as test set.")
        test_dir = calib_dir

    # Datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    calib_dataset = ImageFolder(root=calib_dir, transform=eval_transform)
    test_dataset  = ImageFolder(root=test_dir,  transform=eval_transform)

    print("Datasets created")
    print(f"Training samples:    {len(train_dataset)}")
    print(f"Calibration samples: {len(calib_dataset)}")
    print(f"Test samples:        {len(test_dataset)}")

    # (Optional) sanity check: classes should match across splits
    if train_dataset.classes != calib_dataset.classes:
        print("⚠️ Warning: Train and calib classes do not match!")
        print("Train classes:", train_dataset.classes)
        print("Calib classes:", calib_dataset.classes)

    if train_dataset.classes != test_dataset.classes:
        print("⚠️ Warning: Train and test classes do not match!")
        print("Train classes:", train_dataset.classes)
        print("Test classes:", test_dataset.classes)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # keep all calib samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # keep all test samples
    )

    print("Dataloaders created")
    return train_loader, calib_loader, test_loader


def get_class_names():
    """Optional: fixed class list (only use if you truly need hardcoded names)."""
    return ["3d_anime", "90s_anime", "modern_anime"]


if __name__ == "__main__":
    print("data_loader ran successfully")

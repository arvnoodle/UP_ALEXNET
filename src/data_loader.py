"""
Data Loading and Preprocessing
Extracted from the original alexnetv3.py by JM
Modular, configurable data pipeline for production use
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_data_loaders(config):
    """
    Create train and test data loaders based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing data settings
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Extract data configuration
    data_config = config['data']
    model_config = config['model']
    
    batch_size = config['training']['batch_size']
    num_workers = data_config['num_workers']
    pin_memory = data_config['pin_memory']
    drop_last = data_config['drop_last']
    data_dir = data_config['data_dir']
    image_dim = model_config['image_dim']
    
    # Define transformation and augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),  # to resize CIFAR-10 images and retain the AlexNet architecture as is
        transforms.RandomHorizontalFlip(),  # data augmentation for training data
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Define transformation for testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),  # to resize CIFAR-10 images and retain the AlexNet architecture as is
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the CIFAR-10 dataset
    train_dataset = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    
    print('Datasets created')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=drop_last,
        batch_size=batch_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,  # No need to shuffle test data
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=drop_last,
        batch_size=batch_size
    )
    
    print('Dataloaders created')
    
    return train_loader, test_loader


def get_class_names():
    """
    Get CIFAR-10 class names.
    
    Returns:
        list: List of class names
    """
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

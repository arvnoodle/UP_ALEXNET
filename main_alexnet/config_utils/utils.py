"""
Utility functions for training and inference
Helper functions for checkpointing, device management, and metrics
"""

import os
import torch
import yaml


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """
    Setup device for training/inference.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        torch.device: Device to use
    """
    device_config = config['device']
    
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def create_directories(config):
    """
    Create necessary directories for outputs.
    
    Args:
        config (dict): Configuration dictionary
    """
    paths = config['paths']
    
    # Create directories if they don't exist
    os.makedirs(paths['output_dir'], exist_ok=True)
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)
    os.makedirs(paths['log_dir'], exist_ok=True)
    
    print(f"Created directories:")
    print(f"  Output: {paths['output_dir']}")
    print(f"  Checkpoints: {paths['checkpoint_dir']}")
    print(f"  Logs: {paths['log_dir']}")


def save_checkpoint(model, optimizer, scheduler, epoch, total_steps, seed, checkpoint_path):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        epoch (int): Current epoch
        total_steps (int): Total training steps
        seed (int): Random seed
        checkpoint_path (str): Path to save checkpoint
    """
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'model': model.state_dict(),
        'seed': seed,
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        checkpoint_path (str): Path to checkpoint
        device: Device to load on
        
    Returns:
        dict: Checkpoint state
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['total_steps']}")
    
    return checkpoint


def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy for a batch.
    
    Args:
        outputs: Model outputs
        labels: True labels
        
    Returns:
        float: Accuracy percentage
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def get_model_summary(model, input_size):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size (tuple): Input tensor size
    """
    # Create a dummy input
    dummy_input = torch.randn(input_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input size: {input_size}")

"""
Training Pipeline for AlexNet
Production-ready training script with CLI interface
Combines and cleans up the training logic from the original alexnetv3.py
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from model import AlexNet
from data_loader import get_data_loaders
from config_utils import (
    load_config, setup_device, create_directories, 
    save_checkpoint, load_checkpoint, calculate_accuracy, 
    set_seed, get_model_summary
)


def train_model(config, resume_checkpoint=None):
    """
    Main training function.
    
    Args:
        config (dict): Configuration dictionary
        resume_checkpoint (str, optional): Path to checkpoint to resume from
    """
    # Setup
    set_seed(config['training']['seed'])
    device = setup_device(config)
    create_directories(config)
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(config)
    
    # Create model
    model = AlexNet(num_classes=config['model']['num_classes']).to(device)
    
    # Handle multi-GPU training
    device_ids = config['device']['device_ids']
    if len(device_ids) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model, device_ids=device_ids)
        print(f"Using DataParallel with GPUs: {device_ids}")
    
    # Print model summary
    get_model_summary(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training']['lr_step_size'],
        gamma=config['training']['lr_gamma']
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=config['paths']['log_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    total_steps = 1
    if resume_checkpoint:
        checkpoint = load_checkpoint(model, optimizer, scheduler, resume_checkpoint, device)
        start_epoch = checkpoint['epoch'] + 1
        total_steps = checkpoint['total_steps']
    
    # Training configuration
    num_epochs = config['training']['num_epochs']
    checkpoint_dir = config['paths']['checkpoint_dir']
    
    print(f"Starting training from epoch {start_epoch} to {num_epochs}")
    print(f"Total training steps so far: {total_steps}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_steps = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move tensors to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            epoch_steps += 1
            total_steps += 1
            
            # Log to TensorBoard every 10 steps
            if total_steps % 10 == 0:
                accuracy = calculate_accuracy(outputs, labels)
                writer.add_scalar('Loss/Train', loss.item(), total_steps)
                writer.add_scalar('Accuracy/Train', accuracy, total_steps)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], total_steps)
                
                print(f'Epoch: {epoch+1} | Step: {total_steps} | Loss: {loss.item():.4f} | Acc: {accuracy:.2f}%')
            
            # Log gradients and weights every 100 steps
            if total_steps % 100 == 0:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        writer.add_scalar(f'Gradients/{name}', avg_grad.item(), total_steps)
                        writer.add_histogram(f'Gradients_Hist/{name}', parameter.grad.cpu().numpy(), total_steps)
                    
                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        writer.add_scalar(f'Weights/{name}', avg_weight.item(), total_steps)
                        writer.add_histogram(f'Weights_Hist/{name}', parameter.data.cpu().numpy(), total_steps)
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_train_loss = running_loss / epoch_steps
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Log epoch metrics
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Val_Epoch', val_accuracy, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'alexnet_epoch_{epoch+1}.pkl')
        save_checkpoint(model, optimizer, scheduler, epoch, total_steps, config['training']['seed'], checkpoint_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'alexnet_final.pkl')
    save_checkpoint(model, optimizer, scheduler, num_epochs-1, total_steps, config['training']['seed'], final_checkpoint_path)
    
    writer.close()
    print("Training completed!")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR-10')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found!")
        return
    
    # Start training
    train_model(config, args.resume)


if __name__ == '__main__':
    main()

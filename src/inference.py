"""
Inference Pipeline for AlexNet
Production-ready inference script with CLI interface
Ready for Gradio integration later
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from model import AlexNet
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config_utils.utils import load_config, setup_device
from data_loader import get_class_names


def load_model(checkpoint_path, config, device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        config (dict): Configuration dictionary
        device: Device to load model on
        
    Returns:
        model: Loaded PyTorch model
    """
    # Create model
    model = AlexNet(num_classes=config['model']['num_classes']).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    
    return model


def preprocess_image(image_path, config):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to image file
        config (dict): Configuration dictionary
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define transform (same as test transform in training)
    image_dim = config['model']['image_dim']
    transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict_single_image(model, image_path, config, device):
    """
    Predict class for a single image.
    
    Args:
        model: Trained PyTorch model
        image_path (str): Path to image file
        config (dict): Configuration dictionary
        device: Device to run inference on
        
    Returns:
        dict: Prediction results
    """
    # Preprocess image
    image_tensor = preprocess_image(image_path, config).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class names
    class_names = get_class_names()
    
    # Prepare results
    results = {
        'image_path': image_path,
        'predicted_class': class_names[predicted.item()],
        'predicted_index': predicted.item(),
        'confidence': confidence.item() * 100,
        'all_probabilities': {
            class_names[i]: prob.item() * 100 
            for i, prob in enumerate(probabilities[0])
        }
    }
    
    return results


def predict_batch_images(model, image_dir, config, device):
    """
    Predict classes for all images in a directory.
    
    Args:
        model: Trained PyTorch model
        image_dir (str): Path to directory containing images
        config (dict): Configuration dictionary
        device: Device to run inference on
        
    Returns:
        list: List of prediction results for each image
    """
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all image files
    image_files = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename.lower())[1] in supported_extensions:
            image_files.append(os.path.join(image_dir, filename))
    
    if not image_files:
        print(f"No supported image files found in {image_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process...")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        result = predict_single_image(model, image_path, config, device)
        results.append(result)
    
    return results


def print_prediction_results(results, show_all_probs=False):
    """
    Print prediction results in a formatted way.
    
    Args:
        results: Prediction results (dict or list)
        show_all_probs (bool): Whether to show all class probabilities
    """
    if isinstance(results, dict):
        results = [results]  # Convert single result to list
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {os.path.basename(result['image_path'])}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        
        if show_all_probs:
            print("\nAll Probabilities:")
            # Sort by probability
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                print(f"  {class_name}: {prob:.2f}%")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Run inference with trained AlexNet model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image file')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Path to directory containing images')
    parser.add_argument('--config', type=str, default='config_utils/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--show-all-probs', action='store_true',
                       help='Show probabilities for all classes')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        print("Error: Must specify either --image or --image-dir")
        return
    
    if args.image and args.image_dir:
        print("Error: Cannot specify both --image and --image-dir")
        return
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found!")
        return
    
    # Setup device
    device = setup_device(config)
    
    # Load model
    try:
        model = load_model(args.checkpoint, config, device)
    except FileNotFoundError:
        print(f"Checkpoint file {args.checkpoint} not found!")
        return
    
    # Run inference
    if args.image:
        # Single image inference
        if not os.path.exists(args.image):
            print(f"Image file {args.image} not found!")
            return
        
        print(f"Running inference on single image: {args.image}")
        results = predict_single_image(model, args.image, config, device)
    else:
        # Batch inference
        if not os.path.exists(args.image_dir):
            print(f"Image directory {args.image_dir} not found!")
            return
        
        print(f"Running batch inference on directory: {args.image_dir}")
        results = predict_batch_images(model, args.image_dir, config, device)
    
    # Print results
    print_prediction_results(results, args.show_all_probs)
    
    print(f"\nInference completed!")


if __name__ == '__main__':
    main()

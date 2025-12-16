"""
Inference Pipeline for AlexNet
Production-ready inference script with CLI interface
Ready for Gradio integration later
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from src.model import AlexNet
from config_utils import load_config, setup_device
from src.data_loader import get_class_names, get_data_loaders
from src.conformal import ConformalPredictor


def load_model(checkpoint_path, config, device):
    """
    Load trained model from checkpoint.
    """
    model = AlexNet(num_classes=config['model']['num_classes']).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")

    return model


def preprocess_image(image_path, config):
    """
    Preprocess a single image for inference.
    """
    image_dim = config['model']['image_dim']
    transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict_single_image(model, image_path, config, device):
    """
    Predict class for a single image (top-1).
    """
    image_tensor = preprocess_image(image_path, config).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = get_class_names()

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


def predict_single_image_conformal(model, image_path, config, device, conformal_path):
    """
    Predict using conformal prediction set (requires saved conformal.pt).
    Returns same dict but adds 'conformal_set' and 'conformal_set_size'.
    """
    conf = torch.load(conformal_path, map_location="cpu")
    cp = ConformalPredictor(alpha=conf["alpha"], device=device)
    cp.qhat = conf["qhat"]

    image_tensor = preprocess_image(image_path, config).to(device)

    pred_sets, probs = cp.predict(model, image_tensor)  # pred_sets: [1, K]
    pred_idxs = torch.where(pred_sets[0])[0].tolist()

    class_names = get_class_names()
    conformal_labels = [class_names[i] for i in pred_idxs]

    top_idx = int(torch.argmax(probs[0]).item())
    top_conf = float(probs[0, top_idx].item()) * 100

    results = {
        'image_path': image_path,
        'predicted_class': class_names[top_idx],
        'predicted_index': top_idx,
        'confidence': top_conf,
        'conformal_set': conformal_labels,
        'conformal_set_size': len(conformal_labels),
        'conformal_alpha': float(conf["alpha"]),
        'conformal_qhat': float(conf["qhat"]),
        'all_probabilities': {
            class_names[i]: float(probs[0, i].item()) * 100
            for i in range(probs.shape[1])
        }
    }

    return results


def predict_batch_images(model, image_dir, config, device, use_conformal=False, conformal_path=None):
    """
    Predict classes for all images in a directory (recursively).
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # ✅ Recursive scan (handles ImageFolder-style test/class_name/*.jpg)
    image_files = []
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if os.path.splitext(filename.lower())[1] in supported_extensions:
                image_files.append(os.path.join(root, filename))

    if not image_files:
        print(f"No supported image files found in {image_dir}")
        return []

    print(f"Found {len(image_files)} images to process...")

    results = []
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        if use_conformal:
            result = predict_single_image_conformal(model, image_path, config, device, conformal_path)
        else:
            result = predict_single_image(model, image_path, config, device)
        results.append(result)

    return results


def print_prediction_results(results, show_all_probs=False):
    """
    Print prediction results in a formatted way.
    """
    if isinstance(results, dict):
        results = [results]

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {os.path.basename(result['image_path'])}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2f}%")

        if "conformal_set" in result:
            print(f"Conformal Set (alpha={result['conformal_alpha']}): {result['conformal_set']}")
            print(f"Set Size: {result['conformal_set_size']}")

        if show_all_probs:
            print("\nAll Probabilities:")
            sorted_probs = sorted(result['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                print(f"  {class_name}: {prob:.2f}%")


def calibrate_conformal(config, checkpoint_path, device, alpha=0.1, conformal_out="conformal.pt"):
    """
    Compute qhat using the calib_loader and save conformal file.
    """
    model = load_model(checkpoint_path, config, device)

    _, calib_loader, _ = get_data_loaders(config)

    cp = ConformalPredictor(alpha=alpha, device=device)
    qhat = cp.calibrate(model, calib_loader)

    os.makedirs(os.path.dirname(conformal_out) or ".", exist_ok=True)
    torch.save({"qhat": qhat, "alpha": alpha}, conformal_out)

    print("✅ Conformal calibration completed")
    print(f"   alpha = {alpha}")
    print(f"   qhat  = {qhat:.6f}")
    print(f"   saved = {conformal_out}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='Run inference with trained AlexNet model')

    parser.add_argument('--checkpoint', type=str, required=False,
                        help='Path to model checkpoint file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image file')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Path to directory containing images')
    parser.add_argument('--config', type=str, default='config_utils/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--show-all-probs', action='store_true',
                        help='Show probabilities for all classes')

    parser.add_argument('--conformal', action='store_true',
                        help='Use conformal prediction set (requires --conformal-path)')
    parser.add_argument('--conformal-path', type=str, default='alexnet_data_out/models/conformal.pt',
                        help='Path to conformal file containing qhat')

    parser.add_argument('--calibrate', action='store_true',
                        help='Compute qhat on calib set and save conformal file (no image needed)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Miscoverage level alpha for conformal (default 0.1)')
    parser.add_argument('--conformal-out', type=str, default='alexnet_data_out/models/conformal.pt',
                        help='Where to save conformal qhat file')

    # ✅ Debug: print class order without checkpoint
    parser.add_argument('--print-classes', action='store_true',
                        help='Print ImageFolder class order for train/calib/test and exit')

    parser.add_argument("--dry-run", action="store_true",
                        help="Run inference setup without loading model or data")

    args = parser.parse_args()

    # ✅ If user runs with no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.dry_run:
        print("🧪 DRY RUN MODE")
        print("✓ Argument parsing works")
        print("✓ Inference module imports correctly")
        print("✓ No checkpoint or image required")
        return

    # Load configuration (needed for print-classes too)
    try:
        config = load_config(args.config)
        print(f"Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found!")
        return

    # ✅ Print classes mode (no checkpoint needed)
    if args.print_classes:
        train_loader, calib_loader, test_loader = get_data_loaders(config)
        print("TRAIN classes:", train_loader.dataset.classes)
        print("CALIB classes:", calib_loader.dataset.classes)
        print("TEST  classes:", test_loader.dataset.classes)
        return

    device = setup_device(config)

    # ✅ checkpoint required only when actually doing calib/inference
    needs_checkpoint = args.calibrate or args.image or args.image_dir
    if needs_checkpoint and not args.checkpoint:
        raise ValueError("--checkpoint is required for --calibrate or inference (--image/--image-dir).")

    if args.calibrate:
        calibrate_conformal(
            config=config,
            checkpoint_path=args.checkpoint,
            device=device,
            alpha=args.alpha,
            conformal_out=args.conformal_out
        )
        return

    # Validate args for prediction
    if not args.image and not args.image_dir:
        print("Error: Must specify either --image or --image-dir (or use --calibrate)")
        parser.print_help()
        return

    if args.image and args.image_dir:
        print("Error: Cannot specify both --image and --image-dir")
        return

    try:
        model = load_model(args.checkpoint, config, device)
    except FileNotFoundError:
        print(f"Checkpoint file {args.checkpoint} not found!")
        return

    if args.conformal and not os.path.exists(args.conformal_path):
        print(f"Conformal file not found: {args.conformal_path}")
        print("Run calibration first:")
        print(f"  python -m src.inference --calibrate --checkpoint {args.checkpoint} --conformal-out {args.conformal_path}")
        return

    # Run inference
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file {args.image} not found!")
            return

        print(f"Running inference on single image: {args.image}")
        if args.conformal:
            results = predict_single_image_conformal(model, args.image, config, device, args.conformal_path)
        else:
            results = predict_single_image(model, args.image, config, device)

    else:
        if not os.path.exists(args.image_dir):
            print(f"Image directory {args.image_dir} not found!")
            return

        print(f"Running batch inference on directory: {args.image_dir}")
        results = predict_batch_images(
            model, args.image_dir, config, device,
            use_conformal=args.conformal,
            conformal_path=args.conformal_path
        )

    print_prediction_results(results, args.show_all_probs)
    print("\nInference completed!")
    print("✅ inference.py ran successfully")


if __name__ == '__main__':
    main()

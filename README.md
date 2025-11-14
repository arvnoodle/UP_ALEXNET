# AlexNet Implementation

Production-ready implementation of AlexNet for CIFAR-10 classification.

## Project Structure

```
UP_ALEXNET/
├── README.md
├── requirements.txt
├── app/
├── config_utils/
│   ├── config.yaml
│   └── utils.py
├── notebook/
│   └── AlexNetv3.ipynb
├── src/
│   ├── data_loader.py
│   ├── inference.py
│   ├── model.py
│   └── train.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python src/train.py --config config_utils/config.yaml
```

To resume training from a checkpoint:

```bash
python src/train.py --config config_utils/config.yaml --resume path/to/checkpoint.pkl
```

## Inference

To run inference on a single image:

```bash
python src/inference.py --checkpoint path/to/checkpoint.pkl --image path/to/image.jpg
```

To run inference on a directory of images:

```bash
python src/inference.py --checkpoint path/to/checkpoint.pkl --image-dir path/to/images/
```

To show all class probabilities:

```bash
python src/inference.py --checkpoint path/to/checkpoint.pkl --image path/to/image.jpg --show-all-probs
```

## Configuration

The configuration is stored in `config_utils/config.yaml` and includes:

- Model settings (number of classes, image dimensions)
- Training hyperparameters (epochs, batch size, learning rate)
- Data settings (dataset, workers)
- Paths for output and checkpoints
- Device settings (CUDA, GPU IDs)

## Model Architecture

The AlexNet implementation includes:
- 5 convolutional layers with pooling
- 3 fully connected layers with dropout
- Local response normalization

## CIFAR-10 Classes

The model classifies images into 10 categories:
- airplane, automobile, bird, cat, deer
- dog, frog, horse, ship, truck

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

## Virtual Environment Setup

### Windows

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Installation

After activating your virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

## Training

To train the model (the config path is set by default):

```bash
python src/train.py
```

Or explicitly specify the config path:

```bash
python src/train.py --config config_utils/config.yaml
```

To resume training from a checkpoint:

```bash
python src/train.py --config config_utils/config.yaml --resume path/to/checkpoint.pkl
```

## Inference

To run inference on a single image (default config is config_utils/config.yaml):

```bash
python src/inference.py --checkpoint path/to/checkpoint.pkl --image path/to/image.jpg
```

For example, if you have a dog image in a sample_images directory (default config is used):

```bash
python src/inference.py --checkpoint alexnet_data_out/models/alexnet_final.pkl --image sample_images/my_dog.jpg
```

To run inference on a directory of images (e.g., 10 dog images) (default config is used):

```bash
python src/inference.py --checkpoint path/to/checkpoint.pkl --image-dir path/to/images/
```

To show all class probabilities (default config is used):

```bash
python src/inference.py --checkpoint path/to/checkpoint.pkl --image path/to/image.jpg --show-all-probs
```

## Monitoring Training with TensorBoard

To monitor training progress with TensorBoard:

```bash
tensorboard --logdir alexnet_data_out/tblogs
```

For remote i.e Runpod

```bash
tensorboard --logdir alexnet_data_out/tblogs --host 0.0.0.0 --port 6006
```

Then access TensorBoard from your local browser at:
```
http://[ADDRESS]:6006
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

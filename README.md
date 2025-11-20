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

## Running on Google Colab

To run this project on Google Colab without any local setup (Use GPU):

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook or upload an existing one
3. In a cell, run the following commands to set up the project:


```python
!git clone https://github.com/arvnoodle/UP_ALEXNET.git
cd UP_ALEXNET/
!pip install -r requirements.txt
!python src/train.py --config config_utils/config.yaml
```

4. To run inference on a single image after training (or using a pre-trained model):


```python
!python src/inference.py --checkpoint alexnet_data_out/models/alexnet_final.pkl --image path/to/your/image.jpg
```

5. To run inference on a directory of images:


```python
!python src/inference.py --checkpoint alexnet_data_out/models/alexnet_final.pkl --image-dir path/to/your/images/
```

6. To monitor training with TensorBoard in Colab:


```python
%load_ext tensorboard
%tensorboard --logdir alexnet_data_out/tblogs
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

After activating your virtual environment, install the required packages to reproduce an AlexNet architecture in PyTorch. Conceptually, this corresponds to setting up the **experimental environment** described in *ImageNet Classification with Deep Convolutional Neural Networks* (Krizhevsky et al., 2012): we prepare the tools that will let us define a deep convolutional network and train it efficiently on a large image dataset (here, CIFAR‑10 instead of ImageNet).

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
- Momentum and weight decay (L2 regularization)
- Image and label configuration (AlexNet expects 227×227 RGB inputs matching the effective input size used in the 2012 paper even though the paper text mentions 224×224, the dimensions after the first convolution are consistent with 227×227).
- Data settings (dataset, workers)
- Paths for output and checkpoints
    * `INPUT_ROOT_DIR`, `TRAIN_IMG_DIR` – where the training images live.
    * `OUTPUT_DIR`, `LOG_DIR`, `CHECKPOINT_DIR` – where TensorBoard logs and model checkpoints are saved.
- Device settings (CUDA, GPU IDs)

## Model Architecture

The AlexNet implementation includes:
- 5 convolutional layers with pooling:
  1. **Conv1:** - `nn.Conv2d(3 → 96, kernel_size=11, stride=4)` followed by `ReLU`, `LocalResponseNorm`, and `MaxPool2d(3×3, stride=2)`. This is a large receptive field that strides quickly and is expected to reduce spatial resolution (227×227 → 55×55) and capture low‑level edges and color blobs, as described in the paper.
  2. **Conv2:** - `nn.Conv2d(96 → 256, kernel_size=5, padding=2)` + `ReLU`, `LocalResponseNorm`, `MaxPool2d`. Builds more complex edge and texture features. L
  3. **Conv3–Conv5:** - Three stacked `3×3` convolutions (`256 → 384 → 384 → 256`) with `ReLU` activations and a final `MaxPool2d` that reduces the spatial dimensions to 6×6. These deeper layers capture **high‑level object parts and shapes**, a key idea in convolutional neural networks and discussed in the "Using Pre‑Trained Models" lecture as the **feature extraction** part of a CNN.

The final output of `self.features` has shape **(batch_size, 256, 6, 6)**, which is flattened to `256 * 6 * 6` and passed into the classifier.

- 3 fully connected layers with dropout:
  1. `Dropout(p=0.5)` → `Linear(256*6*6 → 4096)` → `ReLU`
  2.  `Dropout(p=0.5)` → `Linear(4096 → 4096)` → `ReLU`
  3. `Linear(4096 → num_classes)`
The **two 4096‑unit hidden layers** are exactly as in the paper. Dropout** is applied before each large FC layer, as in AlexNet, to combat overfitting by randomly zeroing activations during training. The final linear layer maps to `num_classes` (10 for CIFAR‑10, 1000 in the original ImageNet setting).
- Local response normalization was used in the original AlexNet to encourage competition between nearby feature maps.

-  Forward Pass (`forward` method). The input `x` passes through `self.features` (the convolutional backbone), is flattened, and then passed through `self.classifier`. This corresponds to the **end‑to‑end feed‑forward computation** described in the paper, where convolutions learn visual features and fully connected layers perform final category discrimination. Conceptually, this matches the **feature extractor + classifier** decomposition emphasized in the pre‑trained CNN models lecture: the convolutional stack learns reusable image features, while the final FC layers are more task‑specific.


## Data Pipeline

1. **Transforms (`train_transform`, `test_transform`)**
   * `transforms.Resize((227, 227))` – CIFAR‑10 images are originally **32×32**. We upsample them to **227×227** to match AlexNet’s expected input size.
   * `transforms.RandomHorizontalFlip()` (train only) – simple **data augmentation** that creates left–right mirrored versions of images, improving generalization. This is similar in spirit to the random data augmentations used in the AlexNet paper.
   * `transforms.ToTensor()` – converts PIL images into PyTorch tensors with channels‑first shape `(C, H, W)` and pixel values in `[0, 1]`.
   * `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` – subtracts the ImageNet channel means and divides by their standard deviations. These statistics are standard for **ImageNet‑pretrained models** and match the preprocessing used in many modern CNN libraries.

   Even though we are not loading a pre‑trained model here, using the same normalization scheme makes the inputs comparable to the **ImageNet setting** described in the lecture on pre‑trained models.

2. **Datasets and DataLoaders**
   * `train_dataset` and `test_dataset` wrap the CIFAR‑10 image folders (or torchvision’s built‑in CIFAR‑10) with the transforms defined above.
   * `DataLoader` objects (`train_loader` and `test_loader`) handle:
     * Shuffling (for stochastic gradient descent).
     * Mini‑batching (`batch_size = BATCH_SIZE`).
     * Parallel loading using `num_workers`.
     * Pinned memory (`pin_memory=True`) to speed up host‑to‑GPU transfers.

This **data pipeline** mirrors the one in the AlexNet paper: images are resized, normalized, optionally augmented, and then fed in **mini‑batches** to the GPU. In the lecture slides, this is the “input → preprocessing → batch loading” stage that precedes the forward and backward passes.

The model classifies images into 10 categories:
- airplane, automobile, bird, cat, deer
- dog, frog, horse, ship, truck

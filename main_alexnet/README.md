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
|   |── conformal.py
|   |── eval_conformal.py
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
!python -m src.inference --checkpoint alexnet_data_out\models\alexnet_final.pkl --image-dir data\test
```

Conformal batch inference
```python
!python -m src.inference --checkpoint alexnet_data_out\models\alexnet_final.pkl --image-dir data\test --conformal
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
python -m src.inference --checkpoint alexnet_data_out/models/alexnet_final.pkl --image-dir data/test
```

To show all class probabilities (default config is used):

```bash
python -m src.inference --checkpoint alexnet_data_out/models/alexnet_final.pkl --image-dir data/test --show-all-probs
```

To show all conformal predictions:

```bash
python -m src.inference --checkpoint alexnet_data_out/models/alexnet_final.pkl --image-dir data/test --conformal

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
- **BatchNorm2d** replacing LocalResponseNorm for improved training stability
- **Adaptive pooling** to support flexible input resolutions

## Anime Dataset

This implementation has been adapted for anime style classification.

### Dataset Structure
The model classifies anime into 3 categories:
- **3D anime** (modern CGI/3D animation)
- **90s anime** (classic 90s animation style)  
- **modern anime** (contemporary 2D animation)

### Download Dataset

**Training Dataset** (organized in train/3d, train/90s, train/modern folders):
- [Download Training Data](https://drive.google.com/file/d/1Ig13vLTCmO4stzLcLR5FbC75v1MvUh3Y/view?usp=sharing)

**Calibration Dataset** (organized in train/3d, train/90s, train/modern folders):
- [Download Calibration Data](https://drive.google.com/file/d/1Ig13vLTCmO4stzLcLR5FbC75v1MvUh3Y/view?usp=sharing)

**Batch Inference Dataset** (mixed images for testing):
- [Download Inference Data](https://drive.google.com/file/d/1l3uUvK1AKEoptMlMd4s5e9E0MRoNkDj2/view?usp=sharing)

Extract the training dataset to `./data/` directory before training.

### Result

#### Test Accuracy and Convergence Behavior

The test accuracy curve showed rapid early learning followed by a stable convergence as reflected in the accuracy metrics rising rapidly from 45–50% to ~70% within the first 5 epochs. Gradual improvement continued until epoch 10–12, then marginal performance was finally achieved at around 80–85%. This pattern is consistent with the convergence behavior of deep convolutional networks such as AlexNet, which typically exhibit fast early learning followed by slower refinement as discriminative features stabilize (Krizhevsky et al., 2012). This indicates that the model learned meaningful features with little evidence of overfitting.


#### Conformal Prediction Evaluation

| Metric                      | Value          |
| --------------------------- | -------------- |
| Target Miscoverage (α)      | 0.1            |
| Target Coverage             | 0.900          |
| q̂ (Conformal Threshold)     | 0.910015       |
| Number of Test Samples      | 175            |
| Overall Coverage            | 0.966          |
| 95% Coverage CI             | [0.927, 0.984] |
| Average Prediction Set Size | 1.354          |


Beyond conventional accuracy metrics, we evaluated the trained AlexNet classifier using conformal prediction, chosen for its model‑agnostic framework and its ability to provide distribution‑free uncertainty quantification (Angelopoulos & Bates, 2023). Evaluation was conducted on 175 test images with three style classes (3d_anime, modern_anime, 90s_anime) and reported (a) the top‑1 softmax prediction with its confidence and (b) a conformal prediction set at a nominal α = 0.1, which targets 90% marginal coverage. Under the assumption of exchangeability, the true label should lie within the returned prediction set at least 90% of the time (Angelopoulos & Bates, 2023). The model achieved an empirical coverage of 96.6% across 175 samples, consistent with the tendency of conformal predictors to be conservative rather than under‑covering (Karimi & Samavi, 2023).

#### Conformal Set Size and Efficiency

Across all test images (n = 175), conformal prediction yielded compact and efficient prediction sets. The average set size was 1.354, with 64.6% of samples (n = 113) producing singleton sets, while the remaining 35.4% (n = 62) produced two‑label sets. No prediction sets larger than size two were observed. This aligns with theoretical expectations that well‑calibrated conformal predictors produce small sets when model confidence is high and expand only when uncertainty is present(Angelopoulos & Bates, 2023)(Karimi & Samavi, 2023). These results indicate that for most inputs, the model’s predictions were sufficiently well separated to justify a single‑label conformal set

#### Class-Specific Conformal Behavior

Conformal behavior differed significantly across predicted classes. The 90s_anime class had the highest average confidence (96.9%) and the tightest conformal sets, with an average set size of 1.119 and 88.1% singleton sets. This suggests that 90s_anime possesses the most distinctive and consistently recognized visual cues within the learned representation. In contrast, 3d_anime and modern_anime displayed more frequent ambiguity, each with an average set size of 1.429 and two‑label sets occurring in 42.9% of cases. This pattern is consistent with the idea that conformal prediction naturally expresses uncertainty through multi‑label sets when classes overlap visually(Karimi & Samavi, 2023). The framework effectively captures this ambiguity by returning {3d_anime, modern_anime} for borderline samples rather than forcing an unreliable point prediction.

### References
- Angelopoulos, A. N., & Bates, S. (2023). A Gentle Introduction to Conformal Prediction and Distribution‑Free Uncertainty Quantification.
- Karimi, H., & Samavi, R. (2023). Quantifying Deep Learning Model Uncertainty in Conformal Prediction.
- Krizhevsky, A., Sutskever, I., & Hinton, G. (2012). ImageNet Classification with Deep Convolutional Neural Networks.

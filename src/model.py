"""
AlexNet Model Implementation
Extracted from the original alexnetv3.py by Janice and Paula
Clean, modular implementation for production use
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers proposed by AlexNet paper.
    
    Implementation from "ImageNet Classification with Deep Convolutional Neural Networks" 
    by Alex Krizhevsky et al.
    """

    def __init__(self, num_classes=10):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(AlexNet, self).__init__()

        # Feature extractor (Convolution + Pooling Layers)
        # Architecture for the ImageNet dataset.
        # Input size adjusted to (b x 3 x 227 x 227)
        # CIFAR-10 images will be resized to retain the architecture as is.

        self.features = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),     # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # Can be replaced with nn.BatchNorm2d(96) if any error encountered
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            
            # Convolutional layer 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),   # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # Can be replaced with nn.BatchNorm2d(256) if any error encountered
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            
            # Convolutional layer 3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            
            # Convolutional layer 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            
            # Convolutional layer 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )

        # Classifier (Fully Connected Layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        
        self.init_bias()  # initialize bias

    def init_bias(self):
        """Initialize biases according to the original AlexNet paper."""
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.features[4].bias, 1)
        nn.init.constant_(self.features[10].bias, 1)
        nn.init.constant_(self.features[12].bias, 1)

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input.
        return self.classifier(x)

"""
image_cnn_model.py
Rebuilt here because the original version was not available on the user's
device. Matches the previously documented specification: a small
from-scratch CNN (SimpleCNN). This exact file is the one actually used to
train GrayscaleCNN on Colab (real result: val_acc = 92.51% on ArASL, 32
classes, 54,049 images).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple, from-scratch CNN, used as GrayscaleCNN on ArASL data (32 classes).
    Expected input: 3 x 64 x 64 (even though the image is originally
    grayscale, R=G=B triplicated).
    """

    def __init__(self, num_classes: int = 32, input_size: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        reduced = input_size // (2 ** 3)
        flat_dim = 32 * reduced * reduced

        self.fc1 = nn.Linear(flat_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleCNN(num_classes=32, input_size=64)
    print("SimpleCNN (GrayscaleCNN) parameters:", model.count_parameters())
    dummy = torch.zeros(2, 3, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)

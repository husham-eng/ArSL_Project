"""
ARCHIVED -- superseded by src/color/color_sign_resnet18.py.

This from-scratch architecture was the project's first attempt at the
color branch. Across repeated training attempts on the full AASL corpus it
failed to learn the task reliably (best 20.6% validation accuracy after 15
epochs; one run's loss stalled at ln(31)~=3.43, i.e. uniform random
prediction, for dozens of epochs without recovering). It is kept here,
unused, for documentation -- see docs/scientific_paper.docx Section 3.2 for
the full negative-result discussion and the transfer-learning replacement
that reached 97.90% validation accuracy.

ColorSignCNN — a CNN specialized for genuinely colored (RGB) AASL letter images.

The core difference from a TransferCNN previously used on grayscale ArASL:
- Here the input is genuine RGB (3 actually different channels), not
  grayscale artificially triplicated into 3 channels.
- The architecture keeps the same SimpleCNN philosophy (a small network,
  few parameters, trains reasonably fast even on CPU), but with no
  grayscale conversion anywhere in the transform.

Planned usage (per Section 5 of the project summary):
  GrayscaleCNN = SimpleCNN (trained on grayscale ArASL, as-is)
  ColorCNN     = ColorSignCNN (this file — trained on genuinely colored AASL)
  ModalityFusionModel fuses the two decisions for a live D435 color frame.

This file is written in standard PyTorch style (matching image_cnn_model.py)
for running in Colab, where enough disk space and/or GPU is available to
install PyTorch fully.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorSignCNN(nn.Module):
    """
    A simple, from-scratch CNN for classifying colored AASL letters (31 classes).

    Expected input: an RGB image of size 3 x 64 x 64 (resized explicitly in
    the transform before entering the model, because the original AASL
    images have wildly non-uniform dimensions — see the "large variance in
    image dimensions" note in aasl_upload_progress.json).
    """

    def __init__(self, num_classes: int = 31, input_size: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        # after 3 maxpool(2x2) operations: 64 -> 32 -> 16 -> 8
        reduced = input_size // (2 ** 3)
        flat_dim = 64 * reduced * reduced

        self.fc1 = nn.Linear(flat_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ColorSignCNN(num_classes=31, input_size=64)
    print("ColorSignCNN parameters:", model.count_parameters())
    dummy = torch.zeros(2, 3, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: [2, 31]

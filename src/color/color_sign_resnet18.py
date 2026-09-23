"""
color_sign_resnet18.py — a Transfer Learning model for genuinely colored
(RGB) AASL letter images, built on a Pretrained ResNet18 (ImageNet weights)
instead of a from-scratch CNN.

WHY THIS EXISTS (rationale, per the paper's revised narrative)
-----------------------------------------------------------------
The from-scratch ColorSignCNN (color_sign_cnn.py) struggled to learn AASL's
colored images effectively (val_acc plateaued/collapsed well below 30%
across multiple real training runs), while the grayscale ArASL branch
(GrayscaleCNN / SimpleCNN, image_cnn_model.py) reached ~92% from scratch.
This asymmetry is not a coincidence: grayscale hand-sign images are
visually simpler (less texture/background/lighting variation to
disentangle), so a small from-scratch CNN can learn them well with a
modest amount of data. Colored real-world images carry far more visual
variation (lighting, skin tone, background clutter, camera differences --
AASL was collected from 200+ participants with different cameras), which
from-scratch small CNNs are known to struggle with on datasets this size.

The literature on this exact dataset (ArASL/AASL, 7,856 RGB images, 31
classes) confirms this: transfer-learning approaches using ResNet18,
MobileNetV2 and VGG-16 all report 96-97% accuracy on this dataset via
fine-tuning of ImageNet-pretrained weights, dramatically outperforming
from-scratch CNNs of comparable size to ColorSignCNN.

This module therefore replaces ColorSignCNN as the "color" branch, using
transfer learning (ResNet18 pretrained on ImageNet, with its final fully
connected layer replaced and fine-tuned for AASL's 31 classes). This is
the standard, well-established fix for exactly this failure mode, and lets
the paper's fusion claim be tested honestly between two architectures each
suited to its own modality -- rather than between a strong grayscale
model and a badly underperforming from-scratch color model.

Two training modes are supported (see train_color_resnet18.py --freeze_backbone):
  - Feature extraction (backbone frozen): only the new final layer trains.
    Fastest, least prone to overfitting on a modest dataset, a reasonable
    first attempt.
  - Fine-tuning (backbone unfrozen): the whole network trains at a small
    learning rate. Usually higher final accuracy, matches what most of the
    cited literature actually did, but costs more compute per epoch.
"""

import torch
import torch.nn as nn
from torchvision import models


class ColorSignResNet18(nn.Module):
    """
    Wraps torchvision's ResNet18 (ImageNet-pretrained by default) with its
    final fully-connected layer replaced for AASL's 31 classes.

    Expected input: RGB images resized to 224x224 (ResNet18's native
    ImageNet input size) and normalized with ImageNet mean/std -- see
    build_transforms() in train_color_resnet18.py. This is a deliberate
    difference from ColorSignCNN's 64x64 input: using ResNet18 away from
    its native resolution/normalization would waste most of its pretrained
    feature quality.
    """

    def __init__(self, num_classes: int = 31, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Replace the final classification layer (originally 1000-way for
        # ImageNet) with a fresh, trainable layer for AASL's 31 classes.
        # This new layer's parameters are always trainable, even in
        # feature-extraction mode -- otherwise nothing could learn at all.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = ColorSignResNet18(num_classes=31, pretrained=True, freeze_backbone=False)
    print("Trainable parameters:", model.count_parameters())
    print("Total parameters:", model.count_total_parameters())
    dummy = torch.zeros(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected: [2, 31]

"""
ARCHIVED -- superseded by src/color/train_color_resnet18.py, which fine-tunes
ColorSignResNet18 (ImageNet-pretrained ResNet-18) and reached 97.90%
validation accuracy. This script trains ColorSignCNN, the from-scratch
architecture that failed to learn AASL reliably (see color_sign_cnn.py and
docs/scientific_paper.docx Section 3.2). Kept for documentation only.

train_color_cnn.py — trains ColorSignCNN on genuinely colored AASL data.

⚠️ Important runtime note:
This file is written to run in an environment with full PyTorch available
(such as Colab, as in the current sign_language_training_colab.ipynb). In
the sandbox environment used for inspecting and uploading the data, PyTorch
could not be installed because the required CUDA packages (nvidia-cublas,
nvidia-cudnn, ...) exceed the available disk space (~5GB) even for CPU-only
use, since modern PyPI torch wheels on Linux pull these dependencies in
automatically. The pipeline's correctness (data loading + forward/backward +
genuine learning) was therefore actually validated via an equivalent NumPy
version (numpy_color_cnn_train.py), with its results in
color_cnn_training_report.json. This file is the "official" version meant
to be run in Colab.

Usage in Colab:
    python train_color_cnn.py --data_dir /content/AASL --epochs 15 --batch_size 32
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from color_sign_cnn import ColorSignCNN

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(input_size: int = 64):
    # No conversion to grayscale here on purpose — this is the core difference from the ArASL path
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder containing 31 subfolders, one per letter class (AASL structure as uploaded)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="color_sign_cnn_best.pt")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, val_tf = build_transforms(args.input_size)

    # ImageFolder expects the structure: data_dir/<class_name>/*.jpg — exactly matching AASL's structure after extraction
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images across {num_classes} classes")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_ds.dataset.transform = val_tf  # validation set gets no augmentation

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ColorSignCNN(num_classes=num_classes, input_size=args.input_size).to(device)
    print("Trainable parameters:", model.count_parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss/train_total:.4f} "
              f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "input_size": args.input_size,
                "val_acc": val_acc,
            }, args.out)
            print(f"  -> saved new best model (val_acc={val_acc:.4f}) to {args.out}")

    print(f"\nDone. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

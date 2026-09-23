"""
train_grayscale_cnn.py
Real GPU training script for GrayscaleCNN (SimpleCNN) on the ArASL corpus.
Actually run on Google Colab (T4 GPU): 54,049 images, 32 classes, 15 epochs,
best val_acc = 92.51% at epoch 14. See docs/scientific_paper.docx Section 4.3
for the full per-epoch results table and discussion.

Usage:
    python train_grayscale_cnn.py --data_dir /path/to/ArASL --epochs 15 --batch_size 32
"""

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from image_cnn_model import SimpleCNN

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder containing one subfolder per letter class (ArASL structure)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out", type=str, default="grayscale_simplecnn_best.pt")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(args.data_dir, transform=tf)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images across {num_classes} classes")
    print("Classes order:", full_dataset.classes)

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=num_classes, input_size=args.input_size).to(device)
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

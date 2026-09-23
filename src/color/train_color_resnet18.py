"""
train_color_resnet18.py — trains ColorSignResNet18 (Transfer Learning) on
genuinely colored AASL data.

WHY THIS REPLACES train_color_cnn.py
--------------------------------------
Multiple real from-scratch ColorSignCNN training runs on this exact AASL
data plateaued around val_acc=0.20 at best, and one run collapsed entirely
(loss frozen at ln(31)~=3.43, the uniform-random-guess baseline, for many
consecutive epochs). This script fine-tunes an ImageNet-pretrained
ResNet18 instead, which the literature reports reaching 96-97% on this
same dataset (7,856 RGB images, 31 classes) -- a completely different
performance regime, because the pretrained backbone already knows general
visual features (edges, textures, shapes) that a from-scratch CNN this
size has to learn from only ~6,000 training images.

Carries forward every stability lesson learned from train_color_cnn.py:
  - explicit device logging (confirms GPU is actually used)
  - reproducible augmentation across DataLoader workers (worker_init_fn)
  - gradient clipping (prevents a single bad batch from collapsing the model)
  - a NaN/Inf guard that fails loudly instead of silently burning epochs
  - a stall detector that stops early if val_acc freezes for several
    epochs in a row (a real collapse, not genuine convergence)
  - the TransformedSubset pattern (train/val subsets never accidentally
    share a mutated .transform)

Usage in Colab:
    python train_color_resnet18.py --data_dir "/content/AASL/RGB ArSL dataset" \
        --epochs 20 --batch_size 32 --freeze_backbone false --out color_sign_resnet18_best.pt

Recommended first attempt: fewer epochs than the from-scratch CNN needed
(20 is usually plenty for fine-tuning a pretrained backbone -- it starts
near a good solution already, unlike training from random weights), and
--freeze_backbone true first, as a fast, low-risk sanity check, before a
full fine-tune with --freeze_backbone false if a higher accuracy is wanted.
"""

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from color_sign_resnet18 import ColorSignResNet18

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    """See train_color_cnn.py for why this matters: without it, each
    DataLoader worker gets an uncontrolled random state, so augmentation
    varies run-to-run even with the same global seed, making results hard
    to compare or reproduce."""
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class TransformedSubset(Dataset):
    """See train_color_cnn.py for the bug this avoids: applying
    random_split directly to one ImageFolder and then reassigning
    .dataset.transform on one subset silently mutates BOTH subsets, since
    they share the same underlying ImageFolder object. This class gives
    each split its own genuinely independent transform."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        base_dataset = self.subset.dataset
        real_idx = self.subset.indices[idx]
        path, label = base_dataset.samples[real_idx]
        image = base_dataset.loader(path)
        return self.transform(image), label


def build_transforms(input_size: int = 224):
    # 224x224 + ImageNet mean/std: ResNet18 was pretrained at this exact
    # resolution/normalization, so matching it lets the pretrained features
    # transfer properly. Augmentation is intentionally milder than
    # ColorSignCNN's (no ColorJitter) since fine-tuning a pretrained
    # network is more sensitive to aggressive augmentation early on.
    train_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.0),  # left off on purpose: hand-sign identity is not flip-invariant
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


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder containing 31 subfolders, one per letter class (AASL structure as uploaded)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Lower than from-scratch training on purpose -- fine-tuning a pretrained "
                              "network with a high LR risks destroying its useful pretrained features.")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--freeze_backbone", type=str2bool, default=False,
                         help="true: only train the new final layer (fast, safe first check). "
                              "false: fine-tune the whole network (usually higher final accuracy).")
    parser.add_argument("--out", type=str, default="color_sign_resnet18_best.pt")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")
    print(f"Mode: {'feature extraction (backbone frozen)' if args.freeze_backbone else 'full fine-tuning'}")

    train_tf, val_tf = build_transforms(args.input_size)

    full_dataset = datasets.ImageFolder(args.data_dir, transform=None)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images across {num_classes} classes")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_ds = TransformedSubset(train_subset, train_tf)
    val_ds = TransformedSubset(val_subset, val_tf)

    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
        worker_init_fn=worker_init_fn, generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    model = ColorSignResNet18(
        num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze_backbone,
    ).to(device)
    print("Trainable parameters:", model.count_parameters(),
          "/ total:", model.count_total_parameters())

    criterion = nn.CrossEntropyLoss()
    # Only pass parameters that actually require grad -- matters when the
    # backbone is frozen, otherwise Adam would track (unused) frozen params.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    stall_counter = 0
    last_val_acc = None
    STALL_PATIENCE = 5
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch} -- "
                    f"training has diverged. Stopping now instead of "
                    f"wasting further epochs/compute on a dead model."
                )

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
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss/train_total:.4f} "
              f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | lr={current_lr:.6f}")
        scheduler.step()

        if last_val_acc is not None and abs(val_acc - last_val_acc) < 1e-6:
            stall_counter += 1
        else:
            stall_counter = 0
        last_val_acc = val_acc

        if stall_counter >= STALL_PATIENCE:
            print(f"\n⚠️  val_acc has not changed for {STALL_PATIENCE} consecutive epochs "
                  f"(stuck at {val_acc:.4f}) -- stopping early to avoid wasting compute. "
                  f"If this happens on a pretrained backbone, try a lower --lr.")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "input_size": args.input_size,
                "val_acc": val_acc,
                "freeze_backbone": args.freeze_backbone,
                "architecture": "resnet18",
            }, args.out)
            print(f"  -> saved new best model (val_acc={val_acc:.4f}) to {args.out}")

    print(f"\nDone. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

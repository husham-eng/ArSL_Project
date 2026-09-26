"""
eval_grayscale_heldout.py
=========================
Re-evaluates the deployed GrayscaleCNN checkpoint (grayscale_simplecnn_best.pt)
on exactly the validation split it was trained with, by reproducing the
data pipeline of src/grayscale/train_grayscale_cnn.py line for line:

    Resize((64, 64)) -> ToTensor()      (no normalization)
    ImageFolder(data_dir)
    val_size = int(0.2 * N); random_split(..., generator=manual_seed(42))
    model.eval(), torch.no_grad()

The checkpoint stores the best validation accuracy it reached during
training (val_acc). If the reproduced split is identical, the accuracy
computed here must match that stored value exactly; the script checks this
for every candidate folder and writes results only for a matching folder.

Outputs (in --out_dir):
    heldout_summary.txt          accuracy, macro/weighted averages, lowest-F1 classes
    heldout_per_class.csv        precision, recall, F1, support for all 32 classes
    heldout_confusion_matrix.png 32-class confusion matrix (counts)

Usage (from the repository root, with the project's Python environment):
    python evaluation/eval_grayscale_heldout.py --data_dir "PATH1" "PATH2"
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("grayscale", "fusion", "letter_to_word"):
    sys.path.insert(0, os.path.join(ROOT, "src", sub))
from image_cnn_model import SimpleCNN  # noqa: E402

SEED = 42  # same as train_grayscale_cnn.py


def letter_names(classes):
    """Arabic letter for each ArASL folder name, if the project mapping is available."""
    try:
        from class_mapping import ARASL_TO_ARABIC
        return [ARASL_TO_ARABIC.get(c, c) for c in classes]
    except Exception:
        return list(classes)


def evaluate(data_dir, ckpt, batch_size):
    tf = transforms.Compose([
        transforms.Resize((ckpt["input_size"], ckpt["input_size"])),
        transforms.ToTensor(),
    ])
    full = datasets.ImageFolder(data_dir, transform=tf)
    info = {"n_images": len(full), "classes_match": full.classes == ckpt["classes"]}
    if not info["classes_match"]:
        return info, None, None

    val_size = int(len(full) * 0.2)
    train_size = len(full) - val_size
    _, val_ds = random_split(full, [train_size, val_size],
                             generator=torch.Generator().manual_seed(SEED))

    model = SimpleCNN(num_classes=len(ckpt["classes"]), input_size=ckpt["input_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # num_workers=0 avoids Windows multiprocessing issues; it does not change results
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            y_pred.extend(model(x).argmax(1).tolist())
            y_true.extend(y.tolist())
            if i % 50 == 0:
                print(f"    batch {i}/{len(loader)}", flush=True)
    info["n_val"] = len(y_true)
    info["val_acc"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return info, np.array(y_true), np.array(y_pred)


def per_class(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return cm, prec, rec, f1, cm.sum(1)


def plot_cm(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        labels = [get_display(arabic_reshaper.reshape(l)) for l in labels]
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", nargs="+", required=True,
                    help="one or more candidate ArASL folders (each containing the 32 class folders)")
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "grayscale_simplecnn_best.pt"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "evaluation", "heldout_results"))
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    stored = ckpt.get("val_acc")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Stored best val_acc: {stored}")

    match = None
    for d in args.data_dir:
        print(f"\n=== Candidate: {d}")
        if not os.path.isdir(d):
            print("    folder not found")
            continue
        info, y_true, y_pred = evaluate(d, ckpt, args.batch_size)
        print(f"    images: {info['n_images']}  |  classes match checkpoint: {info['classes_match']}")
        if y_true is None:
            continue
        same = stored is not None and abs(info["val_acc"] - stored) < 1e-9
        print(f"    val_acc: {info['val_acc']:.6f} on {info['n_val']} images  |  "
              f"matches stored value: {same}")
        if same and match is None:
            match = (d, info, y_true, y_pred)

    if match is None:
        print("\nNo candidate reproduces the stored val_acc exactly. No results were written.")
        print("Send the output above for diagnosis.")
        return

    d, info, y_true, y_pred = match
    classes = ckpt["classes"]
    names = letter_names(classes)
    cm, prec, rec, f1, sup = per_class(y_true, y_pred, len(classes))
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "heldout_per_class.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["folder", "letter", "precision", "recall", "f1", "support"])
        for i, c in enumerate(classes):
            w.writerow([c, names[i], f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])

    wavg = lambda v: float((v * sup).sum() / sup.sum())
    order = np.argsort(f1)
    lines = [
        f"Data folder: {d}",
        f"Held-out validation images: {info['n_val']} (seed {SEED}, identical to training split)",
        f"Accuracy: {info['val_acc']:.4f}  (stored in checkpoint: {stored:.4f})",
        f"Macro avg     precision {prec.mean():.4f}  recall {rec.mean():.4f}  F1 {f1.mean():.4f}",
        f"Weighted avg  precision {wavg(prec):.4f}  recall {wavg(rec):.4f}  F1 {wavg(f1):.4f}",
        "Lowest-F1 classes:",
    ] + [f"  {classes[i]} ({names[i]}): P {prec[i]:.2f}  R {rec[i]:.2f}  F1 {f1[i]:.2f}  n={int(sup[i])}"
         for i in order[:6]]
    with open(os.path.join(args.out_dir, "heldout_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    plot_cm(cm, names, os.path.join(args.out_dir, "heldout_confusion_matrix.png"),
            f"GrayscaleCNN, held-out ArASL validation split (n = {info['n_val']})")

    print("\n" + "\n".join(lines))
    print(f"\nResults written to: {args.out_dir}")


if __name__ == "__main__":
    main()

"""
plot_confusion_matrices.py
============================
Turns the 3 saved .npy confusion matrices from evaluate_real_fusion.py
(grayscale-alone, color-alone, fusion) into readable PNG heatmaps, using
the real Arabic/English class names from fusion_eval_class_order.txt.

Usage (same folder where evaluate_real_fusion.py wrote its outputs):
    python evaluation/plot_confusion_matrices.py --prefix fusion_eval
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot_one(cm: np.ndarray, class_names: list[str], title: str, out_path: str):
    # Row-normalize so each row (true class) sums to 1 -- this is what most
    # papers show, since raw counts are hard to compare across classes with
    # different numbers of validation images.
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.32)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fraction of true-class samples")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="fusion_eval")
    args = parser.parse_args()

    with open(f"{args.prefix}_class_order.txt", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    configs = [
        ("grayscale", "GrayscaleCNN alone (on grayscale derivative of live color frame)"),
        ("color", "ColorSignResNet18 alone (on real AASL color images)"),
        ("fusion", "ModalityFusionModel (average fusion)"),
    ]

    for name, title in configs:
        cm = np.load(f"{args.prefix}_{name}_confusion_matrix.npy")
        plot_one(cm, class_names, title, f"{args.prefix}_{name}_confusion_matrix.png")


if __name__ == "__main__":
    main()

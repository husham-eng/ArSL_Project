"""
evaluate_real_fusion.py
========================
Turns ModalityFusionModel from "syntax-checked against synthetic dummy
tensors" into a real, empirically evaluated result on real images.

WHY THIS IS THE RIGHT EVALUATION DESIGN
----------------------------------------
In actual deployment, a single live color frame from the D435 feeds BOTH
branches: the color frame itself goes to ColorSignResNet18, and a grayscale
version *derived from that same frame* (via rgb_to_grayscale_3ch) goes to
GrayscaleCNN. So the correct way to evaluate the fusion end-to-end is to
take AASL's real color validation images (which have real ground-truth
labels), derive the grayscale input for GrayscaleCNN from each one, and
score all three outputs (GrayscaleCNN-alone, ColorSignResNet18-alone, and
their fusion) against AASL's real labels. This is a genuine empirical test
on real data -- not a synthetic-tensor sanity check -- and it directly
answers the question the paper needs answered: does fusing the two
decisions help compared to either network alone?

CLASS-SPACE ALIGNMENT
----------------------
AASL has 31 classes; ArASL (grayscale) has 32. The one ArASL class with no
AASL counterpart is "yaa" (ى, alef maksura) -- AASL does not distinguish it
from plain "ya" (ي). The fused/unified class space used here is AASL's own
31-class space (since AASL supplies the ground truth for this evaluation);
GrayscaleCNN's "yaa" predictions are simply outside that space and are
remapped to zero probability rather than guessed into some other class.
This mapping is built here explicitly from each folder name's real Arabic
letter (via ARASL_TO_ARABIC / AASL_TO_ARABIC below), not by position, so it
is correct regardless of either dataset's on-disk sort order.

Usage (from the repo root, same environment where both checkpoints and both
datasets are available -- e.g. the Colab session used for ColorSignResNet18
training):
    python evaluation/evaluate_real_fusion.py \
        --aasl_dir /content/AASL \
        --grayscale_checkpoint grayscale_simplecnn_best.pt \
        --color_checkpoint color_sign_resnet18_best.pt

Real results obtained with this script (1,571 held-out AASL images, see
docs/scientific_paper.docx Section 4.7 for full discussion):
    grayscale_alone_acc: 0.2126 (n=1529)
    color_alone_acc:     0.9790 (n=1571)
    fusion_average_acc:  0.9656 (n=1571)
"""

import argparse
import re
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, "src/grayscale") if __name__ == "__main__" else None
sys.path.insert(0, "src/color") if __name__ == "__main__" else None
sys.path.insert(0, "src/fusion") if __name__ == "__main__" else None

from image_cnn_model import SimpleCNN          # noqa: E402
from color_sign_resnet18 import ColorSignResNet18  # noqa: E402
from modality_fusion_model import ModalityFusionModel, rgb_to_grayscale_3ch  # noqa: E402

SEED = 42  # matches every other training/eval script in this project

ARASL_TO_ARABIC = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'ه', 'haa': 'ح',
    'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ت',
    'taa': 'ط', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ي',
    'yaa': 'ى', 'zay': 'ز',
}

# AASL folder names carry a numeric suffix (e.g. "Ain_11") that is not part
# of the letter identity. Stripping it and lower-casing gives a stable key
# to map onto the same Arabic letters ArASL uses -- this also makes the
# match immune to the case-sensitivity defect (Section 4.2/4.4 of the
# paper): AASL's own "thal_13" folder is lowercase while the other 30 are
# capitalized, which would otherwise corrupt a positional mapping.
AASL_NAME_TO_ARASL_KEY = {
    'ain': 'ain', 'al': 'al', 'alef': 'aleff', 'beh': 'bb', 'dad': 'dhad',
    'dal': 'dal', 'feh': 'fa', 'ghain': 'ghain', 'hah': 'haa', 'heh': 'ha',
    'jeem': 'jeem', 'kaf': 'kaaf', 'khah': 'khaa', 'laa': 'la', 'lam': 'laam',
    'meem': 'meem', 'noon': 'nun', 'qaf': 'gaaf', 'reh': 'ra', 'sad': 'saad',
    'seen': 'seen', 'sheen': 'sheen', 'tah': 'taa', 'teh': 'ta',
    'tehmarbuta': 'toot', 'theh': 'thaa', 'waw': 'waw', 'yeh': 'ya',
    'zah': 'dha', 'zain': 'zay', 'thal': 'thal',
}


def aasl_folder_to_arasl_key(folder_name: str) -> str:
    stripped = re.sub(r'_\d+$', '', folder_name).replace('_', '').lower()
    key = AASL_NAME_TO_ARASL_KEY.get(stripped)
    if key is None:
        raise KeyError(f"Unrecognized AASL folder name '{folder_name}' (stripped: '{stripped}') "
                        f"-- update AASL_NAME_TO_ARASL_KEY if the real folder names differ from "
                        f"the ones recorded in data/aasl_upload_progress.json.")
    return key


def build_class_mappings(aasl_classes, grayscale_classes):
    """Returns (grayscale_class_to_fused_idx, color_class_to_fused_idx),
    both keyed by LOCAL index within each network's own checkpoint, mapping
    into the unified fused space = AASL's own class order (0..30)."""
    color_class_to_fused_idx = {i: i for i in range(len(aasl_classes))}

    arasl_key_to_local_idx = {name: i for i, name in enumerate(grayscale_classes)}
    grayscale_class_to_fused_idx = {}
    unmatched = []
    for fused_idx, aasl_name in enumerate(aasl_classes):
        arasl_key = aasl_folder_to_arasl_key(aasl_name)
        local_idx = arasl_key_to_local_idx.get(arasl_key)
        if local_idx is None:
            unmatched.append((aasl_name, arasl_key))
            continue
        grayscale_class_to_fused_idx[local_idx] = fused_idx

    if unmatched:
        print(f"[info] {len(unmatched)} AASL class(es) had no ArASL counterpart "
              f"(expected exactly one: 'yaa' has no AASL equivalent): {unmatched}")

    dropped = set(range(len(grayscale_classes))) - set(grayscale_class_to_fused_idx.keys())
    if dropped:
        dropped_names = [grayscale_classes[i] for i in dropped]
        print(f"[info] ArASL class(es) with no fused slot (expected: ['yaa']): {dropped_names}")

    return grayscale_class_to_fused_idx, color_class_to_fused_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aasl_dir", required=True, help="Real AASL dataset folder (31 class subfolders)")
    parser.add_argument("--grayscale_checkpoint", default="grayscale_simplecnn_best.pt")
    parser.add_argument("--color_checkpoint", default="color_sign_resnet18_best.pt")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_prefix", default="fusion_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gray_ckpt = torch.load(args.grayscale_checkpoint, map_location="cpu")
    color_ckpt = torch.load(args.color_checkpoint, map_location="cpu")

    grayscale_classes = gray_ckpt["classes"]
    gray_input_size = gray_ckpt["input_size"]
    color_input_size = color_ckpt["input_size"]

    grayscale_model = SimpleCNN(num_classes=len(grayscale_classes), input_size=gray_input_size)
    grayscale_model.load_state_dict(gray_ckpt["model_state_dict"])

    # AASL's real color transform must match training exactly (Resize + Normalize,
    # per train_color_resnet18.py's val_tf) for ColorSignResNet18's numbers to be meaningful.
    input_size = color_input_size  # both branches see the same resolution live frame in deployment
    color_tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(args.aasl_dir, transform=color_tf)
    aasl_classes = full_dataset.classes
    print(f"AASL: {len(full_dataset)} images, {len(aasl_classes)} classes")

    color_model = ColorSignResNet18(num_classes=len(aasl_classes), pretrained=False)
    # pretrained=False here on purpose: no need to re-download ImageNet
    # weights just to immediately overwrite them with the fine-tuned
    # checkpoint below -- saves time and an unnecessary network call.
    color_model.load_state_dict(color_ckpt["model_state_dict"])

    grayscale_class_to_fused_idx, color_class_to_fused_idx = build_class_mappings(aasl_classes, grayscale_classes)

    fusion = ModalityFusionModel(
        grayscale_model, color_model, num_classes=len(aasl_classes), fusion_mode="average",
        freeze_backbones=True,
        grayscale_class_to_fused_idx=grayscale_class_to_fused_idx,
        color_class_to_fused_idx=color_class_to_fused_idx,
    ).to(device)
    fusion.eval()

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Undo ColorSignResNet18's ImageNet normalization to recover a genuine RGB image in
    # [0,1] before deriving the grayscale input the same way the live app does
    # (rgb_to_grayscale_3ch operates on a plain RGB tensor, not a normalized one).
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    all_labels, gray_preds, color_preds, fused_preds = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            rgb01 = (images.cpu() * std + mean).clamp(0, 1).to(device)
            gray_input = rgb_to_grayscale_3ch(rgb01)
            if gray_input.shape[-1] != gray_input_size:
                gray_input = torch.nn.functional.interpolate(
                    gray_input, size=(gray_input_size, gray_input_size), mode="bilinear", align_corners=False
                )

            logits_gray = grayscale_model(gray_input)
            logits_color = color_model(images)
            gray_only_pred = logits_gray.argmax(1)
            color_only_pred = logits_color.argmax(1)

            fused_logp = fusion(images, grayscale_input=gray_input)
            fused_pred = fused_logp.argmax(1)

            # remap gray-only predictions into the fused/AASL space for a fair comparison
            remap = grayscale_class_to_fused_idx
            gray_only_fused = torch.tensor(
                [remap.get(p.item(), -1) for p in gray_only_pred]
            )

            all_labels.extend(labels.tolist())
            gray_preds.extend(gray_only_fused.tolist())
            color_preds.extend(color_only_pred.cpu().tolist())
            fused_preds.extend(fused_pred.cpu().tolist())

    all_labels = np.array(all_labels)

    def acc(preds):
        preds = np.array(preds)
        valid = preds != -1
        return float((preds[valid] == all_labels[valid]).mean()), int(valid.sum())

    gray_acc, gray_n = acc(gray_preds)
    color_acc, color_n = acc(color_preds)
    fused_acc, fused_n = acc(fused_preds)

    print(f"\n=== Real fusion evaluation on {len(val_ds)} real AASL validation images ===")
    print(f"GrayscaleCNN alone (fed a grayscale derivative of the real color image): "
          f"{gray_acc:.4f}  (n={gray_n}, excludes 'yaa' predictions with no AASL slot)")
    print(f"ColorSignResNet18 alone:                                                {color_acc:.4f}  (n={color_n})")
    print(f"ModalityFusionModel (average fusion):                                   {fused_acc:.4f}  (n={fused_n})")

    report = classification_report(all_labels, fused_preds, target_names=aasl_classes, zero_division=0)

    n_classes = len(aasl_classes)
    label_range = list(range(n_classes))

    cm_fused = confusion_matrix(all_labels, fused_preds, labels=label_range)
    cm_color = confusion_matrix(all_labels, color_preds, labels=label_range)

    gray_preds_arr = np.array(gray_preds)
    gray_valid = gray_preds_arr != -1
    cm_gray = confusion_matrix(
        all_labels[gray_valid], gray_preds_arr[gray_valid], labels=label_range
    )

    with open(f"{args.out_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"n_validation_images: {len(val_ds)}\n")
        f.write(f"grayscale_alone_acc: {gray_acc:.4f} (n={gray_n})\n")
        f.write(f"color_alone_acc: {color_acc:.4f} (n={color_n})\n")
        f.write(f"fusion_average_acc: {fused_acc:.4f} (n={fused_n})\n\n")
        f.write("Fusion per-class report:\n")
        f.write(report)

    np.save(f"{args.out_prefix}_grayscale_confusion_matrix.npy", cm_gray)
    np.save(f"{args.out_prefix}_color_confusion_matrix.npy", cm_color)
    np.save(f"{args.out_prefix}_fusion_confusion_matrix.npy", cm_fused)

    with open(f"{args.out_prefix}_class_order.txt", "w", encoding="utf-8") as f:
        for name in aasl_classes:
            f.write(f"{name}\n")

    print(f"\nSaved {args.out_prefix}_summary.txt, {args.out_prefix}_class_order.txt, and 3 confusion matrices "
          f"({args.out_prefix}_grayscale/color/fusion_confusion_matrix.npy)")


if __name__ == "__main__":
    main()

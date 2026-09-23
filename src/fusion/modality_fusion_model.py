"""
modality_fusion_model.py
==========================
Fuses the decision of two genuinely specialized networks (not just an
architectural variation) — per Section 5 of the project summary:

  - GrayscaleCNN: trained on genuinely grayscale ArASL data (the current
    SimpleCNN from image_cnn_model.py)
  - ColorSignCNN: trained on genuinely colored AASL data (color_sign_cnn.py)

During live operation on a D435 (a genuinely colored live image): a
grayscale copy derived from the same frame + the original color version are
both fed to the two networks, and their two decisions are fused.

⚠️ Runtime note: this file has not actually been run in the environment it
was written in (a sandbox without PyTorch, due to the disk constraints
described in train_color_cnn.py). Only syntactic correctness has been
verified (py_compile). You must test it in Colab with dummy inputs (dummy
tensors) — see the sanity_check() function below — before wiring it up to
the two real trained models.

Expected usage:
    grayscale_model = SimpleCNN(num_classes=32)   # your current model, from image_cnn_model.py
    grayscale_model.load_state_dict(torch.load("arasl_simplecnn_best.pt"))

    color_model = ColorSignCNN(num_classes=31)    # from color_sign_cnn.py
    color_model.load_state_dict(torch.load("color_sign_cnn_best.pt")["model_state_dict"])

    fusion = ModalityFusionModel(grayscale_model, color_model, fusion_mode="learned")
    logits = fusion(rgb_batch)  # rgb_batch: (N, 3, H, W) a genuine live color image from D435
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_grayscale_3ch(x: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of genuine RGB images (N,3,H,W) into a 3-channel
    replicated grayscale version (the same format ArASL's original data is
    in: R=G=B exactly), so it can feed directly into GrayscaleCNN without
    any change to its architecture.

    We use standard luminance weights (ITU-R BT.601) rather than a simple
    average, since this is closer to what most image-processing libraries
    (OpenCV/PIL) actually produce for a genuine grayscale conversion.
    """
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB input, got shape {x.shape}")
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.repeat(1, 3, 1, 1)  # replicate to 3 channels, matching ArASL's structure


class ModalityFusionModel(nn.Module):
    """
    Fuses the decision of GrayscaleCNN and ColorSignCNN. Supports 3 fusion
    strategies:

      - "average":  a simple average of the two networks' probabilities (no
                    extra training needed, the fastest way to get started)
      - "weighted": a weighted average with a single learnable weight per
                    network (very simple training)
      - "learned":  a small MLP that learns the fusion from the concatenated
                    pair of probability vectors (the strongest option, but
                    requires actual training)

    Important note: the two networks may be trained on slightly different
    class sets (ArASL's labels/class count may differ slightly from AASL's
    31 classes — see the discrepancies noted in
    aasl_upload_progress.json). You must pass class_mapping if the class
    order differs between the two networks, otherwise the fusion will
    combine probabilities for genuinely mismatched classes.
    """

    def __init__(self, grayscale_model: nn.Module, color_model: nn.Module,
                 num_classes: int, fusion_mode: str = "learned",
                 freeze_backbones: bool = True,
                 grayscale_class_to_fused_idx: dict | None = None,
                 color_class_to_fused_idx: dict | None = None):
        super().__init__()

        if fusion_mode not in {"average", "weighted", "learned"}:
            raise ValueError("fusion_mode must be one of: average, weighted, learned")

        self.grayscale_model = grayscale_model
        self.color_model = color_model
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode

        # Remapping tables in case the two networks' classes are not in the
        # same order/count (common between ArASL and AASL)
        self.grayscale_class_to_fused_idx = grayscale_class_to_fused_idx
        self.color_class_to_fused_idx = color_class_to_fused_idx

        if freeze_backbones:
            for p in self.grayscale_model.parameters():
                p.requires_grad = False
            for p in self.color_model.parameters():
                p.requires_grad = False

        if fusion_mode == "weighted":
            # a single learnable weight per network (starts equal, 0.5/0.5, via softmax)
            self.fusion_weights = nn.Parameter(torch.zeros(2))
        elif fusion_mode == "learned":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

    def _remap(self, probs: torch.Tensor, mapping: dict | None) -> torch.Tensor:
        """Reorders/pads a sub-model's probabilities to match the unified
        fused class space."""
        if mapping is None:
            return probs
        out = torch.zeros(probs.shape[0], self.num_classes, device=probs.device, dtype=probs.dtype)
        for local_idx, fused_idx in mapping.items():
            out[:, fused_idx] = probs[:, local_idx]
        return out

    def forward(self, rgb_input: torch.Tensor, grayscale_input: torch.Tensor | None = None):
        """
        rgb_input: (N, 3, H, W) — the color image as it comes directly from
            the D435.
        grayscale_input: optional — if you have a genuine separate grayscale
            channel (e.g. from the camera itself), pass it here instead of
            deriving it automatically from rgb_input.
        """
        gray_in = grayscale_input if grayscale_input is not None else rgb_to_grayscale_3ch(rgb_input)

        logits_gray = self.grayscale_model(gray_in)
        logits_color = self.color_model(rgb_input)

        probs_gray = self._remap(F.softmax(logits_gray, dim=1), self.grayscale_class_to_fused_idx)
        probs_color = self._remap(F.softmax(logits_color, dim=1), self.color_class_to_fused_idx)

        if self.fusion_mode == "average":
            fused_probs = (probs_gray + probs_color) / 2.0
            return torch.log(fused_probs + 1e-9)  # return log-probs to stay compatible with CrossEntropyLoss if training is needed later

        if self.fusion_mode == "weighted":
            w = F.softmax(self.fusion_weights, dim=0)
            fused_probs = w[0] * probs_gray + w[1] * probs_color
            return torch.log(fused_probs + 1e-9)

        # learned
        combined = torch.cat([probs_gray, probs_color], dim=1)
        return self.fusion_mlp(combined)  # raw logits (no softmax) — used directly with CrossEntropyLoss


def sanity_check():
    """A sanity check with dummy inputs — run this in Colab after installing
    PyTorch, before wiring up the real trained models."""
    import torch.nn as nn

    class DummyGrayscaleCNN(nn.Module):
        def __init__(self, num_classes=32):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, num_classes)

        def forward(self, x):
            return self.fc(x.flatten(1))

    class DummyColorCNN(nn.Module):
        def __init__(self, num_classes=31):
            super().__init__()
            self.fc = nn.Linear(3 * 32 * 32, num_classes)

        def forward(self, x):
            return self.fc(x.flatten(1))

    num_fused_classes = 31
    gray_map = {i: i for i in range(28)}   # example: ArASL's first 28 classes map onto the unified AASL class space
    color_map = {i: i for i in range(31)}  # all 31 AASL classes map onto themselves

    fusion = ModalityFusionModel(
        DummyGrayscaleCNN(32), DummyColorCNN(31),
        num_classes=num_fused_classes, fusion_mode="learned",
        grayscale_class_to_fused_idx=gray_map,
        color_class_to_fused_idx=color_map,
    )
    dummy_rgb = torch.rand(4, 3, 32, 32)
    out = fusion(dummy_rgb)
    assert out.shape == (4, num_fused_classes), f"unexpected shape {out.shape}"
    print("sanity_check passed. Output shape:", out.shape)


if __name__ == "__main__":
    sanity_check()

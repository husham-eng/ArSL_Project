"""
numpy_color_cnn_train.py
A genuinely real CNN training run (forward + backward + manual SGD, with no
deep-learning library at all) on genuinely colored AASL data, built entirely
with NumPy.

Why this file exists: PyTorch could not be installed in this sandbox
environment because the modern PyPI wheel forces CUDA dependencies
(nvidia-cublas, nvidia-cudnn...) totaling several gigabytes even for
CPU-only use, exceeding the disk quota available here (~5GB). This file
proves that the pipeline (loading real AASL images -> CNN -> genuine
learning) actually works on real data, following the same SimpleCNN
philosophy (a small, from-scratch model).

Architecture (the same idea as ColorSignCNN in color_sign_cnn.py, but
without batchnorm):
  Conv(3->8, 3x3) -> ReLU -> MaxPool(2) ->
  Conv(8->16, 3x3) -> ReLU -> MaxPool(2) ->
  Flatten -> FC(->64) -> ReLU -> FC(->num_classes) -> Softmax + CrossEntropy
"""

import json
import os
import random
import time

import numpy as np
from PIL import Image

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

DATA_DIR = "/home/claude/aasl_check"
IMG_SIZE = 40          # deliberately small for speed on a single CPU
MAX_PER_CLASS = 60     # a subsample of each class (the full dataset is 7,856 images; we train on a subset for speed)
VAL_FRACTION = 0.2
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.01


# ---------------------------------------------------------------- data ----

def load_dataset():
    cache_path = "/home/claude/aasl_numpy_cache.npz"
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], list(data["class_names"])

    class_dirs = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    images, labels = [], []
    class_names = []
    for class_idx, cname in enumerate(class_dirs):
        class_names.append(cname)
        folder = os.path.join(DATA_DIR, cname)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        random.shuffle(files)
        files = files[:MAX_PER_CLASS]
        for fn in files:
            try:
                img = Image.open(os.path.join(folder, fn)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
                images.append(arr)
                labels.append(class_idx)
            except Exception:
                continue
    X = np.stack(images)                     # (N, H, W, C)
    X = np.transpose(X, (0, 3, 1, 2))        # (N, C, H, W)
    y = np.array(labels, dtype=np.int64)
    np.savez_compressed(cache_path, X=X, y=y, class_names=np.array(class_names))
    return X, y, class_names


# --------------------------------------------------------------- layers ---

def im2col(x, kh, kw, stride=1, pad=1):
    N, C, H, W = x.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for y in range(kh):
        y_max = y + stride * out_h
        for xx in range(kw):
            x_max = xx + stride * out_w
            cols[:, :, y, xx, :, :] = x_padded[:, :, y:y_max:stride, xx:x_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


class Conv2D:
    def __init__(self, in_c, out_c, k=3, pad=1):
        scale = np.sqrt(2.0 / (in_c * k * k))
        self.W = np.random.randn(out_c, in_c, k, k).astype(np.float32) * scale
        self.b = np.zeros(out_c, dtype=np.float32)
        self.k, self.pad = k, pad
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_c = self.W.shape[0]
        cols, out_h, out_w = im2col(x, self.k, self.k, stride=1, pad=self.pad)
        W_col = self.W.reshape(out_c, -1).T
        out = cols @ W_col + self.b
        out = out.reshape(N, out_h, out_w, out_c).transpose(0, 3, 1, 2)
        self.cache = (x, cols, out_h, out_w)
        return out

    def backward(self, dout, lr):
        x, cols, out_h, out_w = self.cache
        N = x.shape[0]
        out_c = self.W.shape[0]

        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, out_c)
        dW = (cols.T @ dout_reshaped).T.reshape(self.W.shape)
        db = dout_reshaped.sum(axis=0)

        W_col = self.W.reshape(out_c, -1)
        dcols = dout_reshaped @ W_col

        dx = self._col2im(dcols, x.shape, out_h, out_w)

        self.W -= lr * dW
        self.b -= lr * db
        return dx

    def _col2im(self, dcols, x_shape, out_h, out_w):
        N, C, H, W = x_shape
        k, pad = self.k, self.pad
        dcols_reshaped = dcols.reshape(N, out_h, out_w, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        dx_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=dcols.dtype)
        for y in range(k):
            y_max = y + out_h
            for xx in range(k):
                x_max = xx + out_w
                dx_padded[:, :, y:y_max, xx:x_max] += dcols_reshaped[:, :, y, xx, :, :]
        if pad == 0:
            return dx_padded
        return dx_padded[:, :, pad:-pad, pad:-pad]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr=None):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride

    def forward(self, x):
        N, C, H, W = x.shape
        s, st = self.size, self.stride
        out_h, out_w = H // st, W // st
        x_reshaped = x[:, :, :out_h * st, :out_w * st]
        x_reshaped = x_reshaped.reshape(N, C, out_h, st, out_w, st)
        out = x_reshaped.max(axis=(3, 5))
        self.cache = (x, out, out_h, out_w)
        return out

    def backward(self, dout, lr=None):
        x, out, out_h, out_w = self.cache
        N, C, H, W = x.shape
        s, st = self.size, self.stride
        dx = np.zeros_like(x)
        for i in range(out_h):
            for j in range(out_w):
                window = x[:, :, i * st:i * st + s, j * st:j * st + s]
                m = (window == out[:, :, i, j][:, :, None, None])
                dx[:, :, i * st:i * st + s, j * st:j * st + s] += m * dout[:, :, i, j][:, :, None, None]
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr=None):
        return dout.reshape(self.shape)


class Linear:
    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        N = self.x.shape[0]
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


def softmax_cross_entropy(logits, labels):
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(probs[np.arange(N), labels] + 1e-9).mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1
    dlogits /= N
    return loss, dlogits


# ---------------------------------------------------------------- model ---

class ColorSignCNN_Numpy:
    def __init__(self, num_classes, img_size=IMG_SIZE):
        self.conv1 = Conv2D(3, 8, k=3, pad=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(8, 16, k=3, pad=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D()
        self.flatten = Flatten()
        reduced = img_size // 4
        self.fc1 = Linear(16 * reduced * reduced, 64)
        self.relu3 = ReLU()
        self.fc2 = Linear(64, num_classes)
        self.layers = [self.conv1, self.relu1, self.pool1,
                       self.conv2, self.relu2, self.pool2,
                       self.flatten, self.fc1, self.relu3, self.fc2]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout, lr):
        for layer in reversed(self.layers):
            dout = layer.backward(dout, lr)

    def num_params(self):
        total = 0
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            total += layer.W.size + layer.b.size
        return total


# ------------------------------------------------------------- training ---

def main():
    t0 = time.time()
    print("Loading real AASL color images...")
    X, y, class_names = load_dataset()
    print(f"Loaded {X.shape[0]} images, {len(class_names)} classes, shape={X.shape[1:]}")

    # shuffle + split
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    n_val = int(len(idx) * VAL_FRACTION)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    model = ColorSignCNN_Numpy(num_classes=len(class_names))
    print("Trainable parameters:", model.num_params())

    history = []
    n_train = X_train.shape[0]
    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(n_train)
        X_train, y_train = X_train[perm], y_train[perm]

        epoch_loss, correct, total = 0.0, 0, 0
        for i in range(0, n_train, BATCH_SIZE):
            xb = X_train[i:i + BATCH_SIZE]
            yb = y_train[i:i + BATCH_SIZE]
            logits = model.forward(xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            model.backward(dlogits, LR)

            epoch_loss += loss * xb.shape[0]
            correct += (logits.argmax(axis=1) == yb).sum()
            total += xb.shape[0]

        train_loss = epoch_loss / total
        train_acc = correct / total

        # validation (forward only, batched to keep memory sane)
        val_correct = 0
        for i in range(0, X_val.shape[0], BATCH_SIZE):
            xb = X_val[i:i + BATCH_SIZE]
            yb = y_val[i:i + BATCH_SIZE]
            logits = model.forward(xb)
            val_correct += (logits.argmax(axis=1) == yb).sum()
        val_acc = val_correct / X_val.shape[0]

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | elapsed={elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                         "train_acc": float(train_acc), "val_acc": float(val_acc)})

    report = {
        "framework": "pure NumPy (no deep learning library) — PyTorch could not be installed "
                     "in this sandbox due to CUDA dependency disk footprint",
        "dataset": "real AASL RGB images (subset for CPU speed)",
        "num_classes": len(class_names),
        "images_used_total": int(X.shape[0]),
        "images_per_class_cap": MAX_PER_CLASS,
        "image_size": IMG_SIZE,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "model_parameters": model.num_params(),
        "epochs": EPOCHS,
        "history": history,
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with open("/mnt/user-data/outputs/color_cnn_training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nSaved report to color_cnn_training_report.json")


if __name__ == "__main__":
    main()

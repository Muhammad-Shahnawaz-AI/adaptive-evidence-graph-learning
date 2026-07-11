"""
Data engine for AEGL demo runs.

Uses scikit-learn's `digits` dataset (real 8x8 handwritten digit images, no
network download required) and reshapes it into an open-world / long-tail
benchmark by sub-sampling some classes far more aggressively than others —
mirroring the ImageNet-A/R / iNaturalist heavy-tail scenario named in the
proposal, without needing an external dataset download (this sandbox's
network egress is restricted to package registries only).

Also supports user-uploaded CSV files of flattened pixel rows + a label
column, so the frontend's "upload your training file" panel is real, not
just decorative.
"""

import io
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.datasets import load_digits


class LongTailDigitsDataset(Dataset):
    """Real 8x8 digit images (0-9), artificially long-tailed by class."""

    def __init__(self, split="train", imbalance_ratio=8.0, test_size=0.25, seed=42):
        data = load_digits()
        X, y = data.images.astype(np.float32), data.target.astype(np.int64)

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X))
        X, y = X[perm], y[perm]

        n_test = int(len(X) * test_size)
        if split == "test":
            X, y = X[:n_test], y[:n_test]
        else:
            X, y = X[n_test:], y[n_test:]

        if split == "train":
            X, y = self._apply_long_tail(X, y, imbalance_ratio, rng)

        # normalize to [0, 1], add channel dim -> [N, 1, 8, 8]
        X = X / 16.0
        self.images = torch.from_numpy(X).unsqueeze(1)
        self.labels = torch.from_numpy(y)
        self.num_classes = int(y.max()) + 1 if len(y) else 10
        self.class_counts = np.bincount(y, minlength=10).tolist()

    @staticmethod
    def _apply_long_tail(X, y, imbalance_ratio, rng):
        classes = np.unique(y)
        n_classes = len(classes)
        max_count = max(np.sum(y == c) for c in classes)
        keep_idx = []
        for i, c in enumerate(classes):
            # exponential decay in per-class sample count -> heavy tail
            frac = imbalance_ratio ** (-i / max(1, n_classes - 1))
            target_count = max(3, int(max_count * frac))
            cls_idx = np.where(y == c)[0]
            rng.shuffle(cls_idx)
            keep_idx.extend(cls_idx[:target_count].tolist())
        keep_idx = np.array(keep_idx)
        rng.shuffle(keep_idx)
        return X[keep_idx], y[keep_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class UploadedCSVDataset(Dataset):
    """CSV with one label column ('label' or first column) and the rest as
    flattened pixel intensities. Values are auto-reshaped to a square image
    where possible, otherwise treated as a 1D feature vector padded to a
    small square for the conv encoder."""

    def __init__(self, csv_bytes: bytes):
        import csv as _csv
        text = csv_bytes.decode("utf-8", errors="ignore")
        reader = list(_csv.reader(io.StringIO(text)))
        header = reader[0]
        rows = reader[1:]

        label_col = 0
        for i, h in enumerate(header):
            if h.strip().lower() in ("label", "target", "class", "y"):
                label_col = i
                break

        labels = []
        features = []
        for row in rows:
            if not row:
                continue
            labels.append(int(float(row[label_col])))
            feat = [float(v) for j, v in enumerate(row) if j != label_col]
            features.append(feat)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        # min-max normalize
        if X.max() > X.min():
            X = (X - X.min()) / (X.max() - X.min())

        side = int(np.ceil(np.sqrt(X.shape[1])))
        padded = np.zeros((X.shape[0], side * side), dtype=np.float32)
        padded[:, :X.shape[1]] = X
        X = padded.reshape(-1, 1, side, side)

        # remap labels to a dense 0..K-1 range
        unique_labels = sorted(set(y.tolist()))
        remap = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.array([remap[v] for v in y.tolist()], dtype=np.int64)

        self.images = torch.from_numpy(X)
        self.labels = torch.from_numpy(y)
        self.num_classes = len(unique_labels)
        self.class_counts = np.bincount(y, minlength=self.num_classes).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

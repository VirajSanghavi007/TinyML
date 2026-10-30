"""CIFAR-10 loading (restricted unpickler) and a GPU-resident loader with on-device augmentation."""
import math
import pickle
from pathlib import Path

import numpy as np
import torch

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# The CIFAR python batches are pickles of plain dicts holding numpy arrays. Downloaded
# pickles are untrusted, so only the globals needed to rebuild numpy arrays are allowed.
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_batch(path):
    with open(path, "rb") as f:
        d = _SafeUnpickler(f, encoding="bytes").load()
    x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return x, y


def find_cifar_dir(root):
    for p in sorted(Path(root).rglob("data_batch_1")):
        return p.parent
    raise FileNotFoundError(f"no cifar-10-batches-py under {root}")


def load_cifar10(root):
    d = find_cifar_dir(root)
    parts = [load_batch(d / f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    x_test, y_test = load_batch(d / "test_batch")
    return x_train, y_train, x_test, y_test


def split_train_val(x, y, n_val=5000, seed=0):
    perm = np.random.default_rng(seed).permutation(len(x))
    val, tr = perm[:n_val], perm[n_val:]
    return x[tr], y[tr], x[val], y[val]


def augment(xb):
    """Random crop (pad 4) + horizontal flip on a uint8 batch [B,3,32,32], fully on-device."""
    b, c, h, w = xb.shape
    padded = torch.nn.functional.pad(xb, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), device=xb.device)
    ox = torch.randint(0, 9, (b,), device=xb.device)
    rows = (oy[:, None] + torch.arange(h, device=xb.device))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=xb.device))[:, None, None, :]
    bi = torch.arange(b, device=xb.device)[:, None, None, None]
    ci = torch.arange(c, device=xb.device)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    flip = torch.rand(b, device=xb.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class GPULoader:
    """Holds the whole split on `device` as uint8 and yields normalized float batches."""

    def __init__(self, x_u8, y, batch_size, device, train=False):
        self.x = torch.as_tensor(x_u8).to(device)
        self.y = torch.as_tensor(y).to(device)
        self.bs, self.train, self.device = batch_size, train, device
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.x)
        # Training drops the ragged last batch so BN never sees a tiny batch.
        return n // self.bs if self.train and n >= self.bs else math.ceil(n / self.bs)

    def normalize(self, xb):
        return (xb.float() / 255.0 - self.mean) / self.std

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for i in range(len(self)):
            b = idx[i * self.bs:(i + 1) * self.bs]
            xb = self.x[b]
            if self.train:
                xb = augment(xb)
            yield self.normalize(xb), self.y[b]

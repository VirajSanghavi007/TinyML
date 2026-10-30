import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
REAL_DATA = REPO / "data" / "raw"


def write_fake_cifar(root, n_per_batch=200, seed=0):
    """Write CIFAR-10 python-format batches with random pixels (same layout as the real files)."""
    d = Path(root) / "cifar-10-batches-py"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    for name in names:
        batch = {b"data": rng.integers(0, 256, (n_per_batch, 3072), dtype=np.uint8),
                 b"labels": rng.integers(0, 10, n_per_batch).tolist()}
        with open(d / name, "wb") as f:
            pickle.dump(batch, f, protocol=4)
    return Path(root)


@pytest.fixture
def fake_cifar(tmp_path):
    return write_fake_cifar(tmp_path / "data")


@pytest.fixture
def cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def cifar_available():
    try:
        from tinyml.data import find_cifar_dir
        return find_cifar_dir(REAL_DATA)
    except FileNotFoundError:
        pytest.skip("real CIFAR-10 not downloaded to data/raw")

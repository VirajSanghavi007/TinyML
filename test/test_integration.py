"""Integration: real CIFAR-10 files on disk and the real CUDA device. Skip cleanly when absent."""
import random

import numpy as np
import torch

from tinyml import data as D
from tinyml.ofa import Supernet, extract_subnet, sample_arch
from tinyml.qat import QTinyNet, calibrate
from tinyml.train import kd_loss


def test_real_cifar10_loads(cifar_available):
    xtr, ytr, xte, yte = D.load_cifar10(cifar_available)
    assert xtr.shape == (50000, 3, 32, 32) and xte.shape == (10000, 3, 32, 32)
    assert np.bincount(ytr).tolist() == [5000] * 10
    assert np.bincount(yte).tolist() == [1000] * 10
    assert 100 < xtr.mean() < 150


def test_gpu_supernet_kd_step_and_parity(cuda_available, cifar_available):
    xtr, ytr, _, _ = D.load_cifar10(cifar_available)
    loader = D.GPULoader(xtr[:512], ytr[:512], 128, "cuda", train=True)
    x, y = next(iter(loader))
    assert x.is_cuda and x.shape == (128, 3, 32, 32)
    sup = Supernet().cuda()
    opt = torch.optim.SGD(sup.parameters(), lr=0.05, momentum=0.9)
    arch = sample_arch(rng=random.Random(0))
    teacher_logits = torch.randn(128, 10, device="cuda")
    losses = []
    for _ in range(15):  # overfit one batch: loss must go down on the GPU
        opt.zero_grad()
        loss = kd_loss(sup(x, arch), teacher_logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    sup.eval()
    net = extract_subnet(sup, arch).eval()
    with torch.no_grad():
        assert torch.allclose(sup(x, arch), net(x), atol=1e-4)
        cpu_out = net.cpu()(x.cpu())
        assert torch.allclose(cpu_out, sup.cpu()(x.cpu(), arch), atol=1e-4)
    q = calibrate(QTinyNet(net).cuda(), [(x, y)], n_batches=1)
    assert q(x).shape == (128, 10)

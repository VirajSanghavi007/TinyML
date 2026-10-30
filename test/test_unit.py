import copy
import io
import json
import os
import pickle
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyml import data as D
from tinyml.ofa import (SPACE, Supernet, TinyNet, arch_to_spec, extract_subnet, fits, recalibrate_bn,
                        sample_arch, spec_cost)
from tinyml.prune import _keep_count, prune_tinynet
from tinyml.qat import QConv, QTinyNet, calibrate, set_observers
from tinyml.search import arch_key, crossover, evolve, mutate
from tinyml.train import cosine_with_warmup, fit, kd_loss

torch.manual_seed(0)


def _no_fq(qmodel):
    for m in qmodel.modules():
        if hasattr(m, "disable_fake_quant"):
            m.disable_fake_quant()
            m.disable_observer()
    return qmodel


def _randomize_bn(model):
    g = torch.Generator().manual_seed(1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            c = m.num_features
            m.weight.data = torch.rand(c, generator=g) + 0.5
            m.bias.data = torch.randn(c, generator=g) * 0.1
            m.running_mean.data = torch.randn(c, generator=g) * 0.1
            m.running_var.data = torch.rand(c, generator=g) + 0.5
    return model


# ------------------------------------------------------------------ data
class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_safe_unpickler_blocks_arbitrary_globals(tmp_path):
    p = tmp_path / "data_batch_1"
    p.write_bytes(pickle.dumps({b"data": _Evil(), b"labels": [0]}))
    with pytest.raises(pickle.UnpicklingError, match="blocked"):
        D.load_batch(p)


def test_load_cifar_and_split(fake_cifar):
    xtr, ytr, xte, yte = D.load_cifar10(fake_cifar)
    assert xtr.shape == (1000, 3, 32, 32) and xtr.dtype == np.uint8
    assert ytr.shape == (1000,) and xte.shape == (200, 3, 32, 32)
    a, b, c, d = D.split_train_val(xtr, ytr, n_val=100)
    assert len(a) == 900 and len(c) == 100
    # disjoint and complete
    rows = {r.tobytes() for r in np.concatenate([a, c]).reshape(1000, -1)}
    assert rows == {r.tobytes() for r in xtr.reshape(1000, -1)}


def test_augment_preserves_shape_and_is_crop_or_flip():
    x = torch.arange(2 * 3 * 32 * 32, dtype=torch.int64).remainder(251).to(torch.uint8).view(2, 3, 32, 32)
    torch.manual_seed(0)
    out = D.augment(x)
    assert out.shape == x.shape and out.dtype == torch.uint8
    # every output pixel row is either zero padding or comes from the source image
    src = set(x.flatten().tolist()) | {0}
    assert set(out.flatten().tolist()) <= src


def test_loader_batches_and_normalization():
    x = np.full((10, 3, 32, 32), 255, np.uint8)
    y = np.arange(10)
    tr = D.GPULoader(x, y, 4, "cpu", train=True)
    ev = D.GPULoader(x, y, 4, "cpu", train=False)
    assert len(tr) == 2 and len(ev) == 3  # train drops the ragged batch
    xb, yb = next(iter(ev))
    expected = (1.0 - torch.tensor(D.CIFAR_MEAN)) / torch.tensor(D.CIFAR_STD)
    assert torch.allclose(xb[0, :, 5, 5], expected)
    assert sum(len(b[1]) for b in ev) == 10


# ------------------------------------------------------------------ ofa / budget
def test_sample_arch_modes():
    amax, amin = sample_arch(mode="max"), sample_arch(mode="min")
    assert amax["stem"] == 16 and amin["stem"] == 8
    assert [s["d"] for s in amax["stages"]] == [2, 3, 3, 2]
    assert all(k == 5 for s in amax["stages"] for k in s["ks"])


def test_arch_to_spec_channels_chain():
    spec = arch_to_spec(sample_arch(rng=random.Random(3)))
    cin = spec["stem"]
    for b in spec["blocks"]:
        assert b["cin"] == cin
        cin = b["cout"]


def _tiny_spec():
    return {"stem": 4, "res": 8, "num_classes": 2, "head": 6,
            "blocks": [{"cin": 4, "hidden": 8, "cout": 4, "k": 3, "stride": 1}]}


def test_spec_cost_hand_computed():
    c = spec_cost(_tiny_spec())
    # weights: stem 3*4*9=108, block 4*8 + 8*9 + 8*4 = 136, head 4*6=24, fc 6*2=12 -> 280
    # biases: stem 4, block 8+8+4=20, head 6, fc 2 -> 32
    assert c["params"] == 280 + 32
    assert c["flash_bytes"] == 280 + 4 * 32
    # r=4 after stem. peak candidates: input 3*64 + stem 4*16 = 256; dw: h1+h2+keep = 128+128+64 = 320
    assert c["peak_ram_bytes"] == 320


def test_spec_cost_params_match_real_model():
    spec = arch_to_spec(sample_arch(rng=random.Random(5)))
    net = TinyNet(spec)
    conv_fc_weights = sum(m.weight.numel() for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    out_channels = sum(m.out_channels for m in net.modules() if isinstance(m, nn.Conv2d)) + net.fc.out_features
    assert spec_cost(spec)["params"] == conv_fc_weights + out_channels


def test_fits_budget():
    spec = arch_to_spec(sample_arch(mode="max"))
    c = spec_cost(spec)
    assert fits(spec, c["flash_bytes"], c["peak_ram_bytes"])
    assert not fits(spec, c["flash_bytes"] - 1, c["peak_ram_bytes"])
    assert not fits(spec, c["flash_bytes"], c["peak_ram_bytes"] - 1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_extracted_subnet_matches_supernet(seed):
    sup = _randomize_bn(Supernet()).eval()
    arch = sample_arch(rng=random.Random(seed))
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        ref = sup(x, arch)
        out = extract_subnet(sup, arch).eval()(x)
    assert torch.allclose(ref, out, atol=1e-5)


def test_supernet_shapes_for_extremes():
    sup = Supernet()
    for mode in ("max", "min"):
        assert sup(torch.randn(2, 3, 32, 32), sample_arch(mode=mode)).shape == (2, 10)


def test_recalibrate_bn_estimates_data_stats():
    net = nn.Sequential(nn.BatchNorm2d(3))
    data = [(torch.randn(64, 3, 4, 4) * 2 + 5, None) for _ in range(4)]
    recalibrate_bn(net, data, n_batches=4)
    assert torch.allclose(net[0].running_mean, torch.full((3,), 5.0), atol=0.2)
    assert not net.training


# ------------------------------------------------------------------ prune
def test_keep_count():
    assert _keep_count(64, 0.25) == 48
    assert _keep_count(10, 0.9) == 8  # floor at min_keep
    assert _keep_count(6, 0.5) == 6   # never keep more than exist / fewer than min(min_keep, c)


def test_prune_ratio_zero_is_identity():
    net = _randomize_bn(TinyNet(arch_to_spec(sample_arch(rng=random.Random(1))))).eval()
    x = torch.randn(2, 3, 32, 32)
    p = prune_tinynet(net, 0.0).eval()
    assert torch.allclose(net(x), p(x), atol=1e-5)


def test_prune_removes_dead_channels_exactly():
    net = _randomize_bn(TinyNet(arch_to_spec(sample_arch(rng=random.Random(2))))).eval()
    b = net.blocks[0]
    h = b.bn2.num_features
    n_dead = h - _keep_count(h, 0.3)
    with torch.no_grad():  # kill the lowest channels completely: zero BN2 output and their projection
        b.bn2.weight[:n_dead] = 0
        b.bn2.bias[:n_dead] = 0
    x = torch.randn(2, 3, 32, 32)
    p = prune_tinynet(net, 0.3).eval()
    assert p.blocks[0].bn2.num_features == h - n_dead
    assert p.spec["head"] < net.spec["head"]
    # block 0 output must be unchanged since only dead channels were removed there
    stem = lambda m: F.relu6(m.stem_bn(m.stem(x)))  # noqa: E731
    assert torch.allclose(net.blocks[0](stem(net)), p.blocks[0](stem(p)), atol=1e-5)
    assert spec_cost(p.spec)["params"] < spec_cost(net.spec)["params"]


# ------------------------------------------------------------------ qat
def test_qconv_fold_matches_conv_bn():
    conv, bn = nn.Conv2d(4, 6, 3, padding=1, bias=False), _randomize_bn(nn.BatchNorm2d(6)).eval()
    q = _no_fq(QConv(conv, bn, relu6=False))
    x = torch.randn(2, 4, 8, 8)
    assert torch.allclose(q(x), bn(conv(x)), atol=1e-5)


def test_qtinynet_without_fake_quant_matches_float():
    net = _randomize_bn(TinyNet(arch_to_spec(sample_arch(rng=random.Random(4))))).eval()
    q = _no_fq(QTinyNet(net)).eval()
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(q(x), net(x), atol=1e-4)


def test_fake_quant_is_close_but_not_exact():
    net = _randomize_bn(TinyNet(arch_to_spec(sample_arch(rng=random.Random(4))))).eval()
    x = torch.randn(16, 3, 32, 32)
    q = calibrate(QTinyNet(net), [(x, None)], n_batches=1)
    ref, out = net(x), q(x)
    assert not torch.allclose(ref, out)
    assert (ref.argmax(1) == out.argmax(1)).float().mean() > 0.5
    w = q.stem.wfq(q.stem.weight)
    scale = q.stem.wfq.scale.view(-1, 1, 1, 1)
    assert torch.allclose(w / scale, torch.round(w / scale), atol=1e-4)  # int8 grid


def test_set_observers_toggles():
    q = QTinyNet(TinyNet(arch_to_spec(sample_arch(mode="min"))))
    set_observers(q, False)
    assert all(m.observer_enabled.item() == 0 for m in q.modules() if hasattr(m, "observer_enabled"))
    set_observers(q, True)
    assert all(m.observer_enabled.item() == 1 for m in q.modules() if hasattr(m, "observer_enabled"))


# ------------------------------------------------------------------ train
def test_kd_loss_limits():
    s, y = torch.randn(8, 10), torch.randint(0, 10, (8,))
    assert torch.allclose(kd_loss(s, torch.randn(8, 10), y, alpha=0.0), F.cross_entropy(s, y))
    assert kd_loss(s, s.clone(), y, alpha=1.0).abs() < 1e-6


def test_cosine_with_warmup_shape():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    s = cosine_with_warmup(opt, 10, 2)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        s.step()
    assert lrs[0] == 0.5 and lrs[1] == 1.0 and lrs[-1] < 0.05


def _toy_fit(tmp_path, metrics, epochs, patience):
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = cosine_with_warmup(opt, epochs)
    it = iter(metrics)

    def step(x, y):
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        return loss.item()
    loader = [(torch.ones(1, 1), torch.ones(1, 1))]
    return model, fit(model, step, opt, sched, loader, lambda: next(it), epochs,
                      tmp_path / "c.ckpt", patience, "t", log=lambda m: None)


def test_fit_early_stops_and_restores_best(tmp_path):
    model, (best, ran) = _toy_fit(tmp_path, [0.1, 0.5, 0.4, 0.3, 0.2, 0.9], epochs=6, patience=2)
    assert best == 0.5 and ran == 4


def test_fit_resumes_from_checkpoint(tmp_path):
    _toy_fit(tmp_path, [0.1, 0.2], epochs=2, patience=5)
    ck = torch.load(tmp_path / "c.ckpt", weights_only=False)
    assert ck["epoch"] == 2
    # extend to 4 epochs: only 2 more metrics are consumed, starting from epoch 2
    _, (best, ran) = _toy_fit(tmp_path, [0.3, 0.25], epochs=4, patience=5)
    assert ran == 4 and best == 0.3


# ------------------------------------------------------------------ search
def test_arch_key_ignores_unused_depth_slots():
    a = sample_arch(rng=random.Random(0))
    a["stages"][1]["d"] = 1
    b = copy.deepcopy(a)
    b["stages"][1]["ks"][2] = 3 if b["stages"][1]["ks"][2] == 5 else 5
    assert arch_key(a) == arch_key(b)


def test_mutate_and_crossover_stay_in_space():
    rng = random.Random(0)
    a, b = sample_arch(rng=rng), sample_arch(rng=rng)
    for c in (mutate(a, rng, 1.0), crossover(a, b, rng)):
        for st, widths, depths in zip(c["stages"], SPACE.widths, SPACE.depths):
            assert st["w"] in widths and st["d"] in depths
            assert set(st["ks"]) <= set(SPACE.ks) and set(st["es"]) <= set(SPACE.es)


def test_evolve_respects_budget_and_improves(tmp_path):
    flash, ram = 60000, 16000
    score = lambda a: spec_cost(arch_to_spec(a))["params"] / 1e5  # noqa: E731  bigger = better, bounded by budget
    best, val, info = evolve(score, flash, ram, population=12, generations=6, parents=4, patience=10,
                             state_path=tmp_path / "s.json", log=lambda m: None)
    assert fits(arch_to_spec(best), flash, ram)
    assert info["best_hist"][-1] >= info["best_hist"][0]
    assert val == max(info["best_hist"])


def test_evolve_resume_continues(tmp_path):
    score = lambda a: spec_cost(arch_to_spec(a))["params"] / 1e5  # noqa: E731
    sp = tmp_path / "s.json"
    evolve(score, 60000, 16000, population=8, generations=2, parents=4, patience=10, state_path=sp, log=lambda m: None)
    assert json.loads(sp.read_text())["gen"] == 2
    calls = []
    _, _, info = evolve(lambda a: calls.append(1) or score(a), 60000, 16000, population=8, generations=4,
                        parents=4, patience=10, state_path=sp, log=lambda m: None)
    assert info["generations"] == 4
    assert len(calls) <= 2 * 4  # only the 2 new generations' children are evaluated


def test_evolve_raises_on_impossible_budget():
    with pytest.raises(RuntimeError, match="budget"):
        evolve(lambda a: 0.0, 10, 10, population=2, generations=1, log=lambda m: None, max_tries=50)


def test_elastic_kernel_is_centre_crop():
    from tinyml.ofa import DynConv
    conv = DynConv(4, 6, 5)
    assert torch.equal(conv.active_weight(4, 6, 3), conv.weight[:, :, 1:4, 1:4])
    assert torch.equal(conv.active_weight(2, 3, 5), conv.weight[:3, :2])


def test_fit_restores_best_epoch_weights(tmp_path):
    snapshots = []
    metrics = iter([0.1, 0.9, 0.2, 0.3])
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def step(x, y):
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        return loss.item()

    def ev():
        snapshots.append(model.weight.detach().clone())
        return next(metrics)
    fit(model, step, opt, cosine_with_warmup(opt, 4), [(torch.ones(1, 1), torch.full((1, 1), 5.0))], ev, 4,
        tmp_path / "c.ckpt", 10, "t", log=lambda m: None)
    assert torch.equal(model.weight, snapshots[1])
    assert not torch.equal(model.weight, snapshots[-1])


def test_evolve_never_re_evaluates_an_arch():
    calls = []
    score = lambda a: calls.append(arch_key(a)) or spec_cost(arch_to_spec(a))["params"] / 1e5  # noqa: E731
    _, _, info = evolve(score, 60000, 16000, population=12, generations=8, parents=4, mut_prob=0.02,
                        patience=20, log=lambda m: None)
    assert len(calls) == len(set(calls)) == info["evaluations"]


def test_fit_early_stop_not_armed_during_high_lr_phase(tmp_path):
    # plateau from epoch 2 with patience 1: unarmed it must keep going until 60% of 10 epochs
    _, (best, ran) = _toy_fit(tmp_path, [0.5] * 10, epochs=10, patience=1)
    assert ran == 6


def test_fit_min_delta_ignores_tiny_gains(tmp_path):
    metrics = [0.5, 0.5001, 0.5002, 0.5003, 0.5004]
    _, (best, ran) = _toy_fit(tmp_path, metrics, epochs=5, patience=2)
    assert ran == 3 and best == 0.5002  # tiny gains don't reset patience, but the best is still kept

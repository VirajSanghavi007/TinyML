"""Once-for-All style supernet (elastic width/depth/kernel/expand), static subnets, and MCU budget model.

Budget model assumes a TFLite Micro style int8 deployment:
  flash = int8 weights (1 B/param) + int32 biases (4 B/out-channel, BN folded)
  RAM   = peak over ops of the int8 tensors live at that op (inputs, outputs, held residual).
"""
import copy
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CIFAR_MEAN, CIFAR_STD
from .tfl_build import same_pad

# Padding modes. "torch": k//2 zeros on every side (PyTorch padding=k//2).
# "tf": TensorFlow 'same' (for stride 2 on an even input, the extra pixel goes bottom/right), so the
# int8 TFLite graph is a plain 'same' conv with no separate PAD op. The stem pads with the normalized
# value of pixel 128 instead of 0: the deployed model's int8 input is pixel - 128, whose zero padding
# is pixel 128, so torch and TFLite see identical borders.
PAD_MODES = ("torch", "tf")
STEM_FILL = tuple((128.0 / 255.0 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD))


@dataclass(frozen=True)
class Space:
    stem: tuple = (8, 12, 16)
    widths: tuple = ((8, 12, 16), (16, 24, 32), (24, 32, 40), (48, 64))
    strides: tuple = (1, 2, 2, 1)
    depths: tuple = ((1, 2), (1, 2, 3), (1, 2, 3), (1, 2))
    ks: tuple = (3, 5)
    es: tuple = (2, 3, 4)
    head: tuple = (96, 128)
    num_classes: int = 10
    res: int = 32


SPACE = Space()
# Same maximum sizes (so SPACE supernet weights load as-is) plus smaller head/last-stage options: under a
# real 20 KB TFLM arena only architectures near the bottom of SPACE fit, so the search needs room there.
SPACE_V2 = Space(widths=((8, 12, 16), (16, 24, 32), (24, 32, 40), (32, 48, 64)), head=(64, 96, 128))
SPACES = {"v1": SPACE, "v2": SPACE_V2}


# ---------------------------------------------------------------- arch sampling
def sample_arch(space=SPACE, rng=None, mode="random"):
    """Arch = {"stem", "head", "stages": [{"w","d","ks","es"}]}; ks/es hold one entry per max-depth slot."""
    rng = rng or random.Random()
    pick = {"max": max, "min": min, "random": rng.choice}[mode]
    stages = []
    for widths, depths in zip(space.widths, space.depths):
        dmax = max(depths)
        stages.append({
            "w": pick(widths),
            "d": pick(depths),
            "ks": [pick(space.ks) for _ in range(dmax)],
            "es": [pick(space.es) for _ in range(dmax)],
        })
    return {"stem": pick(space.stem), "head": pick(space.head), "stages": stages}


def arch_to_spec(arch, space=SPACE):
    """Flatten an arch into a static layer spec (explicit hidden widths, so pruning can edit them)."""
    blocks, cin = [], arch["stem"]
    for st, stride in zip(arch["stages"], space.strides):
        for j in range(st["d"]):
            blocks.append({"cin": cin, "hidden": cin * st["es"][j], "cout": st["w"],
                           "k": st["ks"][j], "stride": stride if j == 0 else 1})
            cin = st["w"]
    return {"stem": arch["stem"], "blocks": blocks, "head": arch["head"],
            "num_classes": space.num_classes, "res": space.res}


# ---------------------------------------------------------------- budget model
def spec_cost(spec):
    """Return dict(params, flash_bytes, peak_ram_bytes, macs) for an int8 deployment of `spec`."""
    r = spec["res"] // 2  # stem is stride 2
    stem = spec["stem"]
    weights, biases = 3 * stem * 9, stem
    macs = 3 * stem * 9 * r * r
    peak = 3 * spec["res"] ** 2 + stem * r * r
    for b in spec["blocks"]:
        cin, h, cout, k, s = b["cin"], b["hidden"], b["cout"], b["k"], b["stride"]
        ro = r // s
        weights += cin * h + h * k * k + h * cout
        biases += h + h + cout
        macs += cin * h * r * r + h * k * k * ro * ro + h * cout * ro * ro
        x, h1, h2, y = cin * r * r, h * r * r, h * ro * ro, cout * ro * ro
        keep = x if (s == 1 and cin == cout) else 0  # residual input stays live through the block
        peak = max(peak, x + h1, h1 + h2 + keep, h2 + y + keep)
        if keep:
            peak = max(peak, x + 2 * y)  # add: two inputs + output
        r = ro
    cin = spec["blocks"][-1]["cout"] if spec["blocks"] else stem
    head, nc = spec["head"], spec["num_classes"]
    weights += cin * head + head * nc
    biases += head + nc
    macs += cin * head * r * r + head * nc
    peak = max(peak, cin * r * r + head * r * r, head * r * r + head, head + nc)
    return {"params": weights + biases, "flash_bytes": weights + 4 * biases,
            "peak_ram_bytes": peak, "macs": macs}


def fits(spec, flash_limit, ram_limit):
    c = spec_cost(spec)
    return c["flash_bytes"] <= flash_limit and c["peak_ram_bytes"] <= ram_limit


# ---------------------------------------------------------------- padding
def pad_input(x, k, s, mode="torch", fill=None):
    """Explicitly pad x (NCHW) for a k x k / stride s conv that then runs with padding=0.

    `fill` (per-channel tensor) pads with that value instead of 0."""
    if mode == "torch":
        p = (k // 2,) * 4
    elif mode == "tf":
        t, b = same_pad(x.shape[2], k, s)
        l, r = same_pad(x.shape[3], k, s)
        p = (l, r, t, b)
    else:
        raise ValueError(f"pad mode {mode!r} not in {PAD_MODES}")
    if fill is None:
        return F.pad(x, p)
    f = fill.view(1, -1, 1, 1).to(x)
    return F.pad(x - f, p) + f


def _stem_fill(pad_mode):
    return torch.tensor(STEM_FILL) if pad_mode == "tf" else None


def _conv(x, w, stride, groups, k, pad_mode, fill):
    if k == 1:
        return F.conv2d(x, w, None, stride, 0, groups=groups)
    if pad_mode == "torch" and fill is None:
        return F.conv2d(x, w, None, stride, k // 2, groups=groups)
    return F.conv2d(pad_input(x, k, stride, pad_mode, fill), w, None, stride, 0, groups=groups)


class SameConv2d(nn.Conv2d):
    """Bias-free nn.Conv2d whose padding follows `pad_mode` (PAD_MODES); `fill` pads with a per-channel value."""

    def __init__(self, cin, cout, k, stride=1, groups=1, pad_mode="torch", fill=None):
        super().__init__(cin, cout, k, stride, 0, groups=groups, bias=False)
        self.pad_mode = pad_mode
        self.register_buffer("fill", fill, persistent=False)

    def forward(self, x):
        return _conv(x, self.weight, self.stride[0], self.groups, self.kernel_size[0], self.pad_mode, self.fill)


# ---------------------------------------------------------------- dynamic (supernet) layers
class DynConv(nn.Module):
    def __init__(self, cin, cout, k, stride=1, depthwise=False, pad_mode="torch", fill=None):
        super().__init__()
        self.kmax, self.stride, self.depthwise, self.pad_mode = k, stride, depthwise, pad_mode
        self.register_buffer("fill", fill, persistent=False)
        self.weight = nn.Parameter(torch.empty(cout, 1 if depthwise else cin, k, k))
        nn.init.kaiming_normal_(self.weight, mode="fan_out")

    def active_weight(self, cin, cout, k):
        o = (self.kmax - k) // 2  # kernel elasticity: centre crop of the largest kernel
        if self.depthwise:
            return self.weight[:cin, :, o:o + k, o:o + k]
        return self.weight[:cout, :cin, o:o + k, o:o + k]

    def forward(self, x, cout=None, k=None):
        cin, k = x.shape[1], k or self.kmax
        w = self.active_weight(cin, cout, k)
        return _conv(x, w, self.stride, cin if self.depthwise else 1, k, self.pad_mode, self.fill)


class DynBN(nn.BatchNorm2d):
    def forward(self, x):
        c = x.shape[1]
        return F.batch_norm(x, self.running_mean[:c], self.running_var[:c], self.weight[:c], self.bias[:c],
                            self.training, self.momentum, self.eps)


class DynMBConv(nn.Module):
    def __init__(self, cin_max, cout_max, stride, kmax, emax, pad_mode="torch"):
        super().__init__()
        hmax = cin_max * emax
        self.stride = stride
        self.expand, self.bn1 = DynConv(cin_max, hmax, 1), DynBN(hmax)
        self.dw, self.bn2 = DynConv(hmax, hmax, kmax, stride, depthwise=True, pad_mode=pad_mode), DynBN(hmax)
        self.proj, self.bn3 = DynConv(hmax, cout_max, 1), DynBN(cout_max)

    def forward(self, x, cout, k, e):
        h = x.shape[1] * e
        y = F.relu6(self.bn1(self.expand(x, h)))
        y = F.relu6(self.bn2(self.dw(y, k=k)))
        y = self.bn3(self.proj(y, cout))
        return x + y if self.stride == 1 and x.shape[1] == cout else y


class Supernet(nn.Module):
    def __init__(self, space=SPACE, pad_mode="torch"):
        super().__init__()
        self.space, self.pad_mode = space, pad_mode
        smax = max(space.stem)
        self.stem = DynConv(3, smax, 3, stride=2, pad_mode=pad_mode, fill=_stem_fill(pad_mode))
        self.stem_bn = DynBN(smax)
        self.stages, cin = nn.ModuleList(), smax
        for widths, stride, depths in zip(space.widths, space.strides, space.depths):
            wmax, blocks = max(widths), nn.ModuleList()
            for j in range(max(depths)):
                blocks.append(DynMBConv(cin, wmax, stride if j == 0 else 1, max(space.ks), max(space.es),
                                        pad_mode))
                cin = wmax
            self.stages.append(blocks)
        hmax = max(space.head)
        self.head, self.head_bn = DynConv(cin, hmax, 1), DynBN(hmax)
        self.fc = nn.Linear(hmax, space.num_classes)

    def forward(self, x, arch):
        x = F.relu6(self.stem_bn(self.stem(x, arch["stem"])))
        for blocks, st in zip(self.stages, arch["stages"]):
            for j in range(st["d"]):
                x = blocks[j](x, st["w"], st["ks"][j], st["es"][j])
        x = F.relu6(self.head_bn(self.head(x, arch["head"])))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return F.linear(x, self.fc.weight[:, :arch["head"]], self.fc.bias)


# ---------------------------------------------------------------- static subnet
class MBConv(nn.Module):
    def __init__(self, cin, hidden, cout, k, stride, pad_mode="torch"):
        super().__init__()
        self.expand, self.bn1 = nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden)
        self.dw = SameConv2d(hidden, hidden, k, stride, groups=hidden, pad_mode=pad_mode)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.proj, self.bn3 = nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)
        self.use_res = stride == 1 and cin == cout

    def forward(self, x):
        y = F.relu6(self.bn1(self.expand(x)))
        y = F.relu6(self.bn2(self.dw(y)))
        y = self.bn3(self.proj(y))
        return x + y if self.use_res else y


class TinyNet(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        pad = self.pad_mode = spec.get("pad", "torch")  # specs saved before pad modes existed are "torch"
        self.stem = SameConv2d(3, spec["stem"], 3, 2, pad_mode=pad, fill=_stem_fill(pad))
        self.stem_bn = nn.BatchNorm2d(spec["stem"])
        self.blocks = nn.ModuleList(MBConv(b["cin"], b["hidden"], b["cout"], b["k"], b["stride"], pad)
                                    for b in spec["blocks"])
        cin = spec["blocks"][-1]["cout"] if spec["blocks"] else spec["stem"]
        self.head = nn.Conv2d(cin, spec["head"], 1, bias=False)
        self.head_bn = nn.BatchNorm2d(spec["head"])
        self.fc = nn.Linear(spec["head"], spec["num_classes"])

    def forward(self, x):
        x = F.relu6(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        x = F.relu6(self.head_bn(self.head(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def _copy_bn(dst, src, idx):
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(dst, name).data.copy_(getattr(src, name).data[idx])


@torch.no_grad()
def extract_subnet(supernet, arch):
    """Copy the active slice of the supernet into a standalone TinyNet."""
    spec = arch_to_spec(arch, supernet.space)
    spec["pad"] = supernet.pad_mode
    net = TinyNet(spec).to(supernet.fc.weight.device)
    net.stem.weight.copy_(supernet.stem.active_weight(3, spec["stem"], 3))
    _copy_bn(net.stem_bn, supernet.stem_bn, slice(spec["stem"]))
    i = 0
    for blocks, st in zip(supernet.stages, arch["stages"]):
        for j in range(st["d"]):
            src, dst, b = blocks[j], net.blocks[i], spec["blocks"][i]
            dst.expand.weight.copy_(src.expand.active_weight(b["cin"], b["hidden"], 1))
            _copy_bn(dst.bn1, src.bn1, slice(b["hidden"]))
            dst.dw.weight.copy_(src.dw.active_weight(b["hidden"], None, b["k"]))
            _copy_bn(dst.bn2, src.bn2, slice(b["hidden"]))
            dst.proj.weight.copy_(src.proj.active_weight(b["hidden"], b["cout"], 1))
            _copy_bn(dst.bn3, src.bn3, slice(b["cout"]))
            i += 1
    cin = spec["blocks"][-1]["cout"]
    net.head.weight.copy_(supernet.head.active_weight(cin, spec["head"], 1))
    _copy_bn(net.head_bn, supernet.head_bn, slice(spec["head"]))
    net.fc.weight.copy_(supernet.fc.weight[:, :spec["head"]])
    net.fc.bias.copy_(supernet.fc.bias)
    return net


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches=20):
    """Reset BN stats and re-estimate them (cumulative average) on training batches.

    Needed after weight sharing: supernet running stats are a mixture over all subnets."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x)
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()
    return model

"""Int8 fake-quantized TinyNet (TFLite Micro scheme) for PTQ calibration and QAT.

BN is folded into conv weights/bias first (as the deployed model will have it). Weights use
per-channel symmetric int8 [-127,127]; activations use per-tensor asymmetric int8 with EMA
min/max observers. Biases stay float here (int32 at deployment, negligible error)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (FakeQuantize, MovingAverageMinMaxObserver,
                                   MovingAveragePerChannelMinMaxObserver)

from .ofa import pad_input


def act_fq():
    return FakeQuantize(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                        dtype=torch.qint8, qscheme=torch.per_tensor_affine, averaging_constant=0.01)


def weight_fq():
    return FakeQuantize(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-127, quant_max=127,
                        dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)


@torch.no_grad()
def fold_bn(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale.view(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        b = b + conv.bias * scale
    return w.clone(), b.clone()


class QConv(nn.Module):
    def __init__(self, conv, bn, relu6):
        super().__init__()
        w, b = fold_bn(conv, bn)
        self.weight, self.bias = nn.Parameter(w), nn.Parameter(b)
        self.stride, self.groups, self.relu6 = conv.stride, conv.groups, relu6
        self.k = conv.kernel_size[0]
        self.pad_mode = getattr(conv, "pad_mode", "torch")
        self.register_buffer("fill", getattr(conv, "fill", None), persistent=False)
        self.wfq, self.ofq = weight_fq(), act_fq()

    def forward(self, x):
        if self.k > 1:
            x = pad_input(x, self.k, self.stride[0], self.pad_mode, self.fill)
        y = F.conv2d(x, self.wfq(self.weight), self.bias, self.stride, 0, groups=self.groups)
        return self.ofq(F.relu6(y) if self.relu6 else y)


class QMBConv(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.expand = QConv(blk.expand, blk.bn1, True)
        self.dw = QConv(blk.dw, blk.bn2, True)
        self.proj = QConv(blk.proj, blk.bn3, False)
        self.use_res = blk.use_res
        self.add_fq = act_fq() if self.use_res else None

    def forward(self, x):
        y = self.proj(self.dw(self.expand(x)))
        return self.add_fq(x + y) if self.use_res else y


class QTinyNet(nn.Module):
    def __init__(self, tiny):
        super().__init__()
        self.spec = tiny.spec
        self.in_fq = act_fq()
        self.stem = QConv(tiny.stem, tiny.stem_bn, True)
        self.blocks = nn.ModuleList(QMBConv(b) for b in tiny.blocks)
        self.head = QConv(tiny.head, tiny.head_bn, True)
        self.pool_fq = act_fq()
        self.fc_w, self.fc_b = nn.Parameter(tiny.fc.weight.detach().clone()), nn.Parameter(tiny.fc.bias.detach().clone())
        self.fc_wfq, self.out_fq = weight_fq(), act_fq()

    def forward(self, x):
        x = self.stem(self.in_fq(x))
        for b in self.blocks:
            x = b(x)
        x = self.pool_fq(F.adaptive_avg_pool2d(self.head(x), 1).flatten(1))
        return self.out_fq(F.linear(x, self.fc_wfq(self.fc_w), self.fc_b))


def set_observers(model, enabled):
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.enable_observer() if enabled else m.disable_observer()


@torch.no_grad()
def calibrate(model, loader, n_batches=20):
    """PTQ: run observers on training batches (eval mode, fake-quant active), then freeze ranges."""
    model.eval()
    set_observers(model, True)
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(x)
    set_observers(model, False)
    return model

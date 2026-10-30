"""Structured channel pruning of a TinyNet: MBConv hidden (expansion) channels and head channels.

Block outputs are left alone: they feed residual adds, so pruning them would need coupled
masks across a whole stage. Hidden/head channels are free of that coupling."""
import copy

import torch

from .ofa import TinyNet


def _keep_count(c, ratio, min_keep=8, multiple=4):
    k = int(round(c * (1 - ratio) / multiple)) * multiple
    return max(min(min_keep, c), min(c, k))


def _top(scores, k):
    return torch.sort(torch.topk(scores, k).indices).values


@torch.no_grad()
def channel_scores(model):
    """Importance of each prunable channel = |BN gamma| x L1 norm of the outgoing weights."""
    blocks = []
    for b in model.blocks:
        blocks.append(b.bn2.weight.abs() * b.proj.weight.abs().sum(dim=(0, 2, 3)))
    head = model.head_bn.weight.abs() * model.fc.weight.abs().sum(dim=0)
    return blocks, head


def _copy_bn(dst, src, idx):
    for name in ("weight", "bias", "running_mean", "running_var"):
        getattr(dst, name).data.copy_(getattr(src, name).data[idx])


@torch.no_grad()
def prune_tinynet(model, ratio):
    """Return a new, physically smaller TinyNet with `ratio` of hidden/head channels removed."""
    block_scores, head_scores = channel_scores(model)
    spec = copy.deepcopy(model.spec)
    keeps = []
    for b, s in zip(spec["blocks"], block_scores):
        idx = _top(s, _keep_count(b["hidden"], ratio))
        b["hidden"] = len(idx)
        keeps.append(idx)
    head_idx = _top(head_scores, _keep_count(spec["head"], ratio))
    spec["head"] = len(head_idx)

    dev = model.fc.weight.device
    new = TinyNet(spec).to(dev)
    new.stem.load_state_dict(model.stem.state_dict())
    new.stem_bn.load_state_dict(model.stem_bn.state_dict())
    for src, dst, idx in zip(model.blocks, new.blocks, keeps):
        dst.expand.weight.copy_(src.expand.weight[idx])
        _copy_bn(dst.bn1, src.bn1, idx)
        dst.dw.weight.copy_(src.dw.weight[idx])
        _copy_bn(dst.bn2, src.bn2, idx)
        dst.proj.weight.copy_(src.proj.weight[:, idx])
        dst.bn3.load_state_dict(src.bn3.state_dict())
    new.head.weight.copy_(model.head.weight[head_idx])
    _copy_bn(new.head_bn, model.head_bn, head_idx)
    new.fc.weight.copy_(model.fc.weight[:, head_idx])
    new.fc.bias.copy_(model.fc.bias)
    return new

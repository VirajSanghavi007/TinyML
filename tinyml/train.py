"""KD loss, evaluation, and a resumable training loop with early stopping + progress/ETA."""
import copy
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7):
    """Hinton KD: alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(student, y)."""
    kl = F.kl_div(F.log_softmax(student_logits / T, 1), F.softmax(teacher_logits / T, 1),
                  reduction="batchmean") * T * T
    return alpha * kl + (1 - alpha) * F.cross_entropy(student_logits, y)


@torch.no_grad()
def accuracy(forward, loader):
    correct = total = 0
    for x, y in loader:
        correct += (forward(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def evaluate(model, loader):
    model.eval()
    return accuracy(model, loader)


def cosine_with_warmup(opt, total_steps, warmup_steps=0):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit(model, step_fn, opt, sched, train_loader, eval_fn, epochs, ckpt_path, patience, tag, log=print,
        arm_frac=0.6, min_delta=1e-3):
    """Train for up to `epochs`, early-stopping on `eval_fn()` (higher is better).

    `step_fn(x, y)` must compute the loss and call backward; returns a float loss.
    Saves a checkpoint each epoch (model/opt/sched/early-stop state) and resumes from it.
    Early stopping is only armed after `arm_frac` of the epochs: under a cosine schedule the LR is
    still high before that and val plateaus are LR noise, not convergence. An epoch counts as an
    improvement only if it beats the best by more than `min_delta` (the best weights are still kept).
    On exit the model holds the best-validation weights. Returns (best_metric, epochs_run)."""
    ckpt_path = Path(ckpt_path)
    start, best, bad, best_state, finished = 0, -1.0, 0, None, False
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, bad, best_state, finished = ck["epoch"], ck["best"], ck["bad"], ck["best_state"], ck["finished"]
        log(f"[{tag}] resumed at epoch {start}/{epochs}, best={best:.4f}, finished={finished}")
    t0, run_epochs = time.time(), 0
    for ep in range(start, epochs):
        if finished:
            break
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            tot += step_fn(x, y)
            n += 1
            opt.step()
            sched.step()
        metric = eval_fn()
        run_epochs += 1
        if metric > best + min_delta:
            bad = 0
        else:
            bad += 1
        if metric > best:
            best = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        done = ep + 1
        armed = done >= arm_frac * epochs
        elapsed = time.time() - t0
        eta = elapsed / run_epochs * (epochs - done)
        log(f"[{tag}] epoch {done}/{epochs} ({100 * done / epochs:.0f}%) loss={tot / max(n, 1):.4f} "
            f"val={metric:.4f} best={best:.4f} bad={bad}/{patience} - elapsed {elapsed:.0f}s, eta {eta:.0f}s (max)")
        if armed and bad >= patience:
            log(f"[{tag}] early stop at epoch {done}")
            finished = True
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "epoch": done, "best": best, "bad": bad, "best_state": best_state,
                      "finished": finished}, ckpt_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, start + run_epochs


def clone_cpu(model):
    return copy.deepcopy(model).cpu()

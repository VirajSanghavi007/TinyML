"""End-to-end pipeline. Each stage writes its outputs under --run-dir and is skipped on re-run,
and every training stage resumes from its per-epoch checkpoint.

  python -m tinyml.pipeline --data data/raw --run-dir runs/main
  uv run --group export python -m tinyml.pipeline --profile v2 --run-dir runs/v2

Profile v2 (deployment-aware): TF 'same' padding (no PAD ops), the extended SPACE_V2, a teacher and
supernet initialized from --init-from and the supernet fine-tuned for the new padding, and a search
whose budget is the *measured* int8 TFLite size and TFLM arena on the target (tinyml/tflm_cost.py).
It ends by choosing the deployment layout (plain or patch-based) with the smallest measured arena.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .ofa import SPACES, Supernet, TinyNet, arch_to_spec, extract_subnet, recalibrate_bn, sample_arch, spec_cost
from .prune import prune_tinynet
from .qat import QTinyNet, calibrate, set_observers
from .search import evolve
from .teacher import ResNet18
from .train import cosine_with_warmup, evaluate, fit, kd_loss


@dataclass
class Config:
    flash_kb: float = 100.0
    ram_kb: float = 20.0
    batch_size: int = 256
    teacher_epochs: int = 40
    teacher_patience: int = 8
    supernet_epochs: int = 120
    supernet_patience: int = 12
    scratch_epochs: int = 120
    scratch_patience: int = 12
    search_population: int = 40
    search_generations: int = 25
    search_patience: int = 5
    prune_ratio: float = 0.3
    prune_ft_epochs: int = 40
    prune_ft_patience: int = 8
    qat_epochs: int = 20
    qat_patience: int = 5
    qat_observer_epochs: int = 3
    kd_T: float = 4.0
    kd_alpha: float = 0.7
    n_train: int = 0  # 0 = all; >0 subsamples (smoke runs)
    n_eval: int = 0
    seed: int = 0
    space: str = "v1"  # ofa.SPACES key
    pad_mode: str = "torch"  # ofa.PAD_MODES
    cost: str = "analytic"  # "analytic" (ofa.spec_cost estimate) or "tflm" (measured, tinyml/tflm_cost.py)
    init_from: str = ""  # run dir whose teacher.pt / supernet.pt initialize this run
    supernet_lr: float = 0.05
    cost_workers: int = 12


def v2_config():
    """Deployment-aware run: budget = measured model bytes and TFLM arena (strict: 20 KB is the whole arena)."""
    return Config(space="v2", pad_mode="tf", cost="tflm", init_from="runs/main", supernet_epochs=40,
                  supernet_patience=8, supernet_lr=0.02, search_population=40, search_generations=25,
                  search_patience=5)


def smoke_config():
    return Config(teacher_epochs=1, supernet_epochs=1, scratch_epochs=1, search_population=4,
                  search_generations=2, search_patience=1, prune_ft_epochs=1, qat_epochs=1,
                  qat_observer_epochs=1, n_train=512, n_eval=256, batch_size=64)


class Run:
    def __init__(self, cfg, data_root, run_dir, device, log=None):
        self.cfg, self.dir, self.device = cfg, Path(run_dir), device
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
        self.results_path = self.dir / "results.json"
        self.results = json.loads(self.results_path.read_text()) if self.results_path.exists() else {}
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        xtr, ytr, xte, yte = D.load_cifar10(data_root)
        xtr, ytr, xva, yva = D.split_train_val(xtr, ytr, n_val=min(5000, len(xtr) // 10), seed=cfg.seed)
        if cfg.n_train:
            xtr, ytr = xtr[:cfg.n_train], ytr[:cfg.n_train]
        if cfg.n_eval:
            xva, yva, xte, yte = xva[:cfg.n_eval], yva[:cfg.n_eval], xte[:cfg.n_eval], yte[:cfg.n_eval]
        bs = cfg.batch_size
        self.train = D.GPULoader(xtr, ytr, bs, device, train=True)
        self.calib = D.GPULoader(xtr[:min(len(xtr), 20 * bs)], ytr[:min(len(ytr), 20 * bs)], bs, device)
        self.val = D.GPULoader(xva, yva, 1000, device)
        self.test = D.GPULoader(xte, yte, 1000, device)
        self.flash = int(cfg.flash_kb * 1024)
        self.ram = int(cfg.ram_kb * 1024)
        self.space = SPACES[cfg.space]
        self.rc = None
        if cfg.cost == "tflm":
            from .tflm_cost import RealCost
            self.rc = RealCost(self.dir / "tflm_cost_cache.jsonl", self.flash, self.ram, workers=cfg.cost_workers,
                               log=self.log)
        self.log(f"data: train={len(xtr)} val={len(xva)} test={len(xte)} device={device} "
                 f"budget flash<={self.flash}B ram<={self.ram}B")

    # ------------------------------------------------------------ bookkeeping
    def record(self, name, model=None, spec=None, **extra):
        rec = dict(extra)
        if model is not None:
            rec["test_acc"] = evaluate(model, self.test)
            rec["val_acc"] = evaluate(model, self.val)
        if spec is not None:
            rec.update(spec_cost(spec))
            if self.rc is not None:
                c = self.rc.cost(spec, all_options=True)
                rec.update({"tflm_flash": c["flash"], "tflm_arena": c["arena"], "tflm_patch": c["patch"]}
                           if c else {"tflm_arena": None})
        else:
            rec["params"] = sum(p.numel() for p in model.parameters())
        self.results[name] = rec
        self.results_path.write_text(json.dumps(self.results, indent=2))
        write_report(self.results, self.dir / "results.md", self.flash, self.ram)
        self.log(f"[result] {name}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                  for k, v in rec.items()))

    def path(self, name):
        return self.dir / name

    def spec_of(self, arch):
        return dict(arch_to_spec(arch, self.space), pad=self.cfg.pad_mode)

    def _init_file(self, name):
        """Copy name from --init-from into this run (once); returns True if this run now has it."""
        src = Path(self.cfg.init_from) / name if self.cfg.init_from else None
        if src is not None and src.exists() and not self.path(name).exists():
            shutil.copyfile(src, self.path(name))
            self.log(f"[init] {name} copied from {src}")
        return self.path(name).exists()

    def sgd(self, model, lr, wd):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

    def kd_step(self, student_fn, teacher, opt_params=None):
        cfg = self.cfg

        def step(x, y):
            with torch.no_grad():
                t = teacher(x)
            loss = kd_loss(student_fn(x), t, y, cfg.kd_T, cfg.kd_alpha)
            loss.backward()
            return loss.item()
        return step

    # ------------------------------------------------------------ stages
    def stage_teacher(self):
        out = self.path("teacher.pt")
        model = ResNet18().to(self.device)
        if out.exists() or self._init_file("teacher.pt"):
            model.load_state_dict(torch.load(out, map_location=self.device))
            if "1_teacher_fp32" not in self.results:
                self.record("1_teacher_fp32", model.eval(), note=f"ResNet-18, CE (from {self.cfg.init_from})")
            return model.eval()
        cfg = self.cfg
        torch.backends.cudnn.benchmark = True  # static shapes: autotuning pays off here only
        opt = self.sgd(model, 0.1, 5e-4)
        sched = cosine_with_warmup(opt, cfg.teacher_epochs * len(self.train), len(self.train))

        def step(x, y):
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            return loss.item()
        fit(model, step, opt, sched, self.train, lambda: evaluate(model, self.val), cfg.teacher_epochs,
            self.path("teacher.ckpt"), cfg.teacher_patience, "teacher", self.log)
        torch.backends.cudnn.benchmark = False  # subnet shapes vary per step; re-tuning each one costs more
        torch.save(model.state_dict(), out)
        self.record("1_teacher_fp32", model.eval(), note="ResNet-18, CE")
        return model.eval()

    def eval_arch(self, supernet, arch, loader):
        net = extract_subnet(supernet, arch)
        recalibrate_bn(net, self.calib)
        return evaluate(net, loader)

    def stage_supernet(self, teacher):
        out = self.path("supernet.pt")
        cfg = self.cfg
        sup = Supernet(self.space, pad_mode=cfg.pad_mode).to(self.device)
        if out.exists():
            sup.load_state_dict(torch.load(out, map_location=self.device))
            return sup
        init = Path(cfg.init_from) / "supernet.pt" if cfg.init_from else None
        if init is not None and init.exists() and not self.path("supernet.ckpt").exists():
            sup.load_state_dict(torch.load(init, map_location=self.device))  # same max sizes in every space
            self.log(f"[init] supernet weights from {init}; fine-tuning for pad={cfg.pad_mode}, space={cfg.space}")
        rng = random.Random(cfg.seed)
        amax, amin = sample_arch(self.space, mode="max"), sample_arch(self.space, mode="min")
        opt = self.sgd(sup, cfg.supernet_lr, 5e-5)
        sched = cosine_with_warmup(opt, cfg.supernet_epochs * len(self.train), 2 * len(self.train))

        def step(x, y):
            # Sandwich rule: largest + smallest + 2 random subnets per batch, all distilled from the teacher.
            with torch.no_grad():
                t = teacher(x)
            archs = [amax, amin, sample_arch(self.space, rng), sample_arch(self.space, rng)]
            total = 0.0
            for a in archs:
                loss = kd_loss(sup(x, a), t, y, cfg.kd_T, cfg.kd_alpha)
                loss.backward()
                total += loss.item()
            return total / len(archs)

        def val():
            return 0.5 * (self.eval_arch(sup, amax, self.val) + self.eval_arch(sup, amin, self.val))
        fit(sup, step, opt, sched, self.train, val, cfg.supernet_epochs, self.path("supernet.ckpt"),
            cfg.supernet_patience, "supernet", self.log)
        torch.save(sup.state_dict(), out)
        for name, a in (("2a_supernet_max_fp32", amax), ("2b_supernet_min_fp32", amin)):
            net = recalibrate_bn(extract_subnet(sup, a), self.calib)
            self.record(name, net, self.spec_of(a), note="largest/smallest subnet, inherited weights")
        return sup

    def stage_search(self, sup):
        out = self.path("search.json")
        if out.exists():
            return json.loads(out.read_text())["arch"]
        cfg = self.cfg
        t0 = time.time()
        feasible = None
        if self.rc is not None:
            def feasible(archs):
                return self.rc.fits_many([self.spec_of(a) for a in archs])
        arch, val_acc, info = evolve(lambda a: self.eval_arch(sup, a, self.val), self.flash, self.ram, self.space,
                                     population=cfg.search_population, generations=cfg.search_generations,
                                     patience=cfg.search_patience, seed=cfg.seed,
                                     state_path=self.path("search_state.json"), log=self.log,
                                     feasible=feasible, grow_from_min=self.rc is not None)
        info["seconds"] = time.time() - t0
        out.write_text(json.dumps({"arch": arch, "val_acc": val_acc, **info}, indent=2))
        net = recalibrate_bn(extract_subnet(sup, arch), self.calib)
        torch.save({"spec": net.spec, "state": net.state_dict()}, self.path("subnet.pt"))
        self.record("3_ofa_subnet_fp32", net, net.spec, note="searched subnet, inherited weights + BN recal, no fine-tune")
        return arch

    def _train_scratch(self, spec, teacher, name, use_kd):
        out = self.path(f"{name}.pt")
        if out.exists():
            return
        cfg = self.cfg
        net = TinyNet(spec).to(self.device)
        opt = self.sgd(net, 0.05, 5e-5)
        sched = cosine_with_warmup(opt, cfg.scratch_epochs * len(self.train), 2 * len(self.train))
        if use_kd:
            step = self.kd_step(net, teacher)
        else:
            def step(x, y):
                loss = torch.nn.functional.cross_entropy(net(x), y)
                loss.backward()
                return loss.item()
        fit(net, step, opt, sched, self.train, lambda: evaluate(net, self.val), cfg.scratch_epochs,
            self.path(f"{name}.ckpt"), cfg.scratch_patience, name, self.log)
        torch.save(net.state_dict(), out)
        self.record(name, net, spec, note=f"same arch trained from scratch, {'KD' if use_kd else 'CE only'}")

    def stage_baselines(self, arch, teacher):
        spec = self.spec_of(arch)
        self._train_scratch(spec, teacher, "0a_baseline_scratch_ce", use_kd=False)
        self._train_scratch(spec, teacher, "0b_baseline_scratch_kd", use_kd=True)

    def load_tiny(self, name):
        ck = torch.load(self.path(name), map_location=self.device, weights_only=False)
        net = TinyNet(ck["spec"]).to(self.device)
        net.load_state_dict(ck["state"])
        return net.eval()

    def stage_prune(self, teacher):
        out = self.path("pruned_ft.pt")
        if out.exists():
            return self.load_tiny("pruned_ft.pt")
        cfg = self.cfg
        sub = self.load_tiny("subnet.pt")
        pruned = recalibrate_bn(prune_tinynet(sub, cfg.prune_ratio), self.calib)
        self.record("4a_pruned_no_ft_fp32", pruned, pruned.spec, note=f"prune {cfg.prune_ratio:.0%} hidden/head ch, BN recal")
        init = {k: v.clone() for k, v in pruned.state_dict().items()}
        for name, use_kd in (("4b_pruned_ft_ce_fp32", False), ("4c_pruned_ft_kd_fp32", True)):
            net = TinyNet(pruned.spec).to(self.device)
            net.load_state_dict(init)
            opt = self.sgd(net, 0.01, 5e-5)
            sched = cosine_with_warmup(opt, cfg.prune_ft_epochs * len(self.train))
            if use_kd:
                step = self.kd_step(net, teacher)
            else:
                def step(x, y, net=net):
                    loss = torch.nn.functional.cross_entropy(net(x), y)
                    loss.backward()
                    return loss.item()
            fit(net, step, opt, sched, self.train, lambda net=net: evaluate(net, self.val), cfg.prune_ft_epochs,
                self.path(f"{name}.ckpt"), cfg.prune_ft_patience, name, self.log)
            self.record(name, net, net.spec, note=f"pruned + fine-tune ({'KD' if use_kd else 'CE only'})")
        torch.save({"spec": net.spec, "state": net.state_dict()}, out)  # KD variant feeds QAT
        return net.eval()

    def stage_quant(self, fp_model, teacher):
        if "5b_qat_kd_int8" in self.results:
            return
        cfg = self.cfg
        ptq = calibrate(QTinyNet(fp_model).to(self.device), self.calib)
        self.record("5a_ptq_int8", ptq, fp_model.spec, note="post-training quant of 4c, no training")
        q = QTinyNet(fp_model).to(self.device)
        calibrate(q, self.calib)
        set_observers(q, True)  # keep ranges adapting during the first epochs of QAT
        opt = torch.optim.SGD(q.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        sched = cosine_with_warmup(opt, cfg.qat_epochs * len(self.train))
        freeze_at = cfg.qat_observer_epochs * len(self.train)
        kd = self.kd_step(q, teacher)

        def step(x, y):
            if sched.last_epoch == freeze_at:
                set_observers(q, False)
            return kd(x, y)
        fit(q, step, opt, sched, self.train, lambda: evaluate(q, self.val), cfg.qat_epochs,
            self.path("qat.ckpt"), cfg.qat_patience, "qat", self.log)
        set_observers(q, False)
        torch.save(q.state_dict(), self.path("qat_int8_fakequant.pt"))
        self.record("5b_qat_kd_int8", q, fp_model.spec, note="QAT fine-tune of 4c with KD")

    def stage_deploy(self, fp_model):
        """Pick the deployment layout (plain or patch-based) with the smallest measured arena."""
        out = self.path("deploy.json")
        if self.rc is None or out.exists():
            return
        c = self.rc.cost(fp_model.spec, all_options=True)
        out.write_text(json.dumps({"patch": c["patch"], "tflm_flash": c["flash"], "tflm_arena": c["arena"],
                                   "note": "random-weight measurement; export re-measures the trained model"},
                                  indent=2))
        self.log(f"[deploy] layout patch={c['patch']} arena={c['arena']} B flash={c['flash']} B")

    def run_all(self):
        t0 = time.time()
        teacher = self.stage_teacher()
        for p in teacher.parameters():
            p.requires_grad_(False)
        sup = self.stage_supernet(teacher)
        arch = self.stage_search(sup)
        del sup
        self.stage_baselines(arch, teacher)
        fp = self.stage_prune(teacher)
        self.stage_quant(fp, teacher)
        self.stage_deploy(fp)
        if self.rc is not None:
            self.rc.close()
        self.log(f"pipeline done in {time.time() - t0:.0f}s; report at {self.dir / 'results.md'}")
        return self.results


def write_report(results, path, flash, ram):
    lines = [f"Budget: flash <= {flash} B, RAM <= {ram} B. 'est' columns: analytic estimate (ofa.spec_cost); "
             "'TFLM' columns: measured int8 .tflite bytes and arena_used_bytes on the emulated target.",
             "Accuracies on the held-out CIFAR-10 test set; val = 5k split used for early stopping/search.", "",
             "| stage | test acc | val acc | params | est flash B | est RAM B | TFLM flash B | TFLM arena B | MACs | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(results):
        r = results[name]
        f = lambda k, fmt="{:,}": fmt.format(r[k]) if k in r else "-"  # noqa: E731
        lines.append(f"| {name} | {f('test_acc', '{:.4f}')} | {f('val_acc', '{:.4f}')} | {f('params')} | "
                     f"{f('flash_bytes')} | {f('peak_ram_bytes')} | {f('tflm_flash')} | {f('tflm_arena')} | "
                     f"{f('macs')} | {r.get('note', '')} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--run-dir", default="runs/main")
    ap.add_argument("--smoke", action="store_true", help="tiny data + 1 epoch per stage")
    ap.add_argument("--profile", choices=["v1", "v2"], default="v1", help="v2: deployment-aware (see module doc)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    # Strict fp32 training as requested: no TF32 tensor-core shortcuts.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = v2_config() if args.profile == "v2" else Config()
    if args.smoke:  # keep the profile's model/cost choices, shrink everything else
        cfg = replace(smoke_config(), space=cfg.space, pad_mode=cfg.pad_mode, cost=cfg.cost, cost_workers=4)
    run = Run(cfg, args.data, args.run_dir, args.device)
    (Path(args.run_dir) / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (Path(args.run_dir) / "pid").write_text(str(os.getpid()))
    run.run_all()


if __name__ == "__main__":
    sys.exit(main())

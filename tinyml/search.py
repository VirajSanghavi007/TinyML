"""Constrained evolutionary search over supernet subnets, with early stopping and per-generation checkpoints."""
import copy
import json
import os
import random
import time
from pathlib import Path

from .ofa import SPACE, arch_to_spec, fits, sample_arch


def arch_key(arch):
    """Canonical key: only genes that affect the network (unused depth slots are ignored)."""
    st = [(s["w"], s["d"], tuple(s["ks"][:s["d"]]), tuple(s["es"][:s["d"]])) for s in arch["stages"]]
    return json.dumps([arch["stem"], arch["head"], st])


def _genes(space):
    """Mutable gene addresses: ("stem",), ("head",), (i, "w"|"d"), (i, "ks"|"es", j)."""
    g = [("stem",), ("head",)]
    for i, depths in enumerate(space.depths):
        g += [(i, "w"), (i, "d")]
        g += [(i, f, j) for f in ("ks", "es") for j in range(max(depths))]
    return g


def _choices(space, gene):
    if gene[0] == "stem":
        return space.stem
    if gene[0] == "head":
        return space.head
    i, f = gene[0], gene[1]
    return {"w": space.widths[i], "d": space.depths[i], "ks": space.ks, "es": space.es}[f]


def _get(arch, gene):
    if len(gene) == 1:
        return arch[gene[0]]
    st = arch["stages"][gene[0]]
    return st[gene[1]] if len(gene) == 2 else st[gene[1]][gene[2]]


def _set(arch, gene, v):
    if len(gene) == 1:
        arch[gene[0]] = v
    elif len(gene) == 2:
        arch["stages"][gene[0]][gene[1]] = v
    else:
        arch["stages"][gene[0]][gene[1]][gene[2]] = v


def mutate(arch, rng, prob, space=SPACE):
    a = copy.deepcopy(arch)
    for g in _genes(space):
        if rng.random() < prob:
            _set(a, g, rng.choice(_choices(space, g)))
    return a


def crossover(a, b, rng, space=SPACE):
    c = copy.deepcopy(a)
    for g in _genes(space):
        if rng.random() < 0.5:
            _set(c, g, _get(b, g))
    return c


def evolve(eval_fn, flash_limit, ram_limit, space=SPACE, population=32, generations=20, parents=8,
           mut_prob=0.15, patience=4, seed=0, state_path=None, log=print, max_tries=10000, feasible=None,
           grow_from_min=False):
    """Maximize eval_fn(arch) subject to the budget. Stops early after `patience` generations
    without improving the best score. Checkpoints to `state_path` after every generation.

    feasible(list of archs) -> list of bool decides the budget (default: the analytic spec_cost model);
    it is called on batches so an expensive measured cost can process candidates together.
    grow_from_min: draw the initial population as random mutations of the smallest arch instead of
    uniformly, for budgets where almost no uniform sample fits."""
    rng = random.Random(seed)
    if feasible is None:
        def feasible(archs):
            return [fits(arch_to_spec(a, space), flash_limit, ram_limit) for a in archs]
    cache, gen, pop, best_hist = {}, 0, [], []
    if state_path and Path(state_path).exists():
        st = json.loads(Path(state_path).read_text())
        cache, gen, pop, best_hist = st["cache"], st["gen"], st["pop"], st["best_hist"]
        rng.setstate(_tuple_state(st["rng"]))
        log(f"[search] resumed at generation {gen}/{generations}, {len(cache)} cached evals")

    def score(a):
        k = arch_key(a)
        if k not in cache:
            cache[k] = eval_fn(a)
        return cache[k]

    def valid_batch(make, n):
        """n in-budget archs from make(); `feasible` sees each distinct arch once, in batches."""
        out, known, tries = [], {}, 0
        while len(out) < n:
            want = max(8, 3 * (n - len(out)))
            cands = [make() for _ in range(want)]
            tries += want
            fresh = {arch_key(a): a for a in cands if arch_key(a) not in known}
            known.update(zip(fresh, feasible(list(fresh.values()))))
            out += [a for a in cands if known[arch_key(a)]][:n - len(out)]
            if tries >= max_tries and len(out) < n:
                raise RuntimeError("could not sample enough archs inside the budget; budget too tight for the space")
        return out

    t0 = time.time()
    if not pop:
        if grow_from_min:
            amin = sample_arch(space, mode="min")
            make = lambda: mutate(amin, rng, rng.uniform(0.05, 0.4), space)  # noqa: E731
        else:
            make = lambda: sample_arch(space, rng)  # noqa: E731
        pop = [[a, score(a)] for a in valid_batch(make, population)]
    while gen < generations:
        pop.sort(key=lambda p: -p[1])
        top = pop[:parents]
        best_hist.append(top[0][1])
        gen += 1
        elapsed = time.time() - t0
        log(f"[search] generation {gen}/{generations} ({100 * gen / generations:.0f}%) best_val={top[0][1]:.4f} "
            f"evals={len(cache)} - elapsed {elapsed:.0f}s")
        stalled = len(best_hist) > patience and max(best_hist[-patience:]) <= max(best_hist[:-patience])
        if stalled:
            log(f"[search] early stop: no improvement in {patience} generations")
        if state_path:
            _save_json(state_path, {"cache": cache, "gen": gen, "pop": pop, "best_hist": best_hist,
                                    "rng": rng.getstate(), "stopped": stalled})
        if stalled:
            break
        if gen == generations:
            break
        half = (population - parents) // 2

        def mutant():
            return mutate(rng.choice(top)[0], rng, mut_prob, space)

        def cross():
            a, b = rng.sample(top, 2)
            return crossover(a[0], b[0], rng, space)
        children = valid_batch(mutant, half) + valid_batch(cross, population - parents - half)
        pop = top + [[c, score(c)] for c in children]
    pop.sort(key=lambda p: -p[1])
    return pop[0][0], pop[0][1], {"generations": gen, "evaluations": len(cache), "best_hist": best_hist}


def _tuple_state(s):
    return (s[0], tuple(s[1]), s[2])


def _save_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)

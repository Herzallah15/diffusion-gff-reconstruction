"""

Authors: H. Alharazin and J. Yu. Panteleeva


Generate FF training data: 60 000 accepted curves per family.
Save each family as  data/<family_name>.pt  (shape: [60000, 200]).

Usage:
    python generate_training_data.py [--target 60000] [--outdir data] [--seed 42]
"""

import os
import time
import argparse
import itertools
import torch
import numpy as np
import random
torch.set_default_dtype(torch.float64)



from FFs_Classes import *
from Random_FF_Generator import *

# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
BASE_SAMPLERS = {
    "Multipole":              Multipole,
    "Z_Expansion":            Z_Expansion,
    "Meson_Pole":             Meson_Pole,
    "Modified_Exponential":   Modified_Exponential,
    "Pade_Approximant":       Pade_Approximant,
    "Dispersive_Spectral":    Dispersive_Spectral_Sampler,
    "Log_Modified_Multipole": Log_Modified_Multipole,
    "Bag_Model_Bessel":       Bag_Model_Bessel,
}

# Initial oversampling factors (attempts / target).
# The retry loop handles shortfalls automatically.
OVERSAMPLE = {
    "Multipole":              1.2,
    "Z_Expansion":            3.0,
    "Meson_Pole":             3.0,
    "Modified_Exponential":   1.8,
    "Pade_Approximant":       2.5,
    "Dispersive_Spectral":    4.0,
    "Log_Modified_Multipole": 2.5,
    "Bag_Model_Bessel":       2.0,
}


# ─────────────────────────────────────────────────────────────────────
#  Helper: generate exactly `target` curves with automatic retry
# ─────────────────────────────────────────────────────────────────────

def generate_family(name, sampler_fn, target, oversample):
    """
    Call `sampler_fn(Number_Sample=N)` in batches until we have
    >= `target` accepted curves.  Returns tensor [target, 200].
    """
    collected = []
    n_collected = 0
    total_attempts = 0
    attempt_size = int(target * oversample)

    while n_collected < target:
        t0 = time.time()
        try:
            batch = sampler_fn(Number_Sample=attempt_size)
        except RuntimeError as e:
            print(f"  [{name}] RuntimeError: {e} — doubling batch")
            attempt_size *= 2
            continue

        collected.append(batch)
        total_attempts += attempt_size
        n_collected += batch.shape[0]
        dt = time.time() - t0

        print(f"  [{name}]  +{batch.shape[0]:>7,} accepted "
              f"/ {attempt_size:>8,} attempts  |  total: {n_collected:>7,}  "
              f"({dt:.1f}s)")

        if n_collected < target:
            remaining = target - n_collected
            rate = n_collected / total_attempts
            attempt_size = max(int(remaining / max(rate, 0.01) * 1.5), 1000)

    result = torch.cat(collected, dim=0)[:target]
    assert result.shape == (target, 200), f"{name}: got {result.shape}"
    return result


# ─────────────────────────────────────────────────────────────────────
#  Convex pairwise (Class 9) — vectorized
# ─────────────────────────────────────────────────────────────────────
def generate_convex_pair(target, base_pools):
    """
    Pairwise convex combinations from all (8 choose 2) = 28 pairs.
    """
    names = list(base_pools.keys())
    pairs = list(itertools.combinations(names, 2))
    n_pairs = len(pairs)
    per_pair = int(target / n_pairs * 1.3) + 1

    collected = []
    for n_a, n_b in pairs:
        pool_a = base_pools[n_a]
        pool_b = base_pools[n_b]
        N = min(per_pair, len(pool_a), len(pool_b))

        idx_a = torch.randperm(len(pool_a))[:N]
        idx_b = torch.randperm(len(pool_b))[:N]
        lam = torch.rand(N, 1, dtype=torch.float64)

        batch = lam * pool_a[idx_a] + (1.0 - lam) * pool_b[idx_b]
        mask = batch.abs().amax(dim=1) <= 10.0
        collected.append(batch[mask])

    result = torch.cat(collected, dim=0)
    result = result[torch.randperm(len(result))]

    # ── Fill from base pools until we have enough ────────────────
    pool_all = torch.cat(list(base_pools.values()), dim=0)
    while len(result) < target:
        shortage = target - len(result)
        print(f"  [Convex_Pair] {len(result):,} < {target:,} — "
              f"filling {shortage:,} from base pools")
        idx_a = torch.randint(0, len(pool_all), (shortage,))
        idx_b = torch.randint(0, len(pool_all), (shortage,))
        lam2 = torch.rand(shortage, 1, dtype=torch.float64)
        extra = lam2 * pool_all[idx_a] + (1.0 - lam2) * pool_all[idx_b]
        mask = extra.abs().amax(dim=1) <= 10.0
        result = torch.cat([result, extra[mask]], dim=0)

    result = result[:target]
    assert result.shape == (target, 200), f"Convex_Pair: got {result.shape}"
    return result


# ─────────────────────────────────────────────────────────────────────
#  Convex multi-parent (Class 10) — vectorized in batches
# ─────────────────────────────────────────────────────────────────────
def generate_convex_multi(target, base_pools):
    """
    Dirichlet-weighted combinations of M randomly chosen parent
    families, with M randomized per batch for diversity.
    """
    names = list(base_pools.keys())
    n_families = len(names)
    collected = []
    n_collected = 0
    batch_size = min(10_000, target)

    while n_collected < target:
        # Randomize number of parents per batch
        M = random.choice([2, 3, 3, 4, 4, 5])

        # Pick M parent families for each curve in the batch
        family_idx = torch.stack([
            torch.randperm(n_families)[:M] for _ in range(batch_size)
        ])  # [batch_size, M]

        # Dirichlet weights
        alpha_val = 0.3 + 2.7 * torch.rand(batch_size, dtype=torch.float64)
        alphas = alpha_val.unsqueeze(1).expand(-1, M)
        weights = torch.distributions.Dirichlet(alphas).sample()

        # Build curves
        curves = torch.zeros(batch_size, 200, dtype=torch.float64)
        for j in range(M):
            for fam in range(n_families):
                sel = (family_idx[:, j] == fam)
                n_sel = sel.sum().item()
                if n_sel == 0:
                    continue
                pool = base_pools[names[fam]]
                idx = torch.randint(0, len(pool), (n_sel,))
                curves[sel] += weights[sel, j].unsqueeze(1) * pool[idx]

        # Filters
        mask = curves.abs().amax(dim=1) <= 10.0
        mask &= (curves.max(dim=1).values - curves.min(dim=1).values) > 1e-2

        collected.append(curves[mask])
        n_collected += mask.sum().item()

        print(f"  [Convex_Multi M={M}]  +{mask.sum().item():>6,} accepted "
              f"/ {batch_size:>6,}  |  total: {n_collected:>7,}")
    result = torch.cat(collected, dim=0)[:target]
    assert result.shape == (target, 200), f"Convex_Multi: got {result.shape}"
    return result[torch.randperm(len(result))]


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate GFF training data for diffusion model"
    )
    parser.add_argument("--target", type=int, default=60_000,
                        help="accepted curves per family (default: 60000)")
    parser.add_argument("--outdir", type=str, default="data",
                        help="output directory (default: data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    target = args.target
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"Seed:   {args.seed}")
    print(f"Target: {target:,} curves per family")
    print(f"Output: {outdir}/")
    print("=" * 65)

    wall_start = time.time()

    # ── Phase 1: Base families (Classes 1–8) ─────────────────────────
    base_pools = {}
    for name, sampler_fn in BASE_SAMPLERS.items():
        t0 = time.time()
        print(f"\n>>> {name}")

        data = generate_family(name, sampler_fn, target, OVERSAMPLE[name])

        path = os.path.join(outdir, f"{name}.pt")
        torch.save(data, path)
        base_pools[name] = data

        dt = time.time() - t0
        print(f"  -> {path}  [{data.shape[0]:,} x {data.shape[1]}]  "
              f"({dt:.1f}s)")

    # ── Phase 2: Convex pairwise (Class 9) ───────────────────────────
    print(f"\n>>> Convex_Pair")
    t0 = time.time()
    data_pair = generate_convex_pair(target, base_pools)
    path = os.path.join(outdir, "Convex_Pair.pt")
    torch.save(data_pair, path)
    print(f"  -> {path}  [{data_pair.shape[0]:,} x {data_pair.shape[1]}]  "
          f"({time.time() - t0:.1f}s)")

    # ── Phase 3: Convex multi-parent (Class 10) ──────────────────────
    print(f"\n>>> Convex_Multi")
    t0 = time.time()
    data_multi = generate_convex_multi(target, base_pools)
    path = os.path.join(outdir, "Convex_Multi.pt")
    torch.save(data_multi, path)
    print(f"  -> {path}  [{data_multi.shape[0]:,} x {data_multi.shape[1]}]  "
          f"({time.time() - t0:.1f}s)")

    # ── Summary ──────────────────────────────────────────────────────
    wall_total = time.time() - wall_start
    print("\n" + "=" * 65)
    print("Summary:")
    total = 0
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".pt"):
            d = torch.load(os.path.join(outdir, f), weights_only=True)
            print(f"  {f:<35s}  {d.shape[0]:>7,} x {d.shape[1]}")
            total += d.shape[0]
    print(f"  {'---'*12}  {'---'*4}")
    print(f"  {'TOTAL':<35s}  {total:>7,}")
    print(f"\nWall time: {wall_total / 60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    main()
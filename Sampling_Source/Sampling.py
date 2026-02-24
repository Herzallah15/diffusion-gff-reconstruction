"""
GFF Sampling 
=================================================
Authors: H. Alharazin, J. Yu. Panteleeva

Loads the trained DDPM checkpoint and generates complete GFF curves
conditioned on sparse known data points (lattice / ChPT).

Usage:
    python sample_gff.py

The script:
  1. Loads the best model + schedule + normalization stats
  2. Takes user-specified known (t, GF(t)) pairs
  3. Normalizes them to the model's internal space
  4. Runs the full 1000-step DDPM reverse process
  5. De-normalizes the output to physical units
  6. Averages over N_samples to produce mean ± std bands
"""

import os
import sys
import time
import torch
import matplotlib.pyplot as plt

# ── Import from the training script ─────────────────────────────
sys.path.append(os.path.abspath("..")) # Put here the write path for the trained model
from DiffusionModel import *


@torch.no_grad()
def sample_ddpm(
    model,
    schedule,
    mask:       "torch.Tensor",       # (L,) binary
    x_cond:     "torch.Tensor",       # (L,) normalized central values
    sigma_cond: "torch.Tensor" = None,  # (L,) normalized experimental errors 
    n_samples:  int = 60,
    batch_size: int = 2,
    device      = None,
    seed:       int = None,
    x0_clamp:   float = 5.0,
):
    """
    DDPM reverse sampler with replacement + per-sample jitter.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    L = mask.shape[0]
    T = schedule.T

    mask_batch   = mask.unsqueeze(0).to(device)         # (1, L)
    x_cond_base  = x_cond.unsqueeze(0).to(device)       # (1, L)

    # ── NEW: prepare sigma_cond for jitter ────────────────
    if sigma_cond is not None:
        sigma_cond_batch = sigma_cond.unsqueeze(0).to(device)  # (1, L)
    else:
        sigma_cond_batch = None
    # ──────────────────────────────────────────────────────

    all_x0 = []
    n_generated = 0
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    while n_generated < n_samples:
        B = min(batch_size, n_samples - n_generated)

        x_t = torch.randn(B, L, device=device)

        mask_b = mask_batch.expand(B, -1)                # (B, L)

        if sigma_cond_batch is not None:
            jitter = torch.randn(B, L, device=device) * sigma_cond_batch
            x_cond_b = x_cond_base.expand(B, -1) + jitter   # (B, L) different per sample
        else:
            x_cond_b = x_cond_base.expand(B, -1)            # (B, L) identical
        # ══════════════════════════════════════════════════

        with torch.amp.autocast(device_type, enabled=False):
            for tau in range(T, 0, -1):
                t_tensor = torch.full((B,), tau, device=device, dtype=torch.float32)

                # ── Network prediction (concatenation conditioning) ──
                v_pred = model(x_t, t_tensor, mask_b, x_cond_b)

                # ── Tweedie x̂₀ estimate ──
                sqrt_abar    = schedule.sqrt_alpha_bar[tau]
                sqrt_1m_abar = schedule.sqrt_one_minus_alpha_bar[tau]
                x0_hat = sqrt_abar * x_t - sqrt_1m_abar * v_pred
                x0_hat = x0_hat.clamp(-x0_clamp, x0_clamp)

                # ── Posterior mean ──
                alpha_bar_prev = schedule.alpha_bar[tau - 1]
                alpha_bar_curr = schedule.alpha_bar[tau]
                beta_t         = schedule.beta[tau - 1]
                alpha_t        = schedule.alpha[tau - 1]

                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_curr)
                coeff_xt = (alpha_t.sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_curr)
                mu = coeff_x0 * x0_hat + coeff_xt * x_t

                # ── Stochastic step ──
                if tau > 1:
                    sigma = schedule.posterior_variance[tau - 1].sqrt()
                    z = torch.randn_like(x_t)
                    x_t = mu + sigma * z

                    sqrt_abar_prev = schedule.sqrt_alpha_bar[tau - 1]
                    sqrt_1m_prev   = schedule.sqrt_one_minus_alpha_bar[tau - 1]
                    noise_rep      = torch.randn_like(x_t)
                    x_known_noised = sqrt_abar_prev * x_cond_b + sqrt_1m_prev * noise_rep
                    x_t = mask_b * x_known_noised + (1.0 - mask_b) * x_t
                    # ══════════════════════════════════════════

                else:
                    x_t = mu

                    x_t = mask_b * x_cond_b + (1.0 - mask_b) * x_t
                    # ══════════════════════════════════════════

        all_x0.append(x_t.cpu())
        n_generated += B
        print(f"  Generated {n_generated}/{n_samples} samples", end="\r", flush=True)

    print()
    return torch.cat(all_x0, dim=0)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  2. HELPER: Build Conditioning from Physical Data Points        ║
# ╚═══════════════════════════════════════════════════════════════════╝

def build_conditioning(
    t_grid:     "torch.Tensor",       # (L,) the full t-grid
    t_known:    "torch.Tensor",       # (K,) t-values where GF is known
    gff_known:  "torch.Tensor",       # (K,) GF(t) central values (PHYSICAL)
    norm_mu:    "torch.Tensor",       # (L,) per-grid-point mean from training
    norm_sigma: "torch.Tensor",       # (L,) per-grid-point std from training
    gff_sigma:  "torch.Tensor" = None,  # (K,) experimental errors (PHYSICAL)
):
    """
    Convert physical (t, GF(t)) data points into the model's normalized
    mask + x_cond + sigma_cond vectors.

    Returns:
        mask       : (L,) binary
        x_cond     : (L,) normalized central values at known points, 0 elsewhere
        sigma_cond : (L,) normalized experimental errors at known points, 0 elsewhere 
        indices    : list of matched grid indices
    """
    L = t_grid.shape[0]
    mask       = torch.zeros(L)
    x_cond     = torch.zeros(L)
    sigma_cond = torch.zeros(L)        
    indices    = []

    grid_spacing = (t_grid[1] - t_grid[0]).abs().item()

    for k in range(len(t_known)):
        diffs = (t_grid - t_known[k]).abs()
        j = int(diffs.argmin().item())

        if diffs[j].item() > 0.5 * grid_spacing:
            print(f"  WARNING: t_known={t_known[k]:.4f} matched to "
                  f"t_grid[{j}]={t_grid[j]:.4f} (Δ={diffs[j]:.4f})")

        x_cond[j] = (gff_known[k] - norm_mu[j]) / norm_sigma[j]
        mask[j] = 1.0
        indices.append(j)

        # ── NEW: normalize the experimental error ─────────
        if gff_sigma is not None:
            sigma_cond[j] = gff_sigma[k] / norm_sigma[j]
        # ──────────────────────────────────────────────────

    n_known = int(mask.sum().item())
    print(f"  Conditioning: {n_known} known points out of {L} grid points "
          f"({100 * n_known / L:.1f}%)")

    return mask, x_cond, sigma_cond, indices    # ← sigma_cond added


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  3. DE-NORMALIZE AND COMPUTE STATISTICS                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

def denormalize_and_summarize(
    samples_norm: torch.Tensor,     # (N, L) in normalized space
    norm_mu:      torch.Tensor,     # (L,)
    norm_sigma:   torch.Tensor,     # (L,)
):
    """
    Convert N generated curves from normalized → physical space,
    then compute point-wise statistics.

    GF_physical[j] = sample_norm[j] * sigma[j] + mu[j]

    Returns:
        samples_phys : (N, L) all curves in physical units
        mean_curve   : (L,)   point-wise mean
        std_curve    : (L,)   point-wise std (1σ uncertainty band)
        median_curve : (L,)   point-wise median
        q16, q84     : (L,)   16th and 84th percentiles (≈ 1σ for Gaussian)
    """
    # De-normalize: broadcast mu, sigma over batch dimension
    samples_phys = samples_norm * norm_sigma.unsqueeze(0) + norm_mu.unsqueeze(0)

    mean_curve   = samples_phys.mean(dim=0)
    std_curve    = samples_phys.std(dim=0)
    median_curve = samples_phys.median(dim=0).values
    q16          = samples_phys.quantile(0.16, dim=0)
    q84          = samples_phys.quantile(0.84, dim=0)

    return samples_phys, mean_curve, std_curve, median_curve, q16, q84


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  4. PLOTTING                                                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

def plot_results(
    t_grid:       torch.Tensor,
    mean_curve:   torch.Tensor,
    std_curve:    torch.Tensor,
    q16:          torch.Tensor,
    q84:          torch.Tensor,
    t_known:      torch.Tensor,
    gff_known:     torch.Tensor,
    samples_phys: torch.Tensor = None,   # (N, L) for spaghetti plot
    n_show:       int = 20,
    title:        str = "GFF Reconstruction",
    save_path:    str = None,
):
    """Plot the reconstruction with uncertainty bands and known data points."""

    # Convert to numpy only at the matplotlib boundary
    t_np     = -t_grid.numpy()
    mean_np  = mean_curve.numpy()
    std_np   = std_curve.numpy()
    q16_np   = q16.numpy()
    q84_np   = q84.numpy()
    t_kn_np  = -t_known.numpy()
    gf_kn_np = gff_known.numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # ── Individual samples (faint spaghetti) ──────────────────
    if samples_phys is not None:
        samples_np = samples_phys.numpy()
        for i in range(min(n_show, len(samples_np))):
            ax.plot(t_np, samples_np[i], color="steelblue",
                    alpha=0.08, linewidth=0.5)

    # ── Percentile band (68% CI) ─────────────────────────────
    ax.fill_between(t_np, q16_np, q84_np,
                    alpha=0.3, color="steelblue", label="68% CI (16th–84th)")

    # ── ±1σ band ─────────────────────────────────────────────
    ax.fill_between(t_np, mean_np - std_np, mean_np + std_np,
                    alpha=0.15, color="royalblue", label="Mean ± 1σ")

    # ── Mean curve ───────────────────────────────────────────
    ax.plot(t_np, mean_np, color="darkblue", linewidth=2, label="Mean")

    # ── Known data points ────────────────────────────────────
    ax.scatter(t_kn_np, gf_kn_np, color="red", s=60, zorder=5,
               edgecolors="darkred", linewidths=1.0, label="Known data")

    ax.set_xlabel(r"$t\;[\mathrm{GeV}^2]$", fontsize=13)
    ax.set_ylabel(r"$\mathrm{GF}(t)$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")

    #plt.show()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  5. MAIN: Put It All Together                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

def sample_GFF(n_samples = None, batch_size = None, seed = None,
              t_known = None, gff_known = None, gff_sigma = None, save_path = None, require_negative_at_zero=False):
    # ══════════════════════════════════════════════════════════════
    #  USER CONFIGURATION — edit this section for your data
    # ══════════════════════════════════════════════════════════════

    checkpoint_path = "../Trained_Model/trained_interpolator/checkpoints/gff_ddpm_best.pt"
    n_samples       = n_samples       # number of curves to generate
    batch_size      = batch_size        # samples per GPU batch
    seed            = seed      # set to int for reproducible debugging

    # ── Load the t-grid ──────────────────────────────────────
    # CRITICAL: must match the grid used during data generation
    # t = -torch.linspace(0.0, 2.0, 200, dtype=torch.float64)
    t_grid = -torch.linspace(0.0, 2.0, 200)
    print(f"t-grid: {t_grid.shape}, range [{t_grid.min():.3f}, {t_grid.max():.3f}]")


    # ── Your known data points (physical units) ──────────────
    # EDIT THIS: replace with actual lattice / ChPT values
    # t runs from 0 to -2 GeV²
    t_known  = t_known
    gff_known = gff_known
    # These are EXAMPLE values — replace with your real data!

    # ══════════════════════════════════════════════════════════════
    #  PIPELINE
    # ══════════════════════════════════════════════════════════════

    # ── Step 1: Load trained model ────────────────────────────

    model, schedule, config, norm = load_model(checkpoint_path)
    device = next(model.parameters()).device

    if norm is None:
        raise RuntimeError(
            "Checkpoint does not contain normalization stats (mu, sigma). "
            "Cannot de-normalize. Re-train with mu.pt and sigma.pt present "
            "in Training/, or load them manually."
        )

    norm_mu    = norm['mu'].float().cpu()    # (L,)
    norm_sigma = norm['sigma'].float().cpu()    # (L,)

    # ── Step 2: Build conditioning vectors ────────────────────
    print("\nBuilding conditioning...")
    mask, x_cond, sigma_cond, indices = build_conditioning(
        t_grid, t_known, gff_known, norm_mu, norm_sigma, gff_sigma
    )

    # ── Step 3: Sample ────────────────────────────────────────
    print(f"\nSampling {n_samples} curves (T={schedule.T} steps)...")
    t_start = time.time()

    samples_norm = sample_ddpm(
        model, schedule, mask, x_cond, sigma_cond=sigma_cond,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )

    elapsed = time.time() - t_start
    print(f"  Output shape: {samples_norm.shape}")  # (n_samples, 200)
    print(f"  Time: {elapsed:.1f}s ({elapsed / n_samples:.2f}s per sample, "
          f"{n_samples / elapsed:.1f} samples/s)")
    torch.save({
        'samples_norm': samples_norm,
        't_grid': t_grid,
        't_known': t_known,
        'gff_known': gff_known,
        'mask': mask,
        'x_cond': x_cond,
    }, f"samples_norm_backup_{save_path}.pt")
    print(f"  Raw normalized samples saved (backup)")
    # ── Step 4: De-normalize and compute statistics ───────────
    print("\nDe-normalizing and computing statistics...")
    samples_phys, mean_curve, std_curve, median_curve, q16, q84 = \
        denormalize_and_summarize(samples_norm, norm_mu, norm_sigma)
    # ── Optional: filter out unphysical curves ────────────
    if require_negative_at_zero:
        #keep = samples_phys[:, 0] < 0          # Needed for D(0) < 0
        keep = (samples_phys < 0).all(dim=1) # Needed for more conservative condition D(t) < 0
        n_before = samples_phys.shape[0]
        samples_phys = samples_phys[keep]
        print(f"  Filter D(0)<0: kept {samples_phys.shape[0]}/{n_before} "
              f"({100*samples_phys.shape[0]/n_before:.1f}%)")
        # recompute statistics on filtered set
        mean_curve   = samples_phys.mean(dim=0)
        std_curve    = samples_phys.std(dim=0)
        median_curve = samples_phys.median(dim=0).values
        q16          = samples_phys.quantile(0.16, dim=0)
        q84          = samples_phys.quantile(0.84, dim=0)

    # ── Step 5: Report fidelity at known points ──────────────
    print("\nFidelity check at known points:")
    print(f"  {'Index':>6s}  {'t':>8s}  {'True':>10s}  {'Mean':>10s}  "
          f"{'|Δ|':>8s}  {'Rel %':>7s}")
    print(f"  {'-'*55}")
    for k, j in enumerate(indices):
        true_val = gff_known[k].item()
        pred_val = mean_curve[j].item()
        delta    = abs(pred_val - true_val)
        rel_pct  = 100 * delta / max(abs(true_val), 1e-10)
        print(f"  {j:6d}  {t_known[k]:8.3f}  {true_val:10.4f}  "
              f"{pred_val:10.4f}  {delta:8.4f}  {rel_pct:6.1f}%")

    # ── Step 6: Plot ─────────────────────────────────────────
    print("\nPlotting...")
    plot_results(
        t_grid       = t_grid,
        mean_curve   = mean_curve,
        std_curve    = std_curve,
        q16          = q16,
        q84          = q84,
        t_known      = t_known,
        gff_known     = gff_known,
        samples_phys = samples_phys,
        n_show       = 20,
        title        = "GFF Reconstruction",
        save_path    = "gff_reconstruction.png",
    )

    # ── Step 7: Save raw data ────────────────────────────────
    output = {
        't_grid':       t_grid,                       # (200,)
        'samples_phys': samples_phys,                 # (N, 200)
        'mean_curve':   mean_curve,                   # (200,)
        'std_curve':    std_curve,                     # (200,)
        'median_curve': median_curve,                  # (200,)
        'q16':          q16,                           # (200,)
        'q84':          q84,                           # (200,)
        't_known':      t_known,
        'gff_known':     gff_known,
        'mask':         mask,                          # (200,)
        'x_cond':       x_cond,                        # (200,)
    }
    save_path = f"gff_reconstruction_results_{save_path}.pt"
    torch.save(output, save_path)
    print(f"  Results saved to {save_path}")

    print("\nDone.")
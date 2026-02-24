#!/usr/bin/env python3
"""
Mask-Distribution Fine-Tuning for GFF DDPM (Phase 2)
=====================================================
Authors: H. Alharazin, J. Yu. Panteleeva

Continues from the Phase 1 fine-tuned checkpoint (epoch 100) with an
expanded mask distribution that covers ALL inference scenarios:

  Phase 1 fine-tuning (epochs 1-100):
    - Min-SNR-γ weighting, dropout 0.05, flat LR 2e-5
    - Original 3-type mask distribution

  Phase 2 fine-tuning (epochs 101-150):    ← THIS SCRIPT
    - Same Min-SNR-γ, dropout, LR
    - NEW 5-type mask distribution:

        30%  random         5-30 pts anywhere on grid     (general coverage)
        20%  low-t cluster  5-15 pts in first 30%          (lattice-like, low -t)
        20%  high-t cluster 5-15 pts in last 40%           (high -t extrapolation)
        10%  sparse-spread  8-12 pts quasi-uniformly       (realistic Hackett spacing)
        20%  unconditional  mask = all zeros                (prior regularizer)

The rationale:
  - The model already knows the GFF shape manifold (600K curves, 10 classes)
  - It already knows how to use conditioning (Phase 1: 100 epochs)
  - We are only teaching it new PATTERNS of where observations can appear
  - 50 epochs is sufficient for this distributional adjustment

The checkpoint is a DROP-IN REPLACEMENT: same architecture, same keys.
load_model() and all inference/sampling code work unchanged.

Usage:
    python finetune_masks.py

    python finetune_masks.py --checkpoint checkpoints/gff_ddpm_final.pt \
                             --data X_norm.pt --epochs 50
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Import all architecture classes from original training script ──
from DiffusionModel import *


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  EXTENDED MASK DISTRIBUTION                                      ║
# ╚═══════════════════════════════════════════════════════════════════╝

class VPredictionDDPM_V3(VPredictionDDPM):
    """
    Extends VPredictionDDPM with:
      1. Five-type mask distribution (covering all inference scenarios)
      2. Min-SNR-γ loss weighting (same as Phase 1 fine-tuning)

    Inherited unchanged from parent:
      q_sample, compute_v_target, predict_x0_from_v, predict_eps_from_v
    """

    def __init__(self, model, schedule, device, snr_gamma: float = 5.0):
        super().__init__(model, schedule, device)
        self.snr_gamma = snr_gamma

        # Precompute SNR for timesteps 1..T
        alpha_bar = schedule.alpha_bar[1:]                         # (T,)
        self.snr = (alpha_bar / (1.0 - alpha_bar)).to(device)     # (T,)

    # ── NEW: Five-type mask distribution ─────────────────────────

    def random_mask(self, x0: torch.Tensor) -> tuple:
        """
        Generate conditioning masks from the expanded 5-type distribution.

        Distribution:
          30%  random:        5-30 pts anywhere on the grid
          20%  low-t cluster: 5-15 pts in the first 30% (indices 0..59)
          20%  high-t cluster:5-15 pts in the last 40%  (indices 120..199)
          10%  sparse-spread: 8-12 pts quasi-uniformly across the full grid
          20%  unconditional: mask = all zeros

        The sparse-spread mode divides the grid into K equal segments and
        picks one random point per segment.  This mimics realistic lattice
        data distributions where points are spread but not uniformly random.

        Args:
            x0 : (B, L) clean data
        Returns:
            mask   : (B, L) binary
            x_cond : (B, L) clean values where mask=1, zero elsewhere
        """
        B, L = x0.shape
        device = x0.device

        # ── Decide mask type per sample ──────────────────────
        r = torch.rand(B, device=device)
        is_random      = (r < 0.30)                        # 30%
        is_low_cluster = (r >= 0.30) & (r < 0.50)          # 20%
        is_high_cluster= (r >= 0.50) & (r < 0.70)          # 20%
        is_sparse      = (r >= 0.70) & (r < 0.80)          # 10%
        # remaining 20%: unconditional (mask stays zero)

        mask = torch.zeros(B, L, device=device)

        # ── 30%: Random masks (5-30 pts anywhere) ───────────
        idx = is_random.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            for i in idx:
                n = torch.randint(5, 31, (1,), device=device).item()
                perm = torch.randperm(L, device=device)[:n]
                mask[i, perm] = 1.0

        # ── 20%: Low-t cluster (5-15 pts in first 30%) ──────
        idx = is_low_cluster.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            low_end = max(int(0.30 * L), 15)   # index 0..59
            for i in idx:
                n = torch.randint(5, 16, (1,), device=device).item()
                perm = torch.randperm(low_end, device=device)[:n]
                mask[i, perm] = 1.0

        # ── 20%: High-t cluster (5-15 pts in last 40%) ──────
        idx = is_high_cluster.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            high_start = int(0.60 * L)          # index 120..199
            high_len   = L - high_start          # = 80
            for i in idx:
                n = torch.randint(5, 16, (1,), device=device).item()
                perm = torch.randperm(high_len, device=device)[:n]
                mask[i, high_start + perm] = 1.0

        # ── 10%: Sparse-spread (8-12 pts quasi-uniform) ─────
        #    Divide grid into K equal segments using linspace
        #    boundaries, pick 1 random point per segment.
        #    linspace guarantees full grid coverage regardless
        #    of whether L is divisible by K.
        idx = is_sparse.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            for i in idx:
                n = torch.randint(8, 13, (1,), device=device).item()
                boundaries = torch.linspace(0, L, n + 1, device=device).long()
                for seg in range(n):
                    seg_start = boundaries[seg].item()
                    seg_end   = boundaries[seg + 1].item()
                    seg_end   = max(seg_end, seg_start + 1)
                    j = seg_start + torch.randint(0, seg_end - seg_start, (1,), device=device).item()
                    mask[i, j] = 1.0

        # ── 20%: Unconditional (mask stays zero) ────────────

        # ── Condition values ─────────────────────────────────
        x_cond = x0 * mask

        return mask, x_cond

    # ── Min-SNR-γ weighted training step (same as Phase 1) ───────

    def training_step_weighted(self, x0: torch.Tensor):
        """
        Training step with Min-SNR-γ weighting + new mask distribution.

        Returns:
            weighted_loss   : scalar (used for backprop)
            unweighted_loss : scalar (used for logging, comparable to all runs)
        """
        B = x0.shape[0]

        t = torch.randint(1, self.schedule.T + 1, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Uses the NEW random_mask defined above
        mask, x_cond = self.random_mask(x0)

        v_target = self.compute_v_target(x0, noise, t)
        v_pred = self.model(x_t, t.float(), mask, x_cond)

        per_sample_mse = F.mse_loss(v_pred, v_target, reduction='none').mean(dim=1)
        unweighted_loss = per_sample_mse.mean()

        snr_t = self.snr[t - 1]
        weights = torch.clamp(snr_t, max=self.snr_gamma) / snr_t
        weighted_loss = (weights * per_sample_mse).mean()

        return weighted_loss, unweighted_loss


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FINE-TUNING LOOP (Phase 2)                                      ║
# ╚═══════════════════════════════════════════════════════════════════╝

def finetune_masks(
    checkpoint_path: str   = "checkpoints/gff_ddpm_final.pt",
    data_path:       str   = "X_norm.pt",
    norm_stats_path: str   = "normalization.pt",
    epochs:          int   = 50,
    epoch_offset:    int   = 100,       # labels start at 101
    batch_size:      int   = 256,
    lr:              float = 2e-5,
    weight_decay:    float = 1e-4,
    grad_clip:       float = 1.0,
    ema_decay:       float = 0.9999,
    dropout:         float = 0.05,
    snr_gamma:       float = 5.0,
    val_fraction:    float = 0.05,
    save_every:      int   = 10,
    use_amp:         bool  = True,
    output_dir:      str   = "checkpoints_phase2",
    seed:            int   = 42,
):
    """
    Phase 2 fine-tuning: expanded mask distribution.

    Continues from Phase 1 checkpoint with:
      - Same: flat LR, Min-SNR-γ, dropout 0.05
      - NEW:  5-type mask distribution (random, low-t, high-t, sparse, uncond.)
      - Epoch labels: 101, 102, ..., 150 (for continuity with Phase 1 logs)

    Validation note:
      The validation loss uses the PARENT class's random_mask() (3-type:
      50% random, 30% low-t, 20% unconditional) via the explicit call
      VPredictionDDPM.training_step().  This is intentional: it keeps
      the validation metric comparable across all three training phases
      (original → Phase 1 → Phase 2).  The new mask types (high-t,
      sparse-spread) are exercised only during training.
    """

    # ── Reproducibility ──────────────────────────────────────────
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Device ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        print(f"  cuDNN benchmark: enabled")

    os.makedirs(output_dir, exist_ok=True)

    # ── AMP ──────────────────────────────────────────────────────
    if use_amp and device.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print(f"  AMP: bfloat16")
        else:
            amp_dtype = torch.float16
            print(f"  AMP: float16 (with GradScaler)")
    else:
        use_amp = False
        amp_dtype = torch.float32
        print(f"  AMP: disabled")

    use_scaler = use_amp and (amp_dtype == torch.float16)

    # ── Load Phase 1 checkpoint ──────────────────────────────────
    print(f"\nLoading Phase 1 checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt['config']

    phase1_epoch = ckpt.get('epoch', '?')
    phase1_val   = ckpt.get('val_loss', '?')
    print(f"  Phase 1: epoch {phase1_epoch}, val loss {phase1_val}")
    print(f"  Config: grid_size={config['grid_size']}, "
          f"hidden_dim={config['hidden_dim']}, "
          f"n_res_blocks={config['n_res_blocks']}")

    # ── Rebuild model (same dropout as Phase 1) ──────────────────
    print(f"\n  Rebuilding model with dropout={dropout}")
    model = GFFDiffusionNet(
        grid_size    = config['grid_size'],
        hidden_dim   = config['hidden_dim'],
        kernel_size  = config['kernel_size'],
        n_res_blocks = config['n_res_blocks'],
        n_groups     = config['n_groups'],
        n_heads      = config['n_heads'],
        dropout      = dropout,
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    print(f"  Weights loaded successfully")

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # ── Schedule ─────────────────────────────────────────────────
    T = config['T']
    if 'schedule' in ckpt:
        schedule = CosineSchedule(T=T)
        for key in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar',
                     'alpha', 'beta', 'posterior_variance']:
            setattr(schedule, key, ckpt['schedule'][key])
        schedule.to(device)
    else:
        schedule = CosineSchedule(T=T)
        schedule.to(device)

    # ── EMA (initialized from loaded EMA weights) ────────────────
    ema = EMA(model, decay=ema_decay)
    print(f"  EMA initialized from loaded weights (decay={ema_decay})")

    # ── Diffusion wrapper with NEW mask distribution ─────────────
    diffusion = VPredictionDDPM_V3(model, schedule, device, snr_gamma=snr_gamma)
    diffusion_val = VPredictionDDPM(model, schedule, device) 
    print(f"  Min-SNR-γ: {snr_gamma}")
    print(f"  Mask distribution: 30% random, 20% low-t, 20% high-t, "
          f"10% sparse-spread, 20% unconditional")

    # ── Load data ────────────────────────────────────────────────
    print(f"\nLoading data from {data_path}...")
    X_all = torch.load(data_path, weights_only=True).float()
    print(f"  Dataset shape: {X_all.shape}")
    print(f"  Value range:   [{X_all.min():.3f}, {X_all.max():.3f}]")

    # ── Load normalization statistics ────────────────────────────
    norm_mu    = None
    norm_sigma = None
    if os.path.exists(norm_stats_path):
        stats      = torch.load(norm_stats_path, weights_only=True)
        norm_mu    = stats["mu"].float().cpu()
        norm_sigma = stats["sigma"].float().cpu()
        print(f"  Normalization stats loaded: mu {norm_mu.shape}, sigma {norm_sigma.shape}")
    else:
        print(f"  WARNING: Normalization file not found at {norm_stats_path}")
        print(f"           Checkpoints will not contain normalization statistics.")

    # ── Train/val split ──────────────────────────────────────────
    N = X_all.shape[0]
    n_val   = max(int(N * val_fraction), batch_size)
    n_train = N - n_val

    perm = torch.randperm(N)
    X_train = X_all[perm[:n_train]]
    X_val   = X_all[perm[n_train:]]
    del X_all

    print(f"  Train: {n_train}   Val: {n_val}  ({val_fraction*100:.0f}%)")

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(TensorDataset(X_val), batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True,
                              drop_last=False)

    # ── Optimizer (flat LR, same as Phase 1) ─────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"\n  Optimizer: AdamW, lr={lr} (flat), weight_decay={weight_decay}")

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ── Config (records full lineage) ────────────────────────────
    phase2_config = config.copy()
    phase2_config.update({
        'phase2_from':          checkpoint_path,
        'phase2_from_epoch':    phase1_epoch,
        'phase2_dropout':       dropout,
        'phase2_lr':            lr,
        'phase2_snr_gamma':     snr_gamma,
        'phase2_epochs':        epochs,
        'phase2_epoch_offset':  epoch_offset,
        'phase2_mask_types':    '30% random, 20% low-t, 20% high-t, '
                                '10% sparse-spread, 20% unconditional',
        'phase2_val_frac':      val_fraction,
    })

    # ── CSV log ──────────────────────────────────────────────────
    log_path = os.path.join(output_dir, "phase2_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_weighted,train_unweighted,val_unweighted,lr\n")

    # ── Training loop ────────────────────────────────────────────
    start_label = epoch_offset + 1
    end_label   = epoch_offset + epochs

    print(f"\n{'='*70}")
    print(f"  Phase 2: Mask-Distribution Fine-Tuning")
    print(f"  Epochs {start_label}–{end_label} (continuing from Phase 1 epoch {phase1_epoch})")
    print(f"  Masks: random 30%, low-t 20%, high-t 20%, sparse 10%, uncond 20%")
    print(f"  Same: flat LR {lr}, Min-SNR-γ={snr_gamma}, dropout={dropout}")
    print(f"  Batch size: {batch_size}   Val set: {n_val} ({val_fraction*100:.0f}%)")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')

    for epoch_rel in range(1, epochs + 1):
        epoch_label = epoch_offset + epoch_rel

        model.train()
        total_weighted   = 0.0
        total_unweighted = 0.0
        n_batches = 0

        for (x0_batch,) in train_loader:
            x0_batch = x0_batch.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                weighted_loss, unweighted_loss = diffusion.training_step_weighted(x0_batch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            total_weighted   += weighted_loss.item()
            total_unweighted += unweighted_loss.item()
            n_batches += 1

        avg_train_w  = total_weighted / n_batches
        avg_train_uw = total_unweighted / n_batches

        # ── Validation (unweighted MSE with EMA weights) ─────
        # NOTE: Uses PARENT's random_mask() (3-type distribution)
        # via explicit VPredictionDDPM.training_step() call.
        # This is intentional for cross-phase comparability.
        val_loss = 0.0
        n_val_batches = 0
        ema.apply(model)
        model.eval()
        with torch.no_grad():
            for (x0_val,) in val_loader:
                x0_val = x0_val.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    #vl = VPredictionDDPM.training_step(diffusion, x0_val)
                    vl = diffusion_val.training_step(x0_val)
                val_loss += vl.item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)
        ema.restore(model)

        # ── Logging ──────────────────────────────────────────
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            marker = "  ★ best"

        print(f"Epoch {epoch_label:3d}/{end_label}  |  "
              f"Train(w): {avg_train_w:.6f}  "
              f"Train(uw): {avg_train_uw:.6f}  |  "
              f"Val(uw): {avg_val_loss:.6f}  |  "
              f"LR: {lr:.2e}{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch_label},{avg_train_w:.8f},{avg_train_uw:.8f},"
                    f"{avg_val_loss:.8f},{lr:.2e}\n")

        # ── Save checkpoint ──────────────────────────────────
        if epoch_rel % save_every == 0 or epoch_rel == epochs:
            checkpoint = {
                'epoch':       epoch_label,
                'model_state': model.state_dict(),
                'ema':         ema.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'scaler':      scaler.state_dict() if use_scaler else None,
                'train_loss':  avg_train_uw,
                'val_loss':    avg_val_loss,
                'config':      phase2_config,
            }
            path = os.path.join(output_dir, f"gff_ddpm_p2_epoch{epoch_label:03d}.pt")
            torch.save(checkpoint, path)
            print(f"  → Saved checkpoint: {path}")

        # ── Save best model ──────────────────────────────────
        if marker:
            best_path = os.path.join(output_dir, "gff_ddpm_best.pt")
            ema.apply(model)
            best_save = {
                'epoch':       epoch_label,
                'model_state': model.state_dict(),
                'config':      phase2_config,
                'val_loss':    avg_val_loss,
                'schedule': {
                    'T': schedule.T,
                    'alpha_bar':                schedule.alpha_bar.cpu(),
                    'sqrt_alpha_bar':           schedule.sqrt_alpha_bar.cpu(),
                    'sqrt_one_minus_alpha_bar': schedule.sqrt_one_minus_alpha_bar.cpu(),
                    'alpha':                    schedule.alpha.cpu(),
                    'beta':                     schedule.beta.cpu(),
                    'posterior_variance':        schedule.posterior_variance.cpu(),
                },
            }
            if norm_mu is not None and norm_sigma is not None:
                best_save['normalization'] = {
                    'mu':    norm_mu,
                    'sigma': norm_sigma,
                }
            torch.save(best_save, best_path)
            ema.restore(model)
            print(f"  → Saved best model: {best_path}")

    # ── Save final EMA model ─────────────────────────────────────
    ema.apply(model)
    final_save = {
        'epoch':       end_label,
        'model_state': model.state_dict(),
        'config':      phase2_config,
        'schedule': {
            'T': schedule.T,
            'alpha_bar':                schedule.alpha_bar.cpu(),
            'sqrt_alpha_bar':           schedule.sqrt_alpha_bar.cpu(),
            'sqrt_one_minus_alpha_bar': schedule.sqrt_one_minus_alpha_bar.cpu(),
            'alpha':                    schedule.alpha.cpu(),
            'beta':                     schedule.beta.cpu(),
            'posterior_variance':        schedule.posterior_variance.cpu(),
        },
    }
    if norm_mu is not None and norm_sigma is not None:
        final_save['normalization'] = {
            'mu':    norm_mu,
            'sigma': norm_sigma,
        }
    final_path = os.path.join(output_dir, "gff_ddpm_final.pt")
    torch.save(final_save, final_path)
    print(f"\nPhase 2 complete. Final EMA model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    ema.restore(model)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Mask-distribution fine-tuning for GFF DDPM")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/gff_ddpm_final.pt",
                        help="Path to Phase 1 final checkpoint")
    parser.add_argument("--data", type=str, default="X_norm.pt",
                        help="Path to normalized training data")
    parser.add_argument("--norm_stats", type=str, default="normalization.pt",
                        help="Path to normalization statistics")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of Phase 2 epochs (default: 50)")
    parser.add_argument("--epoch_offset", type=int, default=100,
                        help="Epoch label offset (Phase 1 ended at this epoch)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Flat learning rate (same as Phase 1)")
    parser.add_argument("--dropout", type=float, default=0.05,
                        help="Dropout (same as Phase 1)")
    parser.add_argument("--snr_gamma", type=float, default=5.0,
                        help="Min-SNR clipping (same as Phase 1)")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="checkpoints_phase2")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    finetune_masks(
        checkpoint_path = args.checkpoint,
        data_path       = args.data,
        norm_stats_path = args.norm_stats,
        epochs          = args.epochs,
        epoch_offset    = args.epoch_offset,
        batch_size      = args.batch_size,
        lr              = args.lr,
        dropout         = args.dropout,
        snr_gamma       = args.snr_gamma,
        val_fraction    = args.val_fraction,
        output_dir      = args.output_dir,
        seed            = args.seed,
    )
#!/usr/bin/env python3
"""
v-Prediction DDPM with ResNet-Attention Hybrid for GFF Reconstruction
======================================================================
Authors: H. Alharazin, J. Yu. Panteleeva

Architecture : 1D ResNet + Self-Attention (constant resolution, no downsampling)
Prediction   : v-prediction  (balanced loss across all noise levels)
Conditioning : Concatenation (mask + condition values as extra input channels)
Schedule     : Cosine        (more capacity at intermediate noise levels)
Precision    : float32       (sufficient for stochastic diffusion; ~10x faster on GPU)

The network receives 3 input channels at every forward pass:
  Channel 0  x_t      noisy GFF curve at diffusion timestep t
  Channel 1  mask     binary vector (1 = known grid point, 0 = unknown)
  Channel 2  x_cond   clean normalized values where mask=1, zero elsewhere

During training, random masks teach the model to reconstruct unknown regions
conditioned on any subset of known values:
  50%  random masks   (5-30 known points anywhere on the grid)
  30%  clustered masks (5-15 known points in the low-|t| region, mimicking lattice data)
  20%  unconditional   (mask = 0 everywhere, for diversity)

v-prediction target:
    v_t  =  sqrt(alpha_bar_t) * eps  -  sqrt(1 - alpha_bar_t) * x_0

From a predicted v, one recovers both x_0 and eps analytically:
    x_0_hat  =  sqrt(alpha_bar_t) * x_t  -  sqrt(1 - alpha_bar_t) * v_hat
    eps_hat  =  sqrt(1 - alpha_bar_t) * x_t  +  sqrt(alpha_bar_t) * v_hat

References:
    Ho et al., NeurIPS 2020           (DDPM)
    Nichol & Dhariwal, ICML 2021      (cosine schedule)
    Salimans & Ho, 2022               (v-prediction)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  1. COSINE NOISE SCHEDULE  (Nichol & Dhariwal, ICML 2021)       ║
# ╚═══════════════════════════════════════════════════════════════════╝

class CosineSchedule:
    """
    Defines alpha_bar_t = f(t)/f(0) with f(t) = cos²((t/T + s)/(1+s) · π/2).

    Key properties:
      - alpha_bar_0  ≈ 1       (no noise at t=0, pure signal)
      - alpha_bar_T  ≈ 0       (pure noise at t=T)
      - Smooth, slow onset of noise at small t  →  the model spends more
        capacity learning structure at intermediate noise levels
      - Offset s=0.008 prevents alpha_bar from reaching exactly 0 or 1

    All derived quantities are precomputed in float64 for numerical
    accuracy, then stored as float32 tensors:
      alpha_bar[t]                     cumulative signal retention     (T+1,)
      sqrt_alpha_bar[t]                                                (T+1,)
      sqrt_one_minus_alpha_bar[t]                                      (T+1,)
      alpha[t]  = alpha_bar[t]/alpha_bar[t-1]                         (T,)
      beta[t]   = 1 - alpha[t]                                        (T,)
      posterior_variance[t]            for DDPM sampling step           (T,)
    """

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T

        # --- Build alpha_bar_t for t = 0, 1, ..., T  (T+1 entries) ---
        # ALL computation in float64, convert to float32 only at the end
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        alpha_bar = torch.clamp(alpha_bar, 1e-5, 0.9999)

        # --- Derived quantities (computed in float64, stored as float32) ---
        sqrt_alpha_bar = alpha_bar.sqrt()                # float64
        sqrt_1m_abar   = (1.0 - alpha_bar).sqrt()       # float64

        self.alpha_bar                = alpha_bar.float()          # (T+1,)
        self.sqrt_alpha_bar           = sqrt_alpha_bar.float()     # (T+1,)
        self.sqrt_one_minus_alpha_bar = sqrt_1m_abar.float()       # (T+1,)

        # alpha_t = alpha_bar_t / alpha_bar_{t-1}   for t = 1, ..., T
        alpha = alpha_bar[1:] / alpha_bar[:-1]                     # float64
        alpha = torch.clamp(alpha, 1e-5, 1.0)
        beta  = 1.0 - alpha                                       # float64

        self.alpha = alpha.float()                                 # (T,)
        self.beta  = beta.float()                                  # (T,)

        # Posterior variance:  beta_tilde_t = beta_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        # Used in DDPM sampling (not during training)
        # Computed entirely in float64 for numerical stability
        alpha_bar_prev = alpha_bar[:-1]   # (T,)  = [ᾱ_0, ᾱ_1, ..., ᾱ_{T-1}]
        alpha_bar_curr = alpha_bar[1:]    # (T,)  = [ᾱ_1, ᾱ_2, ..., ᾱ_T]
        posterior_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr)
        self.posterior_variance = torch.clamp(posterior_var, min=1e-20).float()  # (T,)

    def to(self, device):
        """Move all schedule tensors to the specified device."""
        for name in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar',
                      'alpha', 'beta', 'posterior_variance']:
            setattr(self, name, getattr(self, name).to(device))
        return self


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  2. SINUSOIDAL TIMESTEP EMBEDDING                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

class SinusoidalEmbedding(nn.Module):
    """
    Maps scalar timestep t → vector of dimension `dim` using sinusoidal
    positional encoding (Vaswani et al. 2017, adapted for diffusion).

    The first dim/2 components are sin(t · freq_k), the rest cos(t · freq_k),
    where freq_k = exp(-ln(10000) · k / (dim/2)) for k = 0, ..., dim/2 - 1.

    This gives the network a smooth, high-bandwidth representation of the
    noise level so it can modulate its behavior across the full range of t.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Precompute frequencies (fixed, not learned)
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)   # (dim/2,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (B,) float32 timestep values
        Returns:
            (B, dim) sinusoidal embedding
        """
        args = t[:, None] * self.freqs[None, :]        # (B, dim/2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  3. RESIDUAL BLOCK  (1D convolutions + FiLM time conditioning)  ║
# ╚═══════════════════════════════════════════════════════════════════╝

class ResBlock1D(nn.Module):
    """
    Pre-norm residual block with FiLM conditioning from timestep embedding.

    Architecture:
        GroupNorm → SiLU → Conv1d → FiLM(t) → GroupNorm → SiLU → Dropout → Conv1d
        \\____________________________________________________ + skip ___________/

    FiLM (Feature-wise Linear Modulation):
        The timestep embedding is projected to (scale, shift) vectors.
        After the first convolution:  h = h * (1 + scale) + shift
        This lets the network adapt its behavior to the current noise level.

    The second Conv1d is zero-initialized so the block starts as an identity
    mapping, which stabilizes early training (Goyal et al. 2017).
    """

    def __init__(self, channels: int, time_emb_dim: int,
                 kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2  # same-padding to preserve spatial dimension

        # --- Two conv layers with pre-norm ---
        self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)

        self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout)

        # --- FiLM projection: time_emb → (scale, shift) for `channels` features ---
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * channels),
        )

        # --- Zero-init last conv so block starts as identity ---
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x     : (B, C, L)  feature map
            t_emb : (B, time_emb_dim)  timestep embedding
        Returns:
            (B, C, L)  residual-updated feature map
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)                                              # (B, C, L)

        # FiLM conditioning from timestep
        t = self.time_proj(t_emb)                                      # (B, 2C)
        scale, shift = t.chunk(2, dim=1)                               # each (B, C)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]         # broadcast over L

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)                                              # (B, C, L)

        return x + h   # additive residual — no vanishing gradients


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  4. SELF-ATTENTION BLOCK  (global receptive field)              ║
# ╚═══════════════════════════════════════════════════════════════════╝

class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention over the spatial (grid-point) dimension.

    This gives the network a GLOBAL receptive field: grid point j=0 can
    directly attend to j=199.  Critical for GFF reconstruction because
    physics imposes long-range correlations (e.g., D(0) and the large-t
    tail are related by sum rules).

    Pre-norm (GroupNorm before attention) + zero-initialized output
    projection for residual stability.

    For L=200 grid points and 4 heads, the attention matrix is only
    200×200 — negligible compute compared to the convolutional blocks.
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0, f"channels={channels} not divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv  = nn.Conv1d(channels, 3 * channels, kernel_size=1)
        self.out  = nn.Conv1d(channels, channels, kernel_size=1)

        # Zero-init output so attention starts as no-op
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, L)
        Returns:
            (B, C, L)  with global information mixed across all grid points
        """
        B, C, L = x.shape
        h = self.norm(x)

        # Compute Q, K, V via single pointwise conv
        qkv = self.qkv(h)                                             # (B, 3C, L)
        q, k, v = qkv.chunk(3, dim=1)                                 # each (B, C, L)

        # Reshape for multi-head: (B, heads, L, head_dim)
        q = q.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)
        k = k.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)
        v = v.view(B, self.n_heads, self.head_dim, L).transpose(2, 3)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale      # (B, heads, L, L)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                    # (B, heads, L, head_dim)

        # Reshape back: (B, C, L)
        out = out.transpose(2, 3).contiguous().view(B, C, L)
        out = self.out(out)

        return x + out  # residual


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  5. FULL NETWORK: ResNet-Attention Hybrid                       ║
# ╚═══════════════════════════════════════════════════════════════════╝

class GFFDiffusionNet(nn.Module):
    """
    Complete denoising network for GFF reconstruction.

    Architecture (no downsampling — constant spatial resolution = grid_size):

        Input projection:  Conv1d(3, C, k)            3 channels → C channels
        ┌─────────────────────────────────────┐
        │  [ResBlock1D(C)] × blocks_per_group │
        │  SelfAttention1D(C)                 │  × n_groups
        └─────────────────────────────────────┘
        Output projection: GroupNorm → SiLU → Conv1d(C, 1, k)

    The 3 input channels are:
        [x_t, mask, x_cond]   stacked along dim=1

    Default hyperparameters yield ~16M parameters:
        hidden_dim   = 256    channels throughout the network
        kernel_size  = 7      conv kernel (receptive field 7 per layer)
        n_res_blocks = 12     total residual blocks (4 per group)
        n_groups     = 3      number of attention-separated groups
        n_heads      = 4      attention heads
        dropout      = 0.1
    """

    def __init__(self,
                 grid_size:    int = 200,
                 hidden_dim:   int = 256,
                 kernel_size:  int = 7,
                 n_res_blocks: int = 12,
                 n_groups:     int = 3,
                 n_heads:      int = 4,
                 dropout:    float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        time_emb_dim = hidden_dim * 4   # standard: 4× hidden for time MLP

        # ── Timestep embedding: scalar t → vector ──
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),         # (B,) → (B, hidden_dim)
            nn.Linear(hidden_dim, time_emb_dim),     # → (B, time_emb_dim)
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),   # → (B, time_emb_dim)
        )

        # ── Input: 3 channels (x_t, mask, x_cond) → hidden_dim channels ──
        self.input_proj = nn.Conv1d(3, hidden_dim, kernel_size, padding=kernel_size // 2)

        # ── Core: alternating ResBlocks and Attention ──
        blocks_per_group = n_res_blocks // n_groups
        self.blocks = nn.ModuleList()
        for g in range(n_groups):
            for _ in range(blocks_per_group):
                self.blocks.append(ResBlock1D(hidden_dim, time_emb_dim, kernel_size, dropout))
            self.blocks.append(SelfAttention1D(hidden_dim, n_heads))

        # ── Output: hidden_dim → 1 channel (predicted v) ──
        self.output_norm = nn.GroupNorm(min(32, hidden_dim), hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                mask: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy : (B, grid_size)  noisy GFF curve at timestep t
            t       : (B,)            diffusion timestep (float32)
            mask    : (B, grid_size)  binary mask (1 = known, 0 = unknown)
            x_cond  : (B, grid_size)  clean values at known positions, 0 elsewhere

        Returns:
            v_pred  : (B, grid_size)  predicted v-target
        """
        # Stack 3 input channels: (B, 3, grid_size)
        x = torch.stack([x_noisy, mask, x_cond], dim=1)

        # Input projection
        h = self.input_proj(x)            # (B, C, grid_size)

        # Timestep embedding (shared across all blocks)
        t_emb = self.time_mlp(t)          # (B, time_emb_dim)

        # Process through ResBlocks + Attention
        for block in self.blocks:
            if isinstance(block, ResBlock1D):
                h = block(h, t_emb)
            else:  # SelfAttention1D
                h = block(h)

        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_proj(h)           # (B, 1, grid_size)

        return h.squeeze(1)               # (B, grid_size)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  6. EXPONENTIAL MOVING AVERAGE (EMA) of model weights           ║
# ╚═══════════════════════════════════════════════════════════════════╝

class EMA:
    """
    Maintains an exponential moving average of model parameters.

    EMA is essential for diffusion models: the averaged weights produce
    significantly higher-quality samples than the raw training weights.
    Typical decay = 0.9999 (update: shadow ← decay·shadow + (1-decay)·param).

    Usage:
        ema = EMA(model)
        # After each optimizer step:
        ema.update(model)
        # For evaluation/sampling:
        ema.apply(model)    # load EMA weights into model
        ...sample...
        ema.restore(model)  # restore training weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA params. Call restore() to undo."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original (non-EMA) params after apply()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  7. v-PREDICTION DDPM  (diffusion process + loss)               ║
# ╚═══════════════════════════════════════════════════════════════════╝

class VPredictionDDPM:
    """
    Wraps the noise schedule, forward process, conditioning, and loss.

    Forward process (adding noise):
        x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 - ᾱ_t) · ε

    v-prediction target:
        v_t = sqrt(ᾱ_t) · ε  -  sqrt(1 - ᾱ_t) · x_0

    Conditioning via random masking during training:
        50%  random masks:    5-30 known points anywhere on the grid
        30%  clustered masks: 5-15 known points in the low-|t| region (first 30% of grid)
        20%  unconditional:   mask = 0 everywhere (for diversity)

    The clustered masks mimic the actual inference scenario where lattice data
    points are concentrated at low -t values (0 to ~2 GeV²).
    """

    def __init__(self, model: GFFDiffusionNet, schedule: CosineSchedule, device: torch.device):
        self.model    = model
        self.schedule = schedule.to(device)
        self.device   = device

    # ── Forward process: add noise to clean data ──────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t)·x_0, (1-ᾱ_t)·I).

        Args:
            x0    : (B, L) clean normalized GFF
            t     : (B,)   timesteps (integers in [1, T])
            noise : (B, L) optional pre-sampled noise
        Returns:
            x_t   : (B, L) noisy version of x0
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_abar    = self.schedule.sqrt_alpha_bar[t][:, None]            # (B, 1)
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]  # (B, 1)

        return sqrt_abar * x0 + sqrt_1m_abar * noise

    # ── Compute v-target ──────────────────────────────────────────

    def compute_v_target(self, x0: torch.Tensor, noise: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        v_t = sqrt(ᾱ_t) · ε  -  sqrt(1 - ᾱ_t) · x_0
        """
        sqrt_abar    = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]

        return sqrt_abar * noise - sqrt_1m_abar * x0

    # ── Recover x0 and eps from v-prediction ─────────────────────

    def predict_x0_from_v(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """
        x̂_0 = sqrt(ᾱ_t) · x_t  -  sqrt(1 - ᾱ_t) · v̂
        Used during sampling and for gradient guidance.
        """
        sqrt_abar    = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]

        return sqrt_abar * x_t - sqrt_1m_abar * v_pred

    def predict_eps_from_v(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
        """
        ε̂ = sqrt(1 - ᾱ_t) · x_t  +  sqrt(ᾱ_t) · v̂
        Used during sampling to compute the DDPM update step.
        """
        sqrt_abar    = self.schedule.sqrt_alpha_bar[t][:, None]
        sqrt_1m_abar = self.schedule.sqrt_one_minus_alpha_bar[t][:, None]

        return sqrt_1m_abar * x_t + sqrt_abar * v_pred

    # ── Random conditioning mask ─────────────────────────────────

    def random_mask(self, x0: torch.Tensor) -> tuple:
        """
        Generate random conditioning masks for training.

        Distribution over mask types:
          50%  random:      5-30 known points uniformly across the grid
          30%  clustered:   5-15 known points in the first 30% of the grid
                            (mimics lattice data concentrated at low -t)
          20%  unconditional: mask = all zeros

        Uses explicit loops over batch samples with torch.randperm to
        guarantee correct mask generation.  The loop is negligible in
        cost (~0.5ms for B=256) compared to the neural network forward
        pass (~50-200ms).

        Args:
            x0 : (B, L) clean data (used for shape and for x_cond values)
        Returns:
            mask   : (B, L) binary, 1 = known
            x_cond : (B, L) clean values where mask=1, zero elsewhere
        """
        B, L = x0.shape
        device = x0.device

        # --- Decide mask type per sample ---
        r = torch.rand(B, device=device)
        is_random    = (r < 0.5)                     # 50%
        is_clustered = (r >= 0.5) & (r < 0.8)       # 30%
        # remaining 20%: unconditional (mask stays zero)

        # --- Initialize mask to zeros ---
        mask = torch.zeros(B, L, device=device)

        # --- Random masks: 5-30 known points anywhere on the grid ---
        idx_r = is_random.nonzero(as_tuple=True)[0]
        if idx_r.numel() > 0:
            for i in idx_r:
                n = torch.randint(5, 31, (1,), device=device).item()
                perm = torch.randperm(L, device=device)[:n]
                mask[i, perm] = 1.0

        # --- Clustered masks: 5-15 known points in first 30% of grid ---
        idx_c = is_clustered.nonzero(as_tuple=True)[0]
        if idx_c.numel() > 0:
            low_t_end = max(int(0.3 * L), 15)  # first 30% of grid points
            for i in idx_c:
                n = torch.randint(5, 16, (1,), device=device).item()
                perm = torch.randperm(low_t_end, device=device)[:n]
                mask[i, perm] = 1.0

        # --- Unconditional (20%): mask stays zero (already initialized) ---

        # --- Condition values: clean data where mask=1, zero elsewhere ---
        x_cond = x0 * mask

        return mask, x_cond

    # ── Single training step ─────────────────────────────────────

    def training_step(self, x0: torch.Tensor) -> torch.Tensor:
        """
        One training iteration. Returns scalar MSE loss.

        Steps:
          1. Sample random timestep t ~ Uniform{1, ..., T} for each element
          2. Sample noise ε ~ N(0, I)
          3. Compute x_t via forward process
          4. Generate random conditioning mask
          5. Compute v-target
          6. Network predicts v from (x_t, t, mask, x_cond)
          7. Loss = MSE(v_pred, v_target)  averaged over batch and grid
        """
        B = x0.shape[0]

        # 1. Random timesteps
        t = torch.randint(1, self.schedule.T + 1, (B,), device=self.device)

        # 2. Sample noise
        noise = torch.randn_like(x0)

        # 3. Forward process
        x_t = self.q_sample(x0, t, noise)

        # 4. Random conditioning mask
        mask, x_cond = self.random_mask(x0)

        # 5. v-target
        v_target = self.compute_v_target(x0, noise, t)

        # 6. Network prediction (t passed as float for sinusoidal embedding)
        v_pred = self.model(x_t, t.float(), mask, x_cond)

        # 7. MSE loss over all grid points
        loss = F.mse_loss(v_pred, v_target)
        return loss


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  8. LEARNING RATE SCHEDULE WITH WARMUP                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

def get_lr_lambda(warmup_steps: int, total_steps: int):
    """
    Linear warmup for `warmup_steps`, then cosine decay to zero.

    Warmup prevents large gradients in attention layers during
    early training when Q, K projections are still random.

    Args:
        warmup_steps : number of optimizer steps for linear warmup
        total_steps  : total number of optimizer steps (epochs × batches_per_epoch)
    Returns:
        lr_lambda function for torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  9. TRAINING LOOP                                               ║
# ╚═══════════════════════════════════════════════════════════════════╝

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(
    #data_path:       str   = "Training/X_norm.pt",
    data_path:       str   = "X_norm.pt",
    norm_stats_path: str   = "normalization.pt",
    #norm_mu_path:    str   = "Training/mu.pt",
    #norm_sigma_path: str   = "Training/sigma.pt",
    epochs:          int   = 100,
    batch_size:      int   = 256,
    lr:              float = 1e-4,
    weight_decay:    float = 1e-4,
    grad_clip:       float = 1.0,
    ema_decay:       float = 0.9999,
    T:               int   = 1000,
    warmup_epochs:   int   = 2,
    val_fraction:    float = 0.01,
    save_every:      int   = 10,
    use_amp:         bool  = True,
    checkpoint_dir:  str   = "checkpoints",
    resume_from:     str   = None,
    seed:            int   = 42,
):
    """
    Full training procedure.

    Features:
      - Reproducible via manual seed
      - AdamW optimizer (decoupled weight decay)
      - Linear warmup + cosine decay LR schedule (per step, not per epoch)
      - Gradient clipping at norm=1.0
      - EMA with decay=0.9999 (critical for sample quality)
      - Automatic Mixed Precision (AMP) with bfloat16 for ~2× speedup
      - cuDNN benchmark for auto-tuned convolution algorithms (~10-20% speedup)
      - Validation split (1%) for overfitting detection
      - Full checkpoints with EMA state for resumption
      - CSV training log for easy plotting and paper figures
      - Normalization statistics (mu, sigma) bundled into checkpoint for inference
    """

    # ── Reproducibility ──────────────────────────────────────────
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Setup ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms
        print(f"  cuDNN benchmark: enabled")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Determine AMP dtype ──────────────────────────────────────
    # Use bfloat16 if available (Ampere+), otherwise float16
    if use_amp and device.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print(f"  AMP: bfloat16 (native)")
        else:
            amp_dtype = torch.float16
            print(f"  AMP: float16 (with GradScaler)")
    else:
        use_amp = False
        amp_dtype = torch.float32
        print(f"  AMP: disabled")

    # GradScaler only needed for float16 (bfloat16 doesn't need it)
    use_scaler = use_amp and (amp_dtype == torch.float16)

    # ── Load data ────────────────────────────────────────────────
    print(f"\nLoading data from {data_path}...")
    X_all = torch.load(data_path, weights_only=True).float()  # convert to float32
    print(f"  Dataset shape: {X_all.shape}")                   # (600000, 200)
    print(f"  Dtype: {X_all.dtype}")
    print(f"  Value range:   [{X_all.min():.3f}, {X_all.max():.3f}]")
    print(f"  Mean (≈0): {X_all.mean():.4f}")
    print(f"  Std  (≈1): {X_all.std():.4f}")

    # ── Load normalization statistics (for bundling into checkpoint) ──
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
        print(f"           You will need to load mu/sigma separately at inference time.")
    # ── Train/val split ──────────────────────────────────────────
    N = X_all.shape[0]
    n_val   = max(int(N * val_fraction), batch_size)  # at least one batch
    n_train = N - n_val

    # Shuffle before splitting (data may still be ordered by family)
    perm = torch.randperm(N)
    X_train = X_all[perm[:n_train]]
    X_val   = X_all[perm[n_train:]]
    del X_all  # free memory

    print(f"  Train: {n_train}   Val: {n_val}")

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(TensorDataset(X_val), batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True,
                              drop_last=False)

    # ── Initialize model, schedule, diffusion, EMA ───────────────
    grid_size = X_train.shape[1]
    schedule  = CosineSchedule(T=T)
    model     = GFFDiffusionNet(grid_size=grid_size).to(device)
    diffusion = VPredictionDDPM(model, schedule, device)
    ema       = EMA(model, decay=ema_decay)

    n_params = count_parameters(model)
    print(f"\n  Model parameters: {n_params:,}  ({n_params/1e6:.2f}M)")
    print(f"  Grid size: {grid_size}")

    # ── Optimizer and LR schedule ────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    print(f"  Steps/epoch: {steps_per_epoch}   Total steps: {total_steps}   Warmup steps: {warmup_steps}")

    lr_lambda = get_lr_lambda(warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # GradScaler for float16 AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ── Resume from checkpoint ───────────────────────────────────
    start_epoch = 1
    global_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n  Resuming from {resume_from}...")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        if use_scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', (start_epoch - 1) * steps_per_epoch)
        print(f"  Resumed at epoch {start_epoch}, global step {global_step}")

    # ── Config dict (for reproducibility) ────────────────────────
    config = {
        'grid_size':     grid_size,
        'hidden_dim':    256,
        'kernel_size':   7,
        'n_res_blocks':  12,
        'n_groups':      3,
        'n_heads':       4,
        'T':             T,
        'epochs':        epochs,
        'batch_size':    batch_size,
        'lr':            lr,
        'weight_decay':  weight_decay,
        'ema_decay':     ema_decay,
        'warmup_epochs': warmup_epochs,
        'amp_dtype':     str(amp_dtype),
        'seed':          seed,
    }

    # ── CSV training log ─────────────────────────────────────────
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr\n")

    # ── Training loop ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Starting training for {epochs} epochs (from epoch {start_epoch})")
    print(f"  Batch size: {batch_size}   Diffusion steps T: {T}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for (x0_batch,) in train_loader:
            x0_batch = x0_batch.to(device, non_blocking=True)

            # ── Forward + loss (with optional AMP) ───────────
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = diffusion.training_step(x0_batch)

            # ── Backward + optimizer step ────────────────────
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            scheduler.step()   # per-step LR update (not per-epoch)

            total_loss += loss.item()
            n_batches  += 1
            global_step += 1

        avg_train_loss = total_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # ── Validation loss (using EMA weights) ──────────────
        val_loss = 0.0
        n_val_batches = 0
        ema.apply(model)
        model.eval()
        with torch.no_grad():
            for (x0_val,) in val_loader:
                x0_val = x0_val.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    vl = diffusion.training_step(x0_val)
                val_loss += vl.item()
                n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)
        ema.restore(model)

        # ── Logging ──────────────────────────────────────────
        marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            marker = "  ★ best"

        print(f"Epoch {epoch:3d}/{epochs}  |  "
              f"Train: {avg_train_loss:.6f}  |  "
              f"Val: {avg_val_loss:.6f}  |  "
              f"LR: {current_lr:.2e}{marker}")

        # ── Append to CSV log ────────────────────────────────
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.8f},{avg_val_loss:.8f},{current_lr:.2e}\n")

        # ── Save checkpoint ──────────────────────────────────
        if epoch % save_every == 0 or epoch == epochs:
            checkpoint = {
                'epoch':       epoch,
                'global_step': global_step,
                'model_state': model.state_dict(),   # training weights
                'ema':         ema.state_dict(),      # EMA weights
                'optimizer':   optimizer.state_dict(),
                'scheduler':   scheduler.state_dict(),
                'scaler':      scaler.state_dict() if use_scaler else None,
                'train_loss':  avg_train_loss,
                'val_loss':    avg_val_loss,
                'config':      config,
            }
            path = os.path.join(checkpoint_dir, f"gff_ddpm_epoch{epoch:03d}.pt")
            torch.save(checkpoint, path)
            print(f"  → Saved checkpoint: {path}")

        # ── Save best model separately ───────────────────────
        if marker:  # new best val loss
            best_path = os.path.join(checkpoint_dir, "gff_ddpm_best.pt")
            ema.apply(model)
            best_save = {
                'epoch':       epoch,
                'model_state': model.state_dict(),  # EMA weights
                'config':      config,
                'val_loss':    avg_val_loss,
                'schedule': {
                    'T': schedule.T,
                    'alpha_bar':                schedule.alpha_bar.cpu(),
                    'sqrt_alpha_bar':           schedule.sqrt_alpha_bar.cpu(),
                    'sqrt_one_minus_alpha_bar': schedule.sqrt_one_minus_alpha_bar.cpu(),
                    'alpha':                    schedule.alpha.cpu(),
                    'beta':                     schedule.beta.cpu(),
                    'posterior_variance':       schedule.posterior_variance.cpu(),
                },
            }
            # Bundle normalization statistics if available
            if norm_mu is not None and norm_sigma is not None:
                best_save['normalization'] = {
                    'mu':    norm_mu,       # (grid_size,)
                    'sigma': norm_sigma,    # (grid_size,)
                }
            torch.save(best_save, best_path)
            ema.restore(model)
            print(f"  → Saved best model: {best_path}")

    # ── Save final EMA model (for inference) ─────────────────────
    ema.apply(model)
    final_save = {
        'epoch':       epochs,
        'model_state': model.state_dict(),   # these ARE the EMA weights
        'config':      config,
        'schedule': {                         # save schedule for inference
            'T': schedule.T,
            'alpha_bar':                schedule.alpha_bar.cpu(),
            'sqrt_alpha_bar':           schedule.sqrt_alpha_bar.cpu(),
            'sqrt_one_minus_alpha_bar': schedule.sqrt_one_minus_alpha_bar.cpu(),
            'alpha':                    schedule.alpha.cpu(),
            'beta':                     schedule.beta.cpu(),
            'posterior_variance':       schedule.posterior_variance.cpu(),
        },
    }
    # Bundle normalization statistics if available
    if norm_mu is not None and norm_sigma is not None:
        final_save['normalization'] = {
            'mu':    norm_mu,       # (grid_size,)
            'sigma': norm_sigma,    # (grid_size,)
        }
    final_path = os.path.join(checkpoint_dir, "gff_ddpm_final.pt")
    torch.save(final_save, final_path)
    print(f"\nTraining complete. Final EMA model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    ema.restore(model)


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  10. LOAD MODEL FOR INFERENCE                                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def load_model(checkpoint_path: str, device: torch.device = None):
    """
    Load a trained GFF diffusion model from checkpoint for inference.

    Works with:
      - "gff_ddpm_final.pt"      (final EMA weights + schedule)
      - "gff_ddpm_best.pt"       (best EMA weights + schedule)
      - "gff_ddpm_epochNNN.pt"   (epoch checkpoint, extracts EMA weights)

    Usage:
    -------
        # ── Load ──
        model, schedule, config, norm = load_model("checkpoints/gff_ddpm_best.pt")

        # ── Use for a single forward pass ──
        with torch.no_grad():
            v_pred = model(x_noisy, t, mask, x_cond)

        # ── Or wrap in VPredictionDDPM for full sampling ──
        diffusion = VPredictionDDPM(model, schedule, device)
        x0_hat = diffusion.predict_x0_from_v(x_t, v_pred, t)

        # ── De-normalize to physical units ──
        if norm is not None:
            D_physical = x0_hat * norm['sigma'].to(device) + norm['mu'].to(device)

    Args:
        checkpoint_path : path to .pt checkpoint file
        device          : torch device (default: cuda if available, else cpu)

    Returns:
        model    : GFFDiffusionNet with EMA weights loaded, in eval mode
        schedule : CosineSchedule on the correct device
        config   : dict of hyperparameters used during training
        norm     : dict with 'mu' and 'sigma' tensors, or None if not stored
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Extract config ───────────────────────────────────────────
    config = ckpt['config']
    print(f"  Config: grid_size={config['grid_size']}, "
          f"hidden_dim={config['hidden_dim']}, "
          f"n_res_blocks={config['n_res_blocks']}, "
          f"T={config['T']}")

    # ── Rebuild model from config ────────────────────────────────
    model = GFFDiffusionNet(
        grid_size    = config['grid_size'],
        hidden_dim   = config['hidden_dim'],
        kernel_size  = config['kernel_size'],
        n_res_blocks = config['n_res_blocks'],
        n_groups     = config['n_groups'],
        n_heads      = config['n_heads'],
        dropout      = 0.0,    # no dropout at inference
    ).to(device)

    # ── Load weights ─────────────────────────────────────────────
    if 'ema' in ckpt and 'schedule' not in ckpt:
        # Epoch checkpoint: extract EMA weights from shadow dict
        print("  Loading EMA weights from epoch checkpoint...")
        ema_shadow = ckpt['ema']['shadow']
        # Start from the full model_state (includes buffers like freqs)
        state_dict = ckpt['model_state'].copy()
        # Overwrite trainable params with EMA versions
        for name in ema_shadow:
            state_dict[name] = ema_shadow[name]
        model.load_state_dict(state_dict)
    else:
        # Final or best checkpoint: model_state IS the EMA weights
        model.load_state_dict(ckpt['model_state'])

    model.eval()

    # ── Rebuild or load schedule ─────────────────────────────────
    T = config['T']
    if 'schedule' in ckpt:
        # Load pre-saved schedule tensors (avoids recomputation)
        schedule = CosineSchedule(T=T)
        for key in ['alpha_bar', 'sqrt_alpha_bar', 'sqrt_one_minus_alpha_bar',
                     'alpha', 'beta', 'posterior_variance']:
            setattr(schedule, key, ckpt['schedule'][key])
        schedule.to(device)
    else:
        # Recompute (deterministic, gives identical result)
        schedule = CosineSchedule(T=T)
        schedule.to(device)

    # ── Load normalization statistics ────────────────────────────
    norm = None
    if 'normalization' in ckpt:
        norm = ckpt['normalization']
        print(f"  Normalization: mu {norm['mu'].shape}, sigma {norm['sigma'].shape}")
    else:
        print(f"  WARNING: No normalization statistics in checkpoint.")
        print(f"           Load mu/sigma separately for de-normalization.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters ({n_params/1e6:.2f}M)")
    print(f"  Device: {device}")

    if 'val_loss' in ckpt:
        print(f"  Val loss at save: {ckpt['val_loss']:.6f}")
    if 'epoch' in ckpt:
        print(f"  Trained for {ckpt['epoch']} epochs")

    print(f"  Ready for inference.\n")
    return model, schedule, config, norm


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                    ║
# ╚═══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    train()
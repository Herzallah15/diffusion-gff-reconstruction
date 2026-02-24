import torch
import random
from FFs_Classes import *

"""
Authors: H. Alharazin and J. Yu. Panteleeva
"""


# Meson Dominance / Sum of Poles
# Physical meson masses in GeV
MESON_MASSES = {
    # Scalars (0++)
    'f0_500': 0.5,      # f_0(500) - sigma
    'f0_980': 0.98,     # f_0(980)
    'f0_1370': 1.37,    # f_0(1370)
    'f0_1500': 1.50,    # f_0(1500)
    'f0_1710': 1.72,    # f_0(1710)
    'f0_2020': 2.02,    # f_0(2020)
    'f0_2100': 2.10,    # f_0(2100)
    'f0_2200': 2.20,    # f_0(2200)
    # Tensors (2++)
    'f2_1270': 1.275,   # f_2(1270)
    'f2_1430': 1.430,   # f_2'(1525)
    'f2_1525': 1.525,   # f_2(1525)
    'f2_1640': 1.639,   # f_2(1640)
    'f2_1810': 1.815,   # f_2(1810)
    'f2_1950': 1.944,   # f_2(1950)
    'f2_2010': 2.011,   # f_2(2010)
    'f2_2300': 2.297,   # f_2(2300)
    'f2_2340': 2.34,    # f_2(2340)
}
meson_mass_list = list(MESON_MASSES.values())


# ═════════════════════════════════════════════════════════════════════
#  Raw coefficient sampling — 9 strategies
#  Returns tensor of shape (k_max,)
# ═════════════════════════════════════════════════════════════════════

def sample_coefficients(k_max):
    strategy = random.choice([
        'gaussian_natural',
        'gaussian_wide',
        'uniform',
        'decaying',
        'growing',
        'alternating',
        'sparse',
        'log_uniform',
        'dominated_a0',
    ])

    if strategy == 'gaussian_natural':
        a = torch.randn(k_max, dtype=torch.float64)

    elif strategy == 'gaussian_wide':
        sigma = 1.0 + 4.0 * torch.rand(1, dtype=torch.float64).item()
        a = torch.randn(k_max, dtype=torch.float64) * sigma

    elif strategy == 'uniform':
        a = (torch.rand(k_max, dtype=torch.float64) * 2 - 1) * 5.0

    elif strategy == 'decaying':
        r = 0.3 + 0.6 * torch.rand(1, dtype=torch.float64).item()
        decay = torch.tensor([r ** k for k in range(k_max)], dtype=torch.float64)
        a = torch.randn(k_max, dtype=torch.float64) * decay * 3.0

    elif strategy == 'growing':
        envelope = torch.tensor([1.0 + 0.3 * k for k in range(k_max)],
                                dtype=torch.float64)
        a = torch.randn(k_max, dtype=torch.float64) * envelope
        a = torch.clamp(a, -10.0, 10.0)

    elif strategy == 'alternating':
        signs = torch.tensor([(-1.0) ** k for k in range(k_max)],
                             dtype=torch.float64)
        magnitudes = torch.abs(torch.randn(k_max, dtype=torch.float64)) * 2.0
        a = signs * magnitudes

    elif strategy == 'sparse':
        a = torch.zeros(k_max, dtype=torch.float64)
        n_nonzero = random.randint(1, max(2, k_max // 2))
        idx = random.sample(range(k_max), n_nonzero)
        a[idx] = torch.randn(n_nonzero, dtype=torch.float64) * 3.0

    elif strategy == 'log_uniform':
        log_mag = torch.rand(k_max, dtype=torch.float64) * 2 - 1
        signs = 2.0 * torch.randint(0, 2, (k_max,)).to(torch.float64) - 1.0
        a = signs * 10.0 ** log_mag

    elif strategy == 'dominated_a0':
        a = torch.randn(k_max, dtype=torch.float64) * 0.5
        a[0] = (5.0 + 5.0 * torch.rand(1, dtype=torch.float64)).item()
        a[0] *= random.choice([-1.0, 1.0])

    return a




# ═════════════════════════════════════════════════════════════════════
#  Compensated coefficient sampling
#
#  Problem:  |z_max| ~ 0.39  ⟹  z^k ~ 0.39^k  ⟹  terms k >= 3
#            are invisible and the curve looks linear.
#
#  Fix:      Multiply a_k by 1/|z_max|^k so each term a_k * z^k
#            contributes O(1) to the shape.
#
#  Note:     a[0] will be overwritten in Z_Expansion to hit the
#            target FF(0), so compensation on a[0] is irrelevant.
# ═════════════════════════════════════════════════════════════════════

def sample_coefficients_compensated(k_max, z_max):
    a_raw = sample_coefficients(k_max)

    compensation = torch.tensor(
        [1.0 / max(abs(z_max), 1e-6) ** k for k in range(k_max)],
        dtype=torch.float64
    )
    # Cap to avoid absurdly large coefficients at high k
    compensation = torch.clamp(compensation, max=1000.0)

    strategy = random.choice([
        'full',      # all orders contribute equally
        'partial',   # random interpolation between none and full
        'sqrt'      # intermediate
    ])

    if strategy == 'full':
        a = a_raw * compensation
    elif strategy == 'partial':
        alpha = 0.3 + 0.5 * torch.rand(1, dtype=torch.float64).item()
        a = a_raw * compensation ** alpha
    elif strategy == 'sqrt':
        a = a_raw * compensation ** 0.5
    else:
        a = a_raw

    return a


# ═════════════════════════════════════════════════════════════════════
#  FF(0) target sampler — diverse distributions spanning [-10, 10]
# ═════════════════════════════════════════════════════════════════════
def sample_ff0_target():
    strategy = random.choice([
        'uniform',
        'concentrated_zero',
        'extremes',
        'moderate_negative',
        'moderate_positive',
    ])

    if strategy == 'uniform':
        return (torch.rand(1, dtype=torch.float64) * 20.0 - 10.0).item()
    elif strategy == 'concentrated_zero':
        val = (torch.randn(1, dtype=torch.float64) * 3.0).item()
        return float(np.clip(val, -10.0, 10.0))
    elif strategy == 'extremes':
        sign = random.choice([-1.0, 1.0])
        return sign * (7.0 + 3.0 * torch.rand(1, dtype=torch.float64).item())
    elif strategy == 'moderate_negative':
        return -(0.5 + 7.5 * torch.rand(1, dtype=torch.float64).item())
    elif strategy == 'moderate_positive':
        return (0.5 + 7.5 * torch.rand(1, dtype=torch.float64).item())



MESON_MASS_LIST = list(MESON_MASSES.values())

# Minimum allowed pole mass — keeps denominators m_i^2 - t away from
# zero on the spacelike grid  t ∈ [0, −2] GeV^2.
MIN_MASS = 0.40          # GeV
MIN_MASS_SEP = 0.05      # GeV  – minimum separation between poles


# ═════════════════════════════════════════════════════════════════════
#  Helper: FF(0) target sampler  (reuse if already defined elsewhere)
# ═════════════════════════════════════════════════════════════════════
def _sample_ff0_target():
    strategy = random.choice([
        'uniform', 'concentrated_zero', 'extremes',
        'moderate_negative', 'moderate_positive',
    ])
    if strategy == 'uniform':
        return (torch.rand(1, dtype=torch.float64) * 20.0 - 10.0).item()
    elif strategy == 'concentrated_zero':
        val = (torch.randn(1, dtype=torch.float64) * 3.0).item()
        return float(np.clip(val, -10.0, 10.0))
    elif strategy == 'extremes':
        sign = random.choice([-1.0, 1.0])
        return sign * (7.0 + 3.0 * torch.rand(1, dtype=torch.float64).item())
    elif strategy == 'moderate_negative':
        return -(0.5 + 7.5 * torch.rand(1, dtype=torch.float64).item())
    else:  # moderate_positive
        return (0.5 + 7.5 * torch.rand(1, dtype=torch.float64).item())


# ═════════════════════════════════════════════════════════════════════
#  Helper: sample pole masses from / near physical meson states
# ═════════════════════════════════════════════════════════════════════
def _sample_pole_masses(k_max):
    """
    Return a sorted list of k_max pole masses (GeV), each ≥ MIN_MASS
    and mutually separated by ≥ MIN_MASS_SEP.
    """
    strategy = random.choice([
        'physical_exact',       # draw directly from MESON_MASSES
        'physical_jittered',    # physical center + Gaussian jitter
        'scalar_only',          # only 0++ states
        'tensor_only',          # only 2++ states
        'mixed_free',           # some physical, some freely placed
        'log_uniform',          # log-uniform in [MIN_MASS, 2.5]
    ])

    m = torch.zeros(k_max, dtype=torch.float64)

    if strategy == 'physical_exact' and k_max <= len(MESON_MASS_LIST):
        selected = random.sample(MESON_MASS_LIST, k_max)
        m = torch.tensor(selected, dtype=torch.float64)

    elif strategy == 'physical_jittered':
        jitter = 0.02 + 0.10 * random.random()          # 20–120 MeV
        for j in range(k_max):
            m[j] = random.choice(MESON_MASS_LIST) + torch.randn(1).item() * jitter

    elif strategy == 'scalar_only':
        scalars = [v for k, v in MESON_MASSES.items() if k.startswith('f0')]
        for j in range(k_max):
            m[j] = random.choice(scalars) + torch.randn(1).item() * 0.05

    elif strategy == 'tensor_only':
        tensors = [v for k, v in MESON_MASSES.items() if k.startswith('f2')]
        for j in range(k_max):
            m[j] = random.choice(tensors) + torch.randn(1).item() * 0.05

    elif strategy == 'mixed_free':
        n_phys = random.randint(1, k_max)
        for j in range(n_phys):
            m[j] = random.choice(MESON_MASS_LIST) + torch.randn(1).item() * 0.04
        for j in range(n_phys, k_max):
            m[j] = 0.5 + 2.0 * random.random()

    else:  # log_uniform
        log_lo, log_hi = np.log(MIN_MASS), np.log(2.5)
        m = torch.exp(torch.rand(k_max, dtype=torch.float64) * (log_hi - log_lo) + log_lo)

    # ── Enforce minimum mass ──
    m = torch.clamp(m.abs(), min=MIN_MASS)

    # ── Sort and enforce minimum separation ──
    m, _ = m.sort()
    for j in range(1, k_max):
        if m[j] - m[j - 1] < MIN_MASS_SEP:
            m[j] = m[j - 1] + MIN_MASS_SEP + 0.02 * random.random()

    return m.tolist()


# ═════════════════════════════════════════════════════════════════════
#  Helper: sample free residues  (before superconvergence projection)
# ═════════════════════════════════════════════════════════════════════
def _sample_free_residues(n):
    """Return n residues drawn from one of several strategies."""
    strategy = random.choice([
        'gaussian',
        'gaussian_wide',
        'uniform',
        'alternating',
        'log_uniform',
        'decaying',
        'sparse',
    ])

    if strategy == 'gaussian':
        c = torch.randn(n, dtype=torch.float64) * 1.5

    elif strategy == 'gaussian_wide':
        sigma = 2.0 + 3.0 * random.random()
        c = torch.randn(n, dtype=torch.float64) * sigma

    elif strategy == 'uniform':
        c = (torch.rand(n, dtype=torch.float64) * 2 - 1) * 4.0

    elif strategy == 'alternating':
        signs = torch.tensor([(-1.0) ** j for j in range(n)], dtype=torch.float64)
        c = signs * torch.abs(torch.randn(n, dtype=torch.float64)) * 2.0

    elif strategy == 'log_uniform':
        mags = 10.0 ** (torch.rand(n, dtype=torch.float64) * 1.5 - 0.5)
        signs = 2.0 * torch.randint(0, 2, (n,)).to(torch.float64) - 1.0
        c = mags * signs

    elif strategy == 'decaying':
        r = 0.3 + 0.5 * random.random()
        env = torch.tensor([r ** j for j in range(n)], dtype=torch.float64)
        c = torch.randn(n, dtype=torch.float64) * env * 3.0

    else:  # sparse
        c = torch.zeros(n, dtype=torch.float64)
        n_nz = random.randint(1, max(1, n))
        idx = random.sample(range(n), n_nz)
        c[idx] = torch.randn(n_nz, dtype=torch.float64) * 2.5

    return c.tolist()


# ═════════════════════════════════════════════════════════════════════
#  Helper: build residues with optional superconvergence
#
#  Sum rules (Broniowski & Ruiz Arriola, PRD 111, 074017):
#      Σ c_i       = 0        (1st superconvergence)
#      Σ c_i m_i²  = 0        (2nd superconvergence)
#
#  These guarantee D(t) ~ 1/(-t)^2 at large -t.
#  Requires N ≥ 3 (N = 2 is degenerate).
#
#  Method: freely sample c_1 … c_{N-2}, then solve a 2×2 linear
#          system for c_{N-1}, c_N.
# ═════════════════════════════════════════════════════════════════════
def _sample_residues(k_max, masses, enforce_superconvergence):
    """
    Parameters
    ----------
    k_max : int
        Number of poles.
    masses : list[float]
        Pole masses in GeV (length k_max, sorted).
    enforce_superconvergence : bool
        If True and k_max ≥ 3, project residues onto the
        superconvergence subspace.

    Returns
    -------
    list[float] or None   (None signals rejection)
    """
    # ── No superconvergence: sample all residues freely ──
    if not enforce_superconvergence or k_max < 3:
        return _sample_free_residues(k_max)

    # ── Superconvergence: sample N-2 free, solve for last two ──
    n_free = k_max - 2
    c_free = _sample_free_residues(n_free)

    m2 = [mi ** 2 for mi in masses]

    S0 = sum(c_free)                                        # Σ c_i
    S2 = sum(c_free[j] * m2[j] for j in range(n_free))     # Σ c_i m_i²

    det = m2[-1] - m2[-2]          # m_N² − m_{N-1}²
    if abs(det) < 1e-8:
        return None                # degenerate masses → reject

    c_last        = (-S2 + S0 * m2[-2]) / det
    c_second_last = -S0 - c_last

    # Guard: reject if constrained residues blow up
    if abs(c_last) > 200.0 or abs(c_second_last) > 200.0:
        return None

    return c_free + [c_second_last, c_last]






# ═════════════════════════════════════════════════════════════════════
#  Lambda sampler — mass / energy scale  (GeV)
# ═════════════════════════════════════════════════════════════════════
def _sample_lambda():
    strategy = random.choice([
        'soft',              # light scales  ~ chiral / sigma
        'medium',            # typical hadronic
        'hard',              # heavy resonance / pQCD onset
        'meson_inspired',    # near physical meson masses
        'log_uniform',       # broad log-uniform coverage
        'narrow_band',       # concentrated around 1 GeV
        'very_hard',         # up to 4 GeV for slow falloff
    ])

    if strategy == 'soft':
        L = 0.3 + 0.5 * random.random()            # [0.3, 0.8]
    elif strategy == 'medium':
        L = 0.8 + 0.7 * random.random()             # [0.8, 1.5]
    elif strategy == 'hard':
        L = 1.5 + 1.0 * random.random()             # [1.5, 2.5]
    elif strategy == 'meson_inspired':
        L = random.choice([0.50, 0.77, 0.98, 1.27, 1.52, 1.72, 2.0])
        L += torch.randn(1).item() * 0.08
    elif strategy == 'log_uniform':
        L = np.exp(np.random.uniform(np.log(0.3), np.log(3.5)))
    elif strategy == 'narrow_band':
        L = 0.9 + 0.2 * random.random()             # [0.9, 1.1]
    else:  # very_hard
        L = 2.5 + 1.5 * random.random()             # [2.5, 4.0]

    return max(abs(L), 0.25)   # floor at 250 MeV


# ═════════════════════════════════════════════════════════════════════
#  Beta sampler — polynomial power  (≥ 0)
#
#  (1 − t/Λ²)^β  with t<0  gives (1+|t|/Λ²)^β, a growing
#  prefactor.  β = 0 → pure exponential;  β ~ 2-3 → strong
#  polynomial shoulder before exponential damping kicks in.
# ═════════════════════════════════════════════════════════════════════
def _sample_beta():
    strategy = random.choice([
        'pure_exponential',       # β ≈ 0
        'light_polynomial',       # β ∈ (0, 1)
        'integer_motivated',      # β ∈ {1, 2, 3}  (multipole-like)
        'moderate_continuous',    # β ∈ [0.5, 2.0]
        'strong_polynomial',      # β ∈ [2, 3]
        'wide_continuous',        # β ∈ [0, 4]  — broader than paper
        'fractional_small',       # β ∈ [0.1, 0.5]
        'near_dipole',            # β ~ 2 with jitter
    ])

    if strategy == 'pure_exponential':
        return abs(torch.randn(1, dtype=torch.float64).item()) * 0.05
    elif strategy == 'light_polynomial':
        return 0.1 + 0.9 * random.random()
    elif strategy == 'integer_motivated':
        base = random.choice([1, 2, 3])
        return base + torch.randn(1).item() * 0.15
    elif strategy == 'moderate_continuous':
        return 0.5 + 1.5 * random.random()
    elif strategy == 'strong_polynomial':
        return 2.0 + 1.5 * random.random()
    elif strategy == 'wide_continuous':
        return 4.0 * random.random()
    elif strategy == 'fractional_small':
        return 0.1 + 0.4 * random.random()
    else:  # near_dipole
        return 2.0 + torch.randn(1).item() * 0.3

# ═════════════════════════════════════════════════════════════════════
#  Gamma sampler — exponential damping rate  (γ > 0  REQUIRED)
#
#  exp(γ t / Λ²) = exp(−γ|t|/Λ²)  →  1/e  when  |t| = Λ²/γ.
#
#  Small γ  → slow damping, broad form factor
#  Large γ  → fast damping, narrow / compact form factor
#
#  The effective damping at the grid edge |t|_max = 2 GeV² is
#     exp(−γ · 2/Λ²).
#  For Λ ~ 1 GeV and γ ~ 1 this is exp(−2) ≈ 0.14 — reasonable.
#  We sample γ ∈ (0, ~5] to cover everything from almost-no-damping
#  to very-rapid-damping regimes.
# ═════════════════════════════════════════════════════════════════════
def _sample_gamma():
    strategy = random.choice([
        'weak',               # gentle damping
        'moderate',           # standard AdS/QCD-like
        'strong',             # aggressive damping
        'very_weak',          # barely decaying on the grid
        'very_strong',        # almost dead by |t| = 1 GeV²
        'log_uniform',        # broad log-flat coverage
        'peaked_around_one',  # γ ~ 1 ± 0.3
        'continuous_wide',    # uniform in [0.05, 5]
    ])

    if strategy == 'weak':
        g = 0.05 + 0.35 * random.random()           # [0.05, 0.4]
    elif strategy == 'moderate':
        g = 0.4 + 0.8 * random.random()              # [0.4, 1.2]
    elif strategy == 'strong':
        g = 1.2 + 1.3 * random.random()              # [1.2, 2.5]
    elif strategy == 'very_weak':
        g = 0.01 + 0.09 * random.random()            # [0.01, 0.1]
    elif strategy == 'very_strong':
        g = 2.5 + 2.5 * random.random()              # [2.5, 5.0]
    elif strategy == 'log_uniform':
        g = np.exp(np.random.uniform(np.log(0.02), np.log(5.0)))
    elif strategy == 'peaked_around_one':
        g = 1.0 + torch.randn(1).item() * 0.3
    else:  # continuous_wide
        g = 0.05 + 4.95 * random.random()

    return max(g, 0.01)   # enforce strict positivity


# ═════════════════════════════════════════════════════════════════════
#  Joint correlation strategies
#
#  Instead of always sampling (Λ, β, γ) independently, we
#  sometimes draw them from physically-motivated joint regions
#  to cover the interesting corners of parameter space.
# ═════════════════════════════════════════════════════════════════════
def _sample_correlated_params():
    """
    Return (FF0, Lambda, beta, gamma) from a joint strategy that
    targets specific physical regimes.
    """
    regime = random.choice([
        'soft_wall_AdS',        # Brodsky & de Téramond: Λ ~ 0.5, β ~ 0, γ ~ 1
        'hard_wall_AdS',        # Abidin & Carlson: Λ ~ 1, β ~ 1-2, γ moderate
        'constituent_quark',    # soft scale, strong damping
        'chiral_limit',         # light Λ, gentle polynomial
        'heavy_compact',        # large Λ, fast damping → narrow peak
        'dipole_like',          # β ~ 2, γ small → mimics dipole at low |t|
        'pure_gaussian',        # β = 0, γ moderate → Gaussian-like
        'plateau_then_drop',    # small γ, large β → flat near 0, then drops
    ])

    FF0 = _sample_ff0_target()

    if regime == 'soft_wall_AdS':
        Lambda = 0.4 + 0.3 * random.random()
        beta   = abs(torch.randn(1).item()) * 0.3
        gamma  = 0.6 + 0.8 * random.random()

    elif regime == 'hard_wall_AdS':
        Lambda = 0.8 + 0.6 * random.random()
        beta   = 1.0 + 1.0 * random.random()
        gamma  = 0.3 + 0.7 * random.random()

    elif regime == 'constituent_quark':
        Lambda = 0.3 + 0.3 * random.random()
        beta   = 0.5 + 1.0 * random.random()
        gamma  = 1.5 + 2.0 * random.random()

    elif regime == 'chiral_limit':
        Lambda = 0.25 + 0.25 * random.random()
        beta   = 0.2 + 0.8 * random.random()
        gamma  = 0.1 + 0.5 * random.random()

    elif regime == 'heavy_compact':
        Lambda = 2.0 + 1.5 * random.random()
        beta   = 0.5 + 1.5 * random.random()
        gamma  = 2.0 + 3.0 * random.random()

    elif regime == 'dipole_like':
        Lambda = 0.7 + 0.8 * random.random()
        beta   = 1.8 + 0.4 * random.random()
        gamma  = 0.02 + 0.15 * random.random()

    elif regime == 'pure_gaussian':
        Lambda = 0.5 + 1.5 * random.random()
        beta   = abs(torch.randn(1).item()) * 0.05
        gamma  = 0.5 + 1.5 * random.random()

    else:  # plateau_then_drop
        Lambda = 1.0 + 1.0 * random.random()
        beta   = 2.5 + 1.0 * random.random()
        gamma  = 0.05 + 0.2 * random.random()

    beta  = max(beta, 0.0)
    gamma = max(gamma, 0.01)
    Lambda = max(abs(Lambda), 0.25)

    return FF0, Lambda, beta, gamma


# ─────────────────────────────────────────────────────────────────────
#  Padé approximant  [N, M]
#
#  D(t) = (a_0 + Σ_{i=1}^{Na-1} a_i (-t)^i)
#        / (1  + Σ_{i=0}^{Nb-1} |b_i| (-t)^{i+1})
#
#  Code convention:  Na = N + 1  (number of a-coefficients)
#                    Nb = M      (number of b-coefficients)
#
#  • a[0] = D(0)  because  den(0) = 1.
#  • |b_i| in the denominator guarantees  den > 0  ∀ t ≤ 0
#    → no spacelike poles by construction.
#  • pQCD asymptotics: D(t) ~ 1/(-t)^{M−N} at large -t.
#    M − N ≥ 2  ⟹  correct 1/(-t)² falloff.
#
#  Paper combinations [N,M]:
#    [0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [2,4]
#  We also include:  [0,4], [1,4], [3,5]  for extra diversity.
# ─────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════
#  All [N,M] → (Na, Nb) combinations, grouped by pQCD correctness
# ═════════════════════════════════════════════════════════════════════
#  M − N ≥ 2  (correct pQCD falloff ~ 1/t²)
PADE_ORDERS_PQCD = [
    (1, 2),   # [0,2]
    (1, 3),   # [0,3]
    (1, 4),   # [0,4]
    (2, 3),   # [1,3]
    (2, 4),   # [1,4]
    (3, 4),   # [2,4]
    (3, 5),   # [2,5]
    (4, 6),   # [3,6]
]

#  M − N = 1  (~ 1/t falloff — not strict pQCD but adds shape diversity)
PADE_ORDERS_SLOW = [
    (1, 1),   # [0,1]
    (2, 2),   # [1,2]
    (3, 3),   # [2,3]
]

PADE_ORDERS_ALL = PADE_ORDERS_PQCD + PADE_ORDERS_SLOW



# ═════════════════════════════════════════════════════════════════════
#  [Na, Nb] order sampler
# ═════════════════════════════════════════════════════════════════════
def _sample_order():
    """
    Return (Na, Nb) with ~75 % probability of pQCD-correct orders
    and ~25 % for the slower-falloff ones.
    """
    if random.random() < 0.75:
        return random.choices(
            population=PADE_ORDERS_PQCD,
            weights   =[3, 4, 2, 4, 3, 4, 2, 1],   # favor [0,2],[1,3],[2,4]
            k=1,
        )[0]
    else:
        return random.choices(
            population=PADE_ORDERS_SLOW,
            weights   =[2, 3, 3],
            k=1,
        )[0]


# ═════════════════════════════════════════════════════════════════════
#  Numerator coefficient sampler  (a[1], …, a[Na-1])
#
#  a[0] is always set to the FF(0) target, so we only sample the
#  higher-order numerator coefficients here.
# ═════════════════════════════════════════════════════════════════════
def _sample_numerator_coeffs(n_coeffs):
    """
    Sample n_coeffs numerator coefficients (indices 1 … Na-1).
    Returns list of length n_coeffs.  (Returns [] if n_coeffs == 0.)
    """
    if n_coeffs == 0:
        return []

    strategy = random.choice([
        'gaussian_natural',
        'gaussian_wide',
        'uniform',
        'decaying',
        'alternating',
        'sparse',
        'log_uniform',
        'tiny',
    ])

    if strategy == 'gaussian_natural':
        a = torch.randn(n_coeffs, dtype=torch.float64)

    elif strategy == 'gaussian_wide':
        sigma = 1.0 + 3.0 * random.random()
        a = torch.randn(n_coeffs, dtype=torch.float64) * sigma

    elif strategy == 'uniform':
        a = (torch.rand(n_coeffs, dtype=torch.float64) * 2 - 1) * 4.0

    elif strategy == 'decaying':
        r = 0.2 + 0.5 * random.random()
        env = torch.tensor([r ** k for k in range(n_coeffs)], dtype=torch.float64)
        a = torch.randn(n_coeffs, dtype=torch.float64) * env * 3.0

    elif strategy == 'alternating':
        signs = torch.tensor([(-1.0) ** k for k in range(n_coeffs)],
                             dtype=torch.float64)
        mags = torch.abs(torch.randn(n_coeffs, dtype=torch.float64)) * 2.0
        a = signs * mags

    elif strategy == 'sparse':
        a = torch.zeros(n_coeffs, dtype=torch.float64)
        n_nz = random.randint(1, max(1, n_coeffs))
        idx = random.sample(range(n_coeffs), n_nz)
        a[idx] = torch.randn(n_nz, dtype=torch.float64) * 2.5

    elif strategy == 'log_uniform':
        mags = 10.0 ** (torch.rand(n_coeffs, dtype=torch.float64) * 2 - 1)
        signs = 2.0 * torch.randint(0, 2, (n_coeffs,)).to(torch.float64) - 1.0
        a = mags * signs

    else:  # tiny — nearly pure monopole/dipole in the numerator
        a = torch.randn(n_coeffs, dtype=torch.float64) * 0.05

    return a.tolist()


# ═════════════════════════════════════════════════════════════════════
#  Denominator coefficient sampler  (b[0], …, b[Nb-1])
#
#  The code takes |b_i|, so the sign is irrelevant; only the
#  magnitude matters.  Larger |b| → steeper falloff.
#
#  Physical scale:  (-t)^{i+1} at |t| = 1 GeV² is just 1^{i+1} = 1,
#  so b ~ O(1) gives a natural mass scale ~ 1 GeV.  Smaller b
#  pushes the effective mass higher; larger b pulls it lower.
# ═════════════════════════════════════════════════════════════════════
def _sample_denominator_coeffs(n_coeffs):
    """
    Sample n_coeffs denominator coefficients.
    Returns list of length n_coeffs with magnitude > 0.
    """
    strategy = random.choice([
        'natural',            # |b| ~ O(1)
        'light_masses',       # large b → light effective mass → fast falloff
        'heavy_masses',       # small b → heavy effective mass → slow falloff
        'mixed',              # some light, some heavy
        'kelly_inspired',     # Kelly parametrization scale ~ 0.1–5 GeV⁻²
        'log_uniform',        # broad coverage
        'hierarchical',       # b[0] dominates, rest small
        'democratic',         # all b roughly equal
    ])

    if strategy == 'natural':
        b = torch.abs(torch.randn(n_coeffs, dtype=torch.float64)) + 0.01

    elif strategy == 'light_masses':
        b = 2.0 + 3.0 * torch.rand(n_coeffs, dtype=torch.float64)

    elif strategy == 'heavy_masses':
        b = 0.01 + 0.2 * torch.rand(n_coeffs, dtype=torch.float64)

    elif strategy == 'mixed':
        b = torch.zeros(n_coeffs, dtype=torch.float64)
        for j in range(n_coeffs):
            if random.random() < 0.5:
                b[j] = 0.01 + 0.3 * random.random()     # heavy
            else:
                b[j] = 1.0 + 3.0 * random.random()      # light

    elif strategy == 'kelly_inspired':
        # Kelly: G_E^p(t) = (1 − a₂τ) / (1 + b₁τ + b₂τ² + b₃τ³)
        # with τ = -t / (4M_N²),  typical b ~ 5–15 in τ-units
        # → translated to (-t) units:  b ~ 0.5–3
        center = 0.5 + 2.0 * random.random()
        b = torch.abs(torch.randn(n_coeffs, dtype=torch.float64) * 0.5 + center)

    elif strategy == 'log_uniform':
        b = 10.0 ** (torch.rand(n_coeffs, dtype=torch.float64) * 3 - 1.5)
        # range ~ [0.03, 30]

    elif strategy == 'hierarchical':
        b = torch.zeros(n_coeffs, dtype=torch.float64)
        b[0] = 0.5 + 4.0 * random.random()
        for j in range(1, n_coeffs):
            b[j] = b[j - 1] * (0.05 + 0.3 * random.random())

    else:  # democratic
        level = 0.3 + 2.0 * random.random()
        b = torch.abs(torch.randn(n_coeffs, dtype=torch.float64) * 0.15 + level)

    # Floor: avoid exactly-zero denominators coefficients
    b = torch.clamp(b.abs(), min=1e-4)

    return b.tolist()


# ═════════════════════════════════════════════════════════════════════
#  Joint / physics-motivated parameter regimes
# ═════════════════════════════════════════════════════════════════════
def _sample_correlated_pade():
    """
    Return (FF0, Na, Nb, a_higher, b) from a joint strategy that
    targets specific physical form-factor shapes.

    Returns
    -------
    FF0        : float
    Na, Nb     : int
    a_higher   : list[float]   (length Na − 1; excludes a[0])
    b          : list[float]   (length Nb)
    """
    regime = random.choice([
        'pure_multipole',       # [0,M] with single-scale denominator → multipole
        'kelly_em',             # Kelly EM form factor shape
        'stiff_core',           # fast-rising denominator → compact object
        'broad_tail',           # heavy-mass denominator → slow falloff
        'numerator_dip',        # numerator zero creates a dip / shoulder
        'balanced_rational',    # comparable num & den degrees
        'near_monopole',        # [0,1] ≈ monopole
        'high_order_smooth',    # [2,4] or [3,5] with decaying coeffs
    ])

    FF0 = _sample_ff0_target()

    if regime == 'pure_multipole':
        Na = 1                                    # no numerator powers
        Nb = random.choice([2, 3, 4])
        a_higher = []
        # Single-scale denominator: b_i = (1/M²)^{i+1} with M ~ 0.5–2
        M = 0.5 + 1.5 * random.random()
        b = [(1.0 / M ** 2) ** (i + 1) for i in range(Nb)]

    elif regime == 'kelly_em':
        Na, Nb = 2, 3                             # [1,3]
        a_higher = [-(0.1 + 0.3 * random.random())]   # small negative a₁
        center = 1.0 + 1.0 * random.random()
        b = [center * (0.3 + 0.5 * random.random()) ** i for i in range(Nb)]

    elif regime == 'stiff_core':
        Na, Nb = 1, random.choice([2, 3])
        a_higher = []
        b = [3.0 + 5.0 * random.random() for _ in range(Nb)]

    elif regime == 'broad_tail':
        Na, Nb = 1, random.choice([2, 3, 4])
        a_higher = []
        b = [0.02 + 0.1 * random.random() for _ in range(Nb)]

    elif regime == 'numerator_dip':
        # A positive a₁ (for negative FF0) or negative a₁ (for positive FF0)
        # creates a numerator zero → dip/shoulder in D(t)
        Na = random.choice([2, 3])
        Nb = Na + random.choice([1, 2])
        a_higher = []
        for k in range(1, Na):
            sign = -np.sign(FF0) if k == 1 else random.choice([-1, 1])
            a_higher.append(sign * (0.3 + 1.5 * random.random()))
        b = _sample_denominator_coeffs(Nb)

    elif regime == 'balanced_rational':
        Na = random.choice([2, 3])
        Nb = Na + 1
        scale = 0.3 + 1.5 * random.random()
        a_higher = [((-1) ** k) * scale * (0.2 + 0.8 * random.random())
                    for k in range(1, Na)]
        b = [scale * (0.3 + 0.7 * random.random()) for _ in range(Nb)]

    elif regime == 'near_monopole':
        Na, Nb = 1, 1
        a_higher = []
        # Single b ~ 1/M² with M around a meson mass
        M = random.choice([0.50, 0.77, 0.98, 1.27, 1.52])
        b = [1.0 / M ** 2]

    else:  # high_order_smooth
        Na, Nb = random.choice([(3, 4), (3, 5), (4, 6)])
        r = 0.2 + 0.4 * random.random()
        a_higher = [((-1) ** k) * r ** k * (0.5 + random.random())
                    for k in range(1, Na)]
        r_b = 0.3 + 0.5 * random.random()
        b = [r_b ** i * (0.5 + random.random()) for i in range(Nb)]

    return FF0, Na, Nb, a_higher, b


# ═════════════════════════════════════════════════════════════════════
#  Class 6 — Dispersive / Spectral Representation
#
#  D(t) = (1/π) ∫_{t_cut}^{s_max} ds  ρ(s) / (s − t)
#
#  ρ(s) = (s − t_cut)^p  ×  Σ_i c_i Γ_i / [(s − m_i²)² + (Γ_i/2)²]
#                         ×  exp(−s / s_max)
#
#  Key physics:
#  • Two-pion threshold curvature near −t ~ 0.1 GeV²
#  • Breit–Wigner peaks at physical meson masses (0⁺⁺, 2⁺⁺)
#  • Threshold behavior (s − t_cut)^{3/2 to 5/2}
#  • High-energy damping exp(−s/s_max)
#
#  References:
#    Cao et al., Nat. Commun. 16, 6979 (2025) [2409.15547]
#    Hammer & Meißner, EPJA 20, 469 (2004)
#    Höhler et al., NPB 114, 505 (1976)
# ═════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
#  Physical meson widths (PDG-inspired) in GeV
# ─────────────────────────────────────────────────────────────────────
MESON_WIDTHS_PDG = {
    # Scalars (0++)
    'f0_500':  0.550,    # σ  — very broad
    'f0_980':  0.060,
    'f0_1370': 0.350,
    'f0_1500': 0.109,
    'f0_1710': 0.123,
    'f0_2020': 0.400,
    'f0_2100': 0.300,
    'f0_2200': 0.200,
    # Tensors (2++)
    'f2_1270': 0.186,
    'f2_1430': 0.130,
    'f2_1525': 0.073,
    'f2_1640': 0.150,
    'f2_1810': 0.200,
    'f2_1950': 0.472,
    'f2_2010': 0.200,
    'f2_2300': 0.150,
    'f2_2340': 0.320,
}

# Paired list for easy random access: [{key, mass, width}, ...]
MESON_PROPS = [
    {'key': k, 'mass': MESON_MASSES[k], 'width': MESON_WIDTHS_PDG[k]}
    for k in MESON_MASSES if k in MESON_WIDTHS_PDG
]

SCALAR_PROPS = [p for p in MESON_PROPS if p['key'].startswith('f0')]
TENSOR_PROPS = [p for p in MESON_PROPS if p['key'].startswith('f2')]

# Numerical defaults
N_S_DEFAULT = 2000        # integration grid points
MIN_PEAK_MASS = 0.30      # GeV — floor on BW peak masses


# ═════════════════════════════════════════════════════════════════════
#  Helper: number of Breit–Wigner peaks
# ═════════════════════════════════════════════════════════════════════
def _sample_n_peaks():
    return random.choices(
        population=[1,  2,  3,  4,  5,  6],
        weights   =[2,  4,  4,  3,  2,  1],
        k=1,
    )[0]


# ═════════════════════════════════════════════════════════════════════
#  Helper: sample BW peak masses (GeV), returned sorted
# ═════════════════════════════════════════════════════════════════════
def _sample_disp_masses(n_peaks):
    strategy = random.choice([
        'pdg_exact',           # draw directly from known mesons
        'pdg_jittered',        # physical center + Gaussian jitter
        'scalar_only',         # only 0++ states
        'tensor_only',         # only 2++ states
        'low_lying',           # σ / f0(980) / f2(1270) region
        'spread_out',          # evenly spread from low to high
        'log_uniform',         # broad log-uniform coverage
    ])

    if strategy == 'pdg_exact' and n_peaks <= len(MESON_PROPS):
        selected = random.sample(MESON_PROPS, n_peaks)
        masses = [p['mass'] for p in selected]

    elif strategy == 'pdg_jittered':
        jitter = 0.03 + 0.10 * random.random()          # 30–130 MeV
        masses = [random.choice(MESON_PROPS)['mass']
                  + torch.randn(1).item() * jitter
                  for _ in range(n_peaks)]

    elif strategy == 'scalar_only':
        masses = [random.choice(SCALAR_PROPS)['mass']
                  + torch.randn(1).item() * 0.05
                  for _ in range(n_peaks)]

    elif strategy == 'tensor_only':
        masses = [random.choice(TENSOR_PROPS)['mass']
                  + torch.randn(1).item() * 0.05
                  for _ in range(n_peaks)]

    elif strategy == 'low_lying':
        centers = [0.50, 0.77, 0.98, 1.275]
        masses = [random.choice(centers) + torch.randn(1).item() * 0.08
                  for _ in range(n_peaks)]

    elif strategy == 'spread_out':
        lo, hi = 0.5, 2.3
        step = (hi - lo) / max(n_peaks - 1, 1)
        masses = [lo + step * i + torch.randn(1).item() * 0.10
                  for i in range(n_peaks)]

    else:  # log_uniform
        log_lo, log_hi = np.log(0.4), np.log(2.5)
        masses = [np.exp(np.random.uniform(log_lo, log_hi))
                  for _ in range(n_peaks)]

    # Floor and sort
    masses = sorted([max(abs(m), MIN_PEAK_MASS) for m in masses])
    return masses


# ═════════════════════════════════════════════════════════════════════
#  Helper: sample BW peak widths (GeV)
#
#  Physical widths range from ~20 MeV  (f2(1525))  to ~550 MeV (σ).
#  We also allow artificial narrow / broad limits for shape diversity.
# ═════════════════════════════════════════════════════════════════════
def _sample_disp_widths(n_peaks, masses):
    strategy = random.choice([
        'pdg_matched',          # closest PDG width for each mass
        'pdg_jittered',         # PDG × random factor
        'all_narrow',           # Γ ~ 20–80 MeV
        'all_broad',            # Γ ~ 250–700 MeV
        'mixed',                # some narrow, some broad
        'mass_proportional',    # Γ ∝ m
        'log_uniform',          # log-flat in [10 MeV, 800 MeV]
        'sigma_plus_narrow',    # first peak σ-like broad, rest narrow
    ])

    if strategy == 'pdg_matched':
        widths = []
        for m in masses:
            closest = min(MESON_PROPS, key=lambda p: abs(p['mass'] - m))
            widths.append(closest['width'])

    elif strategy == 'pdg_jittered':
        widths = []
        for m in masses:
            closest = min(MESON_PROPS, key=lambda p: abs(p['mass'] - m))
            w = closest['width'] * (0.5 + random.random())   # 50–150 %
            widths.append(w)

    elif strategy == 'all_narrow':
        widths = [0.02 + 0.06 * random.random() for _ in range(n_peaks)]

    elif strategy == 'all_broad':
        widths = [0.25 + 0.45 * random.random() for _ in range(n_peaks)]

    elif strategy == 'mixed':
        widths = []
        for _ in range(n_peaks):
            if random.random() < 0.5:
                widths.append(0.02 + 0.08 * random.random())    # narrow
            else:
                widths.append(0.20 + 0.50 * random.random())    # broad

    elif strategy == 'mass_proportional':
        alpha = 0.05 + 0.25 * random.random()
        widths = [alpha * m for m in masses]

    elif strategy == 'log_uniform':
        widths = [np.exp(np.random.uniform(np.log(0.01), np.log(0.8)))
                  for _ in range(n_peaks)]

    else:  # sigma_plus_narrow
        widths = []
        for i, m in enumerate(masses):
            if i == 0 and m < 0.70:
                widths.append(0.40 + 0.30 * random.random())    # σ-like
            else:
                widths.append(0.03 + 0.12 * random.random())

    # Floor: avoid zero widths (BW becomes δ-function)
    widths = [max(w, 0.005) for w in widths]
    return widths


# ═════════════════════════════════════════════════════════════════════
#  Helper: sample BW couplings (before FF(0) rescaling)
# ═════════════════════════════════════════════════════════════════════
def _sample_disp_couplings(n_peaks):
    strategy = random.choice([
        'gaussian',
        'gaussian_wide',
        'uniform',
        'alternating',
        'dominant_first',
        'dominant_last',
        'all_same_sign',
        'log_uniform',
        'hierarchical',
    ])

    if strategy == 'gaussian':
        c = torch.randn(n_peaks, dtype=torch.float64) * 1.5

    elif strategy == 'gaussian_wide':
        sigma = 2.0 + 4.0 * random.random()
        c = torch.randn(n_peaks, dtype=torch.float64) * sigma

    elif strategy == 'uniform':
        c = (torch.rand(n_peaks, dtype=torch.float64) * 2 - 1) * 4.0

    elif strategy == 'alternating':
        signs = torch.tensor([(-1.0) ** i for i in range(n_peaks)],
                             dtype=torch.float64)
        mags = 0.5 + 2.5 * torch.rand(n_peaks, dtype=torch.float64)
        c = signs * mags

    elif strategy == 'dominant_first':
        c = torch.randn(n_peaks, dtype=torch.float64) * 0.3
        c[0] = (3.0 + 4.0 * random.random()) * random.choice([-1, 1])

    elif strategy == 'dominant_last':
        c = torch.randn(n_peaks, dtype=torch.float64) * 0.3
        c[-1] = (3.0 + 4.0 * random.random()) * random.choice([-1, 1])

    elif strategy == 'all_same_sign':
        sign = random.choice([-1.0, 1.0])
        c = sign * (0.3 + 3.0 * torch.rand(n_peaks, dtype=torch.float64))

    elif strategy == 'log_uniform':
        mags = 10.0 ** (torch.rand(n_peaks, dtype=torch.float64) * 2 - 0.5)
        signs = 2.0 * torch.randint(0, 2, (n_peaks,)).to(torch.float64) - 1.0
        c = mags * signs

    else:  # hierarchical
        r = 0.3 + 0.5 * random.random()
        env = torch.tensor([r ** i for i in range(n_peaks)], dtype=torch.float64)
        c = torch.randn(n_peaks, dtype=torch.float64) * env * 3.0

    return c.tolist()


# ═════════════════════════════════════════════════════════════════════
#  Helper: threshold power  p  in  (s − t_cut)^p
#
#  Paper:  p ∈ {3/2, 5/2}  (partial-wave threshold behavior).
#  We broaden the range for shape diversity but weight toward
#  the physical half-integer values.
# ═════════════════════════════════════════════════════════════════════
def _sample_threshold_power():
    strategy = random.choice([
        'physical_half_int',       # 3/2 or 5/2  (paper values)
        'physical_continuous',     # uniform in [1.0, 3.0]
        'low',                     # sharp onset  [0.5, 1.5]
        'high',                    # suppressed onset  [2.5, 4.0]
        'integer',                 # 1, 2, 3
        'broad',                   # [0.5, 5.0]
    ])

    if strategy == 'physical_half_int':
        base = random.choice([1.5, 2.5])
        return base + torch.randn(1).item() * 0.15
    elif strategy == 'physical_continuous':
        return 1.0 + 2.0 * random.random()
    elif strategy == 'low':
        return 0.5 + 1.0 * random.random()
    elif strategy == 'high':
        return 2.5 + 1.5 * random.random()
    elif strategy == 'integer':
        return float(random.choice([1, 2, 3]))
    else:  # broad
        return 0.5 + 4.5 * random.random()


# ═════════════════════════════════════════════════════════════════════
#  Helper: two-pion threshold  t_cut = 4 m_π²
# ═════════════════════════════════════════════════════════════════════
def _sample_disp_t_cut():
    strategy = random.choice([
        'physical',            # m_π = 140 MeV (± 5 %)
        'lattice',             # m_π = 170 MeV (± 5 %)
        'broad_uniform',       # uniform in [0.05, 0.5] GeV²
        'broad_log',           # log-uniform in [0.05, 0.5]
    ])

    if strategy == 'physical':
        m_pi = 0.140 * np.random.uniform(0.95, 1.05)
        return 4.0 * m_pi ** 2
    elif strategy == 'lattice':
        m_pi = 0.170 * np.random.uniform(0.95, 1.05)
        return 4.0 * m_pi ** 2
    elif strategy == 'broad_uniform':
        return np.random.uniform(0.05, 0.5)
    else:  # broad_log
        return np.exp(np.random.uniform(np.log(0.05), np.log(0.5)))


# ═════════════════════════════════════════════════════════════════════
#  Helper: s_max  —  upper integration limit AND damping scale
#
#  s_max plays a double role:
#    1. Upper limit of the trapezoidal integral
#    2. Scale in exp(−s/s_max)
#
#  Too small → truncates physical resonance contributions.
#  The integral must cover all BW peaks:  s_max > max(m_i²) + margin.
#  We enforce this in the main sampler after masses are chosen.
# ═════════════════════════════════════════════════════════════════════
def _sample_s_max(min_required=2.0):
    """
    Parameters
    ----------
    min_required : float
        Lower bound, typically max(m_i²) + margin.
    """
    strategy = random.choice([
        'compact',             # focused on low-lying resonances
        'moderate',            # standard range
        'large',               # captures high-energy tail
        'very_large',          # very extended integration
        'log_uniform',         # broad log-flat coverage
    ])

    if strategy == 'compact':
        s = 3.0 + 2.0 * random.random()              # [3, 5]
    elif strategy == 'moderate':
        s = 5.0 + 5.0 * random.random()               # [5, 10]
    elif strategy == 'large':
        s = 10.0 + 15.0 * random.random()              # [10, 25]
    elif strategy == 'very_large':
        s = 25.0 + 25.0 * random.random()              # [25, 50]
    else:  # log_uniform
        s = np.exp(np.random.uniform(np.log(3.0), np.log(50.0)))

    return max(s, min_required)


# ═════════════════════════════════════════════════════════════════════
#  Correlated / physics-motivated joint regimes
#
#  8 regimes targeting different physical scenarios for the
#  spectral function.
# ═════════════════════════════════════════════════════════════════════
def _sample_correlated_dispersive():
    """
    Return a dict with all dispersive parameters drawn from a
    joint strategy targeting specific physical regimes.
    """
    regime = random.choice([
        'sigma_dominated',         # broad σ pole dominates
        'tensor_resonances',       # f₂ states drive the spectral fn
        'near_threshold',          # curvature from two-pion threshold
        'high_energy_tail',        # extended integration, many peaks
        'narrow_peaks',            # sharp resonance features
        'broad_continuum',         # many overlapping broad peaks → smooth
        'physical_realistic',      # PDG masses + widths for dominant states
        'interference_pattern',    # alternating-sign couplings → oscillatory ρ
    ])

    FF0_target = _sample_ff0_target()

    if regime == 'sigma_dominated':
        n_peaks = random.choice([1, 2])
        masses = [0.50 + 0.10 * torch.randn(1).item()]
        widths = [0.40 + 0.30 * random.random()]
        if n_peaks == 2:
            masses.append(0.98 + 0.05 * torch.randn(1).item())
            widths.append(0.05 + 0.05 * random.random())
        couplings = [random.choice([-1, 1]) * (1.0 + 2.0 * random.random())
                     for _ in range(n_peaks)]
        t_cut = 4.0 * (0.140 + 0.005 * torch.randn(1).item()) ** 2
        threshold_power = 1.5 + 0.3 * random.random()
        s_max = 5.0 + 5.0 * random.random()

    elif regime == 'tensor_resonances':
        n_peaks = random.choice([2, 3, 4])
        selected = random.sample(TENSOR_PROPS, min(n_peaks, len(TENSOR_PROPS)))
        masses = [p['mass'] + 0.03 * torch.randn(1).item() for p in selected]
        widths = [p['width'] * (0.7 + 0.6 * random.random()) for p in selected]
        # Pad if fewer tensor states than n_peaks
        while len(masses) < n_peaks:
            masses.append(1.0 + 1.5 * random.random())
            widths.append(0.10 + 0.20 * random.random())
        couplings = _sample_disp_couplings(n_peaks)
        t_cut = 4.0 * 0.140 ** 2
        threshold_power = 2.0 + 0.5 * random.random()
        s_max = 10.0 + 10.0 * random.random()

    elif regime == 'near_threshold':
        # Low-lying peaks, small s_max → threshold curvature dominates
        n_peaks = random.choice([1, 2])
        masses = sorted([0.40 + 0.20 * random.random()
                         for _ in range(n_peaks)])
        widths = [0.30 + 0.40 * random.random() for _ in range(n_peaks)]
        couplings = [random.choice([-1, 1]) * (1.0 + 3.0 * random.random())
                     for _ in range(n_peaks)]
        t_cut = 0.05 + 0.05 * random.random()
        threshold_power = 1.0 + 0.5 * random.random()
        s_max = 4.0 + 3.0 * random.random()

    elif regime == 'high_energy_tail':
        n_peaks = random.choice([3, 4, 5])
        masses = sorted([0.8 + 1.5 * random.random()
                         for _ in range(n_peaks)])
        widths = [0.10 + 0.30 * random.random() for _ in range(n_peaks)]
        couplings = _sample_disp_couplings(n_peaks)
        t_cut = 4.0 * 0.140 ** 2
        threshold_power = 2.5 + 1.0 * random.random()
        s_max = 20.0 + 30.0 * random.random()

    elif regime == 'narrow_peaks':
        n_peaks = random.choice([2, 3])
        selected = random.sample(MESON_PROPS, n_peaks)
        masses = [p['mass'] for p in selected]
        widths = [0.01 + 0.03 * random.random() for _ in range(n_peaks)]
        couplings = _sample_disp_couplings(n_peaks)
        t_cut = 4.0 * 0.140 ** 2
        threshold_power = 1.5 + 0.15 * torch.randn(1).item()
        s_max = 8.0 + 7.0 * random.random()

    elif regime == 'broad_continuum':
        n_peaks = random.choice([4, 5, 6])
        masses = sorted([0.5 + 2.0 * random.random()
                         for _ in range(n_peaks)])
        widths = [0.30 + 0.50 * random.random() for _ in range(n_peaks)]
        couplings = _sample_disp_couplings(n_peaks)
        t_cut = 4.0 * 0.140 ** 2
        threshold_power = 2.0 + 1.0 * random.random()
        s_max = 15.0 + 15.0 * random.random()

    elif regime == 'physical_realistic':
        # Three dominant states: σ, f₀(980), f₂(1270)
        states = ['f0_500', 'f0_980', 'f2_1270']
        n_peaks = 3
        masses = [MESON_MASSES[s] for s in states]
        widths = [MESON_WIDTHS_PDG[s] for s in states]
        couplings = [random.choice([-1, 1]) * (0.5 + 2.0 * random.random())
                     for _ in range(n_peaks)]
        t_cut = 4.0 * 0.140 ** 2
        threshold_power = random.choice([1.5, 2.0, 2.5])
        s_max = 8.0 + 7.0 * random.random()

    else:  # interference_pattern
        n_peaks = random.choice([2, 3, 4])
        masses = sorted([0.5 + 1.5 * random.random()
                         for _ in range(n_peaks)])
        widths = [0.05 + 0.15 * random.random() for _ in range(n_peaks)]
        # Force alternating signs → constructive/destructive interference
        couplings = [((-1) ** i) * (1.0 + 3.0 * random.random())
                     for i in range(n_peaks)]
        t_cut = 4.0 * 0.140 ** 2
        threshold_power = 1.5 + 0.5 * random.random()
        s_max = 10.0 + 10.0 * random.random()

    # ── Safety floors ────────────────────────────────────────────────
    masses = sorted([max(abs(m), MIN_PEAK_MASS) for m in masses])
    widths = [max(w, 0.005) for w in widths]
    t_cut = max(t_cut, 0.01)
    threshold_power = max(threshold_power, 0.5)

    return dict(
        FF0_target=FF0_target,
        masses=masses,
        widths=widths,
        couplings=couplings,
        t_cut=t_cut,
        threshold_power=threshold_power,
        s_max=s_max,
    )


#######


# Multipole_Samples
def Multipole(Number_Sample=None):
    """
    Sample diverse multipole form factors with maximum parameter space coverage
    """
    all_samples = []
    
    for i in range(Number_Sample):
        # 1. Sample FF(0) - vary the distribution
        ff0_dist = random.choice(['uniform', 'concentrated_zero', 'extremes'])
        if ff0_dist == 'uniform':
            FF0 = (torch.rand(1, dtype=torch.float64) * 20 - 10).item()
        elif ff0_dist == 'concentrated_zero':
            # More samples near zero (±3)
            FF0 = (torch.randn(1, dtype=torch.float64) * 3).item()
            FF0 = torch.clamp(torch.tensor(FF0), -10, 10).item()
        else:  # extremes
            # Favor extreme values (±7 to ±10)
            sign = random.choice([-1, 1])
            FF0 = sign * (7 + torch.rand(1, dtype=torch.float64) * 3).item()
        
        # 2. Sample M (mass scale) - diverse strategies
        m_strategy = random.choice(['light', 'medium', 'heavy', 'mixed', 
                                     'meson_inspired', 'log_uniform'])
        if m_strategy == 'light':
            M = (0.3 + torch.rand(1, dtype=torch.float64) * 0.7).item()
        elif m_strategy == 'medium':
            M = (0.8 + torch.rand(1, dtype=torch.float64) * 1.2).item()
        elif m_strategy == 'heavy':
            M = (1.5 + torch.rand(1, dtype=torch.float64) * 2.0).item()
        elif m_strategy == 'mixed':
            M = (0.3 + torch.rand(1, dtype=torch.float64) * 3.2).item()
        elif m_strategy == 'meson_inspired':
            # Sample around typical meson masses
            M = random.choice([0.5, 0.77, 0.98, 1.27, 1.52, 2.0])
            M += (torch.randn(1, dtype=torch.float64) * 0.1).item()
            M = abs(M)
        else:  # log_uniform (more samples at low masses)
            M = torch.exp(torch.rand(1, dtype=torch.float64) * 
                          (np.log(3.5) - np.log(0.3)) + np.log(0.3)).item()
        # 3. Sample n (power) - diverse strategies
        n_strategy = random.choice(['low', 'medium', 'high', 'continuous', 
                                     'very_high', 'fractional'])
        if n_strategy == 'low':
            n = random.randint(1, 3)
        elif n_strategy == 'medium':
            n = random.randint(3, 6)
        elif n_strategy == 'high':
            n = random.randint(6, 10)
        elif n_strategy == 'very_high':
            n = random.randint(10, 15)  # Very steep falloff
        elif n_strategy == 'fractional':
            # Non-integer powers between 0.5 and 3
            n = (0.5 + torch.rand(1, dtype=torch.float64) * 2.5).item()
        else:  # continuous
            n = (1.0 + torch.rand(1, dtype=torch.float64) * 12.0).item()
        curve = multipole(FF0=FF0, M=M, n=n)
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue
        if curve.abs().max() > 10.0:
            continue
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue
        all_samples.append(curve)
    
    result = torch.stack(all_samples)
    idx = torch.randperm(result.shape[0])
    return result[idx]









# ═════════════════════════════════════════════════════════════════════
#  Main sampler
# ═════════════════════════════════════════════════════════════════════
def Z_Expansion(Number_Sample=None):
    """
    Generate diverse z-expansion form-factor samples.

    Parameters
    ----------
    Number_Sample : int
        Number of attempts.  Actual yield is lower due to rejections.

    Returns
    -------
    result : torch.Tensor, shape (N_accepted, 200)
        Shuffled array of accepted curves.
    """
    all_samples = []

    for i in range(Number_Sample):

        # ── 1. Truncation order ──────────────────────────────────────
        k_max = random.choices(
            population=[2, 3, 4, 5, 6, 8, 10, 12, 14],
            weights=   [3, 4, 4, 3, 3, 2,  1,  1,  1],
            k=1
        )[0]

        # ── 2. Two-pion threshold ────────────────────────────────────
        t_cut_strategy = random.choice([
            'physical', 'lattice', 'broad_uniform', 'broad_log'
        ])
        if t_cut_strategy == 'physical':
            t_cut = 4.0 * (0.140 * np.random.uniform(0.95, 1.05)) ** 2
        elif t_cut_strategy == 'lattice':
            t_cut = 4.0 * (0.170 * np.random.uniform(0.95, 1.05)) ** 2
        elif t_cut_strategy == 'broad_uniform':
            t_cut = np.random.uniform(0.05, 0.5)
        else:
            t_cut = np.exp(np.random.uniform(np.log(0.05), np.log(0.5)))

        # ── 3. Optimal t_0 with jitter ───────────────────────────────
        t_max_mag = 2.0
        t_0 = t_cut - np.sqrt(t_cut * (t_cut + t_max_mag))
        t_0 *= np.random.uniform(0.85, 1.15)

        # ── 4. Compute z on full grid, check convergence |z| < 1 ────
        sqrt_tc_t0 = np.sqrt(t_cut - t_0)
        z_grid = (torch.sqrt(t_cut - t) - sqrt_tc_t0) / \
                 (torch.sqrt(t_cut - t) + sqrt_tc_t0)

        z_max = z_grid.abs().max().item()
        if z_max >= 1.0:
            continue

        # ── 5. Compensated coefficients ──────────────────────────────
        a = sample_coefficients_compensated(k_max, z_max)

        # ── 6. Fix a_0 analytically to hit target FF(0) ─────────────
        #    D(0) = a_0 + a_1*z0 + a_2*z0² + ...
        #    ⟹  a_0 = target - Σ_{k=1}^{K-1} a_k * z0^k
        target_ff0 = sample_ff0_target()
        z0 = z_grid[0].item()
        z0_powers = torch.tensor(
            [z0 ** k for k in range(1, k_max)], dtype=torch.float64
        )
        a_0 = target_ff0 - (a[1:] * z0_powers).sum().item()

        if abs(a_0) > 50.0:
            continue
        a[0] = a_0

        # ── 7. Evaluate on the t-grid ────────────────────────────────
        curve = z_expansion(
            a=a,
            t_cut=torch.tensor(t_cut, dtype=torch.float64),
            t_0=t_0,
            k_max=k_max
        )

        # ── 8. Sanity checks ────────────────────────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue

        if curve.abs().max() > 10.0:
            continue
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue
        all_samples.append(curve)

    if len(all_samples) == 0:
        raise RuntimeError(
            f"Z_Expansion: 0 accepted out of {Number_Sample} attempts."
        )

    result = torch.stack(all_samples)
    idx = torch.randperm(result.shape[0])
    return result[idx]






# ═════════════════════════════════════════════════════════════════════
#  Main sampler
# ═════════════════════════════════════════════════════════════════════
def Meson_Pole(Number_Sample=None):
    """
    Generate diverse meson-dominance / sum-of-poles form-factor
    samples using physical meson masses from MESON_MASSES.

    FF(t) = Σ_i  c_i / (m_i² − t)

    Features
    --------
    • Number of poles N ∈ {2, 3, 4, 5, 6}, weighted toward N = 3, 4.
    • Six mass-selection strategies centered on physical meson states.
    • Seven residue-sampling strategies for shape diversity.
    • Superconvergence sum rules (Σc_i = 0, Σc_i m_i² = 0) enforced
      for ~70 % of N ≥ 3 samples → correct pQCD falloff D(t) ~ 1/(-t)².
    • Residues rescaled to hit a random D(0) target ∈ [−10, 10].
    • Numerical guards: minimum mass, mass separation, residue caps,
      amplitude clamp, flatness rejection.

    Parameters
    ----------
    Number_Sample : int
        Number of sampling attempts.  Actual yield is lower due to
        physics / numerical rejections.

    Returns
    -------
    result : torch.Tensor, shape (N_accepted, 200)
        Shuffled tensor of accepted curves.
    """
    all_samples = []

    for _ in range(Number_Sample):

        # ── 1. Number of poles ───────────────────────────────────────
        k_max = random.choices(
            population=[2,  3,  4,  5,  6, 8 , 10, 12],
            weights   =[1,  4,  4,  2,  1, 1, 1, 1],
            k=1,
        )[0]

        # ── 2. Pole masses (sorted, well-separated, ≥ MIN_MASS) ─────
        masses = _sample_pole_masses(k_max)

        # ── 3. Decide whether to enforce superconvergence ────────────
        #    N = 2 → impossible (degenerate); N ≥ 3 → 70 % probability
        enforce_sc = (k_max >= 3) and (random.random() < 0.70)

        # ── 4. Sample residues ───────────────────────────────────────
        c = _sample_residues(k_max, masses, enforce_sc)
        if c is None:
            continue

        # ── 5. Rescale all c_i to hit target D(0) ───────────────────
        #    D(0) = Σ c_i / m_i²
        #    Rescaling c → α·c preserves superconvergence (linear).
        target_d0 = _sample_ff0_target()
        d0_current = sum(c[j] / masses[j] ** 2 for j in range(k_max))

        if abs(d0_current) < 1e-10:
            continue

        scale = target_d0 / d0_current
        c_scaled = [ci * scale for ci in c]

        # ── 6. Guard: cap individual residues after rescaling ────────
        if any(abs(ci) > 500.0 for ci in c_scaled):
            continue

        # ── 7. Evaluate on the t-grid ────────────────────────────────
        curve = pole_expansion(c=c_scaled, m=masses, k_max=k_max)

        # ── 8. Numerical & physics sanity checks ─────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue
        if curve.abs().max() > 10.0:
            continue
        if (curve.max() - curve.min()).abs() < 1e-2:       # reject flat
            continue

        all_samples.append(curve)

    # ── Collect and shuffle ──────────────────────────────────────────
    if len(all_samples) == 0:
        raise RuntimeError(
            f"Meson_Pole: 0 curves accepted out of {Number_Sample} attempts. "
            "Consider increasing Number_Sample or relaxing guards."
        )

    result = torch.stack(all_samples)
    return result[torch.randperm(result.shape[0])]


# ═════════════════════════════════════════════════════════════════════
#  Main sampler
# ═════════════════════════════════════════════════════════════════════
def Modified_Exponential(Number_Sample=None):
    """
    Generate diverse modified-exponential form-factor samples.

    D(t) = FF0 · (1 − t/Λ²)^β · exp(γ · t/Λ²)

    Features
    --------
    • ~50 % of samples: all four parameters drawn independently
      from their individual multi-strategy samplers.
    • ~50 % of samples: drawn from 8 physically-motivated joint
      regimes (soft-wall AdS, hard-wall AdS, constituent quark,
      chiral limit, heavy/compact, dipole-like, pure Gaussian,
      plateau-then-drop).
    • 7 Λ strategies ∈ [0.25, 4] GeV  (including meson-inspired).
    • 8 β strategies ∈ [0, ~4]  (pure exponential to strong polynomial).
    • 8 γ strategies ∈ (0, 5]  (very weak to very strong damping).
    • 5 FF(0) distributions spanning [−10, 10].
    • γ > 0 strictly enforced (required for falloff).
    • Amplitude clamp |D(t)| ≤ 10, flatness rejection, NaN/Inf guard.

    Parameters
    ----------
    Number_Sample : int
        Number of sampling attempts.

    Returns
    -------
    result : torch.Tensor, shape (N_accepted, 200)
    """
    all_samples = []

    for _ in range(Number_Sample):

        # ── Choose independent vs. correlated sampling ───────────────
        if random.random() < 0.50:
            # Independent
            FF0    = _sample_ff0_target()
            Lambda = _sample_lambda()
            beta   = _sample_beta()
            gamma  = _sample_gamma()
        else:
            # Correlated / physically-motivated
            FF0, Lambda, beta, gamma = _sample_correlated_params()

        # ── Evaluate ─────────────────────────────────────────────────
        curve = modified_exponential(
            FF0=FF0, Lambda=Lambda, beta=beta, gamma=np.abs(gamma),
        )

        # ── Sanity checks ────────────────────────────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue
        if curve.abs().max() > 10.0:
            continue
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue

        all_samples.append(curve)

    if len(all_samples) == 0:
        raise RuntimeError(
            f"Modified_Exponential: 0 curves accepted out of "
            f"{Number_Sample} attempts."
        )

    result = torch.stack(all_samples)
    return result[torch.randperm(result.shape[0])]



# ═════════════════════════════════════════════════════════════════════
#  Main sampler
# ═════════════════════════════════════════════════════════════════════
def Pade_Approximant(Number_Sample=None):
    """
    Generate diverse Padé-approximant form-factor samples.

    D(t) = (a₀ + Σ aᵢ(-t)^i) / (1 + Σ |bᵢ|(-t)^{i+1})

    Features
    --------
    • ~50 % independent sampling, ~50 % physics-motivated regimes.
    • 11 [N,M] combinations, 75 % pQCD-correct (M−N ≥ 2).
    • 8 numerator coefficient strategies (Gaussian, decaying,
      alternating, sparse, log-uniform, tiny, …).
    • 8 denominator coefficient strategies (natural, light/heavy
      masses, Kelly-inspired, hierarchical, democratic, …).
    • 8 joint physics regimes (pure multipole, Kelly EM, stiff core,
      broad tail, numerator dip, balanced rational, near monopole,
      high-order smooth).
    • a[0] = FF(0) target directly (since den(0) = 1).
    • Denominator guaranteed positive (|b| in code) → no poles.
    • Amplitude clamp |D(t)| ≤ 10, flatness rejection, NaN/Inf guard.

    Parameters
    ----------
    Number_Sample : int
        Number of sampling attempts.

    Returns
    -------
    result : torch.Tensor, shape (N_accepted, 200)
    """
    all_samples = []

    for _ in range(Number_Sample):

        # ── Choose independent vs. correlated ────────────────────────
        if random.random() < 0.50:
            # ── Independent ──────────────────────────────────────────
            Na, Nb = _sample_order()
            FF0 = _sample_ff0_target()
            a_higher = _sample_numerator_coeffs(Na - 1)     # a[1] … a[Na-1]
            b = _sample_denominator_coeffs(Nb)

        else:
            # ── Correlated / physics-motivated ───────────────────────
            FF0, Na, Nb, a_higher, b = _sample_correlated_pade()

        # ── Assemble full a vector: a[0] = FF(0), then higher ────────
        a_full = [FF0] + a_higher

        # ── Quick pre-check: if any coefficient absurdly large, skip ─
        if any(abs(ai) > 100.0 for ai in a_full):
            continue
        if any(abs(bi) > 100.0 for bi in b):
            continue

        # ── Evaluate ─────────────────────────────────────────────────
        try:
            curve = pade_approximant(a=a_full, b=b, Na=Na, Nb=Nb)
        except Exception:
            continue

        # ── Sanity checks ────────────────────────────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue
        if curve.abs().max() > 10.0:
            continue
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue

        all_samples.append(curve)

    if len(all_samples) == 0:
        raise RuntimeError(
            f"Pade_Approximant: 0 curves accepted out of "
            f"{Number_Sample} attempts."
        )

    result = torch.stack(all_samples)
    return result[torch.randperm(result.shape[0])]





# ═════════════════════════════════════════════════════════════════════
#  Main sampler
# ═════════════════════════════════════════════════════════════════════
def Dispersive_Spectral_Sampler(Number_Sample=None):
    all_samples = []

    for _ in range(Number_Sample):

        # ── Choose independent vs. correlated sampling ───────────────
        if random.random() < 0.45:
            # ── Independent ──────────────────────────────────────────
            n_peaks         = _sample_n_peaks()
            masses          = _sample_disp_masses(n_peaks)
            widths          = _sample_disp_widths(n_peaks, masses)
            couplings       = _sample_disp_couplings(n_peaks)
            t_cut           = _sample_disp_t_cut()
            threshold_power = _sample_threshold_power()
            FF0_target      = _sample_ff0_target()

            # s_max must cover all peaks: s_max > max(m²) + margin
            max_m_sq = max(m ** 2 for m in masses)
            s_max = _sample_s_max(min_required=max_m_sq + 2.0)

        else:
            # ── Correlated / physics-motivated ───────────────────────
            params          = _sample_correlated_dispersive()
            masses          = params['masses']
            widths          = params['widths']
            couplings       = params['couplings']
            t_cut           = params['t_cut']
            threshold_power = params['threshold_power']
            s_max           = params['s_max']
            FF0_target      = params['FF0_target']
            n_peaks         = len(masses)

            # Enforce s_max coverage for correlated regimes too
            max_m_sq = max(m ** 2 for m in masses)
            s_max = max(s_max, max_m_sq + 1.5)

        # ── Guard: peak masses should be above threshold ─────────────
        #    m² < t_cut → BW peak sits below threshold, suppressed.
        #    Not fatal (broad peaks still contribute), but for narrow
        #    peaks it produces near-zero spectral weight.
        #    We shift offending masses just above threshold.
        #sqrt_tcut_plus = np.sqrt(t_cut) + 0.05
        #masses = [max(m, sqrt_tcut_plus) for m in masses]
        # ── Guard: peak masses above threshold + no degeneracies ─────
        sqrt_tcut_plus = np.sqrt(t_cut) + 0.05
        masses_fixed = []
        for m in sorted(masses):
            m_new = max(m, sqrt_tcut_plus)
            while masses_fixed and abs(m_new - masses_fixed[-1]) < 0.03:
                m_new += 0.03 + 0.02 * random.random()
            masses_fixed.append(m_new)
        masses = masses_fixed
        # ── Determine n_s: more points for large s_max ───────────────
        if s_max > 25.0:
            n_s = 4000
        elif s_max > 10.0:
            n_s = 3000
        else:
            n_s = N_S_DEFAULT
        # ── FIX 1: Ensure BW peaks are resolved by the grid ─────────
        ds = (s_max - t_cut) / n_s
        min_resolvable_width = 4.0 * ds
        widths = [max(w, min_resolvable_width) for w in widths]
        # ── Evaluate the dispersive integral ─────────────────────────
        try:
            curve = dispersive_spectral(
                pole_masses=masses,
                pole_couplings=couplings,
                pole_widths=widths,
                t_cut=t_cut,
                threshold_power=threshold_power,
                s_max=s_max,
                n_s=n_s,
            )
        except Exception:
            continue

        # ── NaN / Inf guard ──────────────────────────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue

        # ── Rescale to hit target FF(0) ──────────────────────────────
        #    D(t) is linear in the couplings c_i, so scaling the
        #    output curve by α = target / D(0) is exact and preserves
        #    the shape.
        d0 = curve[0].item()
        if abs(d0) < 1e-12:
            continue
        curve = curve * (FF0_target / d0)

        # ── Amplitude clamp ──────────────────────────────────────────
        if curve.abs().max() > 10.0:
            continue

        # ── Flatness rejection ───────────────────────────────────────
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue

        all_samples.append(curve)

    # ── Collect and shuffle ──────────────────────────────────────────
    if len(all_samples) == 0:
        raise RuntimeError(
            f"Dispersive_Spectral: 0 curves accepted out of "
            f"{Number_Sample} attempts.  "
            "Consider increasing Number_Sample or relaxing guards."
        )

    result = torch.stack(all_samples)
    return result[torch.randperm(result.shape[0])]


def Log_Modified_Multipole(Number_Sample=None):
    """
    Generate diverse log-modified multipole form-factor samples.

    Parameters
    ----------
    Number_Sample : int
        Number of attempts.  Actual yield is lower due to rejections.

    Returns
    -------
    result : torch.Tensor, shape (N_accepted, 200)
        Shuffled array of accepted curves.
    """
    all_samples = []

    for i in range(Number_Sample):
        '''
        ff0_strategy = random.choices(
            population=['pheno', 'wide', 'extreme_low', 'mild'],
            weights=   [4,       3,      1,              2],
            k=1
        )[0]

        if ff0_strategy == 'pheno':
            FF0 = np.random.uniform(-5.0, -1.0) if random.random() < 0.5 \
            else np.random.uniform(1.0, 5.0)
        elif ff0_strategy == 'wide':
            FF0 = -np.random.uniform(0.5, 8.0)
        elif ff0_strategy == 'extreme_low':
            FF0 = -np.random.uniform(5.0, 10.0)
        else:  # mild
            FF0 = -np.random.uniform(0.1, 1.5)
        '''
        FF0 = _sample_ff0_target()

        # ── 2. Mass scale M [GeV] ───────────────────────────────────
        #    Broad range; log-uniform gives better low-M coverage.
        M_strategy = random.choices(
            population=['standard', 'log_uniform', 'low', 'high'],
            weights=   [4,          3,             1,     1],
            k=1
        )[0]

        if M_strategy == 'standard':
            M = np.random.uniform(0.5, 2.5)
        elif M_strategy == 'log_uniform':
            M = np.exp(np.random.uniform(np.log(0.4), np.log(3.0)))
        elif M_strategy == 'low':
            M = np.random.uniform(0.3, 0.7)
        else:  # high
            M = np.random.uniform(2.0, 4.0)

        # ── 3. Multipole power n ─────────────────────────────────────
        #    Integer-heavy (dipole/tripole most common), but allow
        #    continuous values for shape diversity.
        n_strategy = random.choices(
            population=['integer', 'continuous', 'high_continuous'],
            weights=   [5,         3,            2],
            k=1
        )[0]

        if n_strategy == 'integer':
            n = random.choices(
                population=[1, 2, 3, 4],
                weights=   [2, 4, 3, 1],
                k=1
            )[0]
            n = float(n)
        elif n_strategy == 'continuous':
            n = np.random.uniform(1.0, 4.5)
        else:  # high_continuous
            n = np.random.uniform(3.0, 7.0)

        # ── 4. Log-correction strength c ─────────────────────────────
        #    c ∈ [0, 0.5] (paper range), but broaden slightly for
        #    diversity.  Include c ≈ 0 (pure multipole recovery).
        c_strategy = random.choices(
            population=['paper', 'near_zero', 'extended', 'tiny'],
            weights=   [4,       2,           2,          1],
            k=1
        )[0]

        if c_strategy == 'paper':
            c = np.random.uniform(0.0, 0.5)
        elif c_strategy == 'near_zero':
            c = np.abs(np.random.normal(0.0, 0.03))
        elif c_strategy == 'extended':
            c = np.random.uniform(0.0, 0.8)
        else:  # tiny
            c = np.random.uniform(0.0, 0.01)

        # ── 5. Anomalous dimension δ ─────────────────────────────────
        #    δ ∈ [-1.5, 0] (paper range), broaden for diversity.
        #    Include δ ≈ 0 (pure multipole recovery).
        delta_strategy = random.choices(
            population=['paper', 'near_zero', 'extended', 'deep'],
            weights=   [4,       2,           2,          1],
            k=1
        )[0]

        if delta_strategy == 'paper':
            delta = np.random.uniform(-1.5, 0.0)
        elif delta_strategy == 'near_zero':
            delta = -np.abs(np.random.normal(0.0, 0.1))
        elif delta_strategy == 'extended':
            delta = np.random.uniform(-2.5, 0.0)
        else:  # deep
            delta = np.random.uniform(-3.0, -1.0)

        # ── 6. Evaluate on the t-grid ────────────────────────────────
        #    u = 1 - t/M²  (since t < 0, u > 1 always)
        u = 1.0 - t / M**2
        log_u = torch.log(u)

        # Guard: 1 + c·ln(u) must be positive everywhere for
        # real-valued (1 + c·ln(u))^δ
        log_factor = 1.0 + c * log_u
        if (log_factor <= 0.0).any():
            continue

        curve = FF0 / u ** n * log_factor ** delta

        # ── 7. Sanity checks ────────────────────────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue

        # Amplitude guard
        if curve.abs().max() > 10.0:
            continue

        # Not too flat
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue

        all_samples.append(curve)

    if len(all_samples) == 0:
        raise RuntimeError(
            f"Log_Modified_Multipole: 0 accepted out of "
            f"{Number_Sample} attempts."
        )

    result = torch.stack(all_samples)
    idx = torch.randperm(result.shape[0])
    return result[idx]

def Bag_Model_Bessel(Number_Sample=None):
    """
    Generate diverse bag-model / Bessel-type form-factor samples.

    Parameters
    ----------
    Number_Sample : int
        Number of attempts.  Actual yield is lower due to rejections.

    Returns
    -------
    result : torch.Tensor, shape (N_accepted, 200)
        Shuffled array of accepted curves.
    """
    all_samples = []

    for i in range(Number_Sample):

        # ── 1. FF(0): target forward-limit value ────────────────────
        ff0_strategy = random.choices(
            population=['uniform', 'gauss_narrow', 'gauss_wide', 'edges', 'log_uniform'],
            weights=   [3,          2,              2,            1,       2],
            k=1
        )[0]

        if ff0_strategy == 'uniform':
            FF0 = np.random.uniform(-10.0, 10.0)
        elif ff0_strategy == 'gauss_narrow':
            FF0 = np.random.normal(0.0, 2.0)
        elif ff0_strategy == 'gauss_wide':
            FF0 = np.random.normal(0.0, 5.0)
        elif ff0_strategy == 'edges':
            FF0 = np.random.choice([-1, 1]) * np.random.uniform(5.0, 10.0)
        else:  # log_uniform in magnitude, random sign
            FF0 = np.random.choice([-1, 1]) * np.exp(
                np.random.uniform(np.log(0.05), np.log(10.0))
            )

        FF0 = float(np.clip(FF0, -10.0, 10.0))
        if abs(FF0) < 0.02:
            continue

        # ── 2. Bag radius R [GeV⁻¹]  (1 fm ≈ 5.068 GeV⁻¹) ────────
        #   Physical: 0.5–1.5 fm → 2.5–7.6 GeV⁻¹
        #   Broaden well beyond for shape diversity
        r_strategy = random.choices(
            population=['physical', 'small', 'large', 'broad_uniform', 'broad_log'],
            weights=   [3,          2,       2,       2,               1],
            k=1
        )[0]

        if r_strategy == 'physical':
            R_fm = np.random.uniform(0.5, 1.5)
            R = R_fm * 5.068
        elif r_strategy == 'small':
            R = np.random.uniform(0.3, 2.5)
        elif r_strategy == 'large':
            R = np.random.uniform(7.0, 18.0)
        elif r_strategy == 'broad_uniform':
            R = np.random.uniform(0.3, 18.0)
        else:  # broad_log
            R = np.exp(np.random.uniform(np.log(0.3), np.log(18.0)))

        # ── 3. Damping coefficient β [GeV⁻²] ───────────────────────
        #   Physical: 0.1–1.0;  broaden for diversity
        beta_strategy = random.choices(
            population=['physical', 'broad_uniform', 'broad_log', 'very_small', 'large'],
            weights=   [3,          2,               2,            1,            1],
            k=1
        )[0]

        if beta_strategy == 'physical':
            beta = np.random.uniform(0.1, 1.0)
        elif beta_strategy == 'broad_uniform':
            beta = np.random.uniform(0.005, 4.0)
        elif beta_strategy == 'broad_log':
            beta = np.exp(np.random.uniform(np.log(0.005), np.log(4.0)))
        elif beta_strategy == 'very_small':
            beta = np.random.uniform(0.001, 0.05)
        else:  # large
            beta = np.random.uniform(2.0, 6.0)

        # ── 4. Evaluate on the t-grid ───────────────────────────────
        curve = bag_model_bessel(FF0=FF0, R=R, beta=beta)

        # ── 5. Sanity checks ────────────────────────────────────────
        if torch.isnan(curve).any() or torch.isinf(curve).any():
            continue
        if curve.abs().max() > 10.0:
            continue
        if (curve.max() - curve.min()).abs() < 1e-2:
            continue

        all_samples.append(curve)

    if len(all_samples) == 0:
        raise RuntimeError(
            f"Bag_Model_Bessel: 0 accepted out of {Number_Sample} attempts."
        )

    result = torch.stack(all_samples)
    idx = torch.randperm(result.shape[0])
    return result[idx]


def Convex_Combination_Pair(sampler_A, sampler_B, Number_Sample=None, N_pool=500):
    """
    Pairwise convex combinations: λ·D_A + (1-λ)·D_B, λ ~ U(0,1).
    """
    pool_A = sampler_A(Number_Sample=N_pool)
    pool_B = sampler_B(Number_Sample=N_pool)

    N = min(Number_Sample, len(pool_A), len(pool_B))
    all_samples = []

    for i in range(N):
        lam = torch.rand(1, dtype=torch.float64).item()
        curve = lam * pool_A[i] + (1.0 - lam) * pool_B[i]

        if curve.abs().max() > 10.0:
            continue
        all_samples.append(curve)

    result = torch.stack(all_samples)
    return result[torch.randperm(len(result))]


def Convex_Combination_Multi(samplers, Number_Sample=None, N_pool=500):
    """
    Dirichlet-weighted combinations of M parent samplers.
    """
    M = len(samplers)
    pools = [s(Number_Sample=N_pool) for s in samplers]

    N = min(Number_Sample, *[len(p) for p in pools])
    all_samples = []

    for i in range(N):
        #weights = torch.distributions.Dirichlet(torch.ones(M, dtype=torch.float64)).sample()
        alpha = torch.ones(M, dtype=torch.float64) * np.random.uniform(0.3, 3.0)
        weights = torch.distributions.Dirichlet(alpha).sample()
        curve = sum(weights[j] * pools[j][i] for j in range(M))

        if curve.abs().max() > 10.0:
            continue
        all_samples.append(curve)

    result = torch.stack(all_samples)
    return result[torch.randperm(len(result))]

'''
def Meson_Pole(Number_Sample=None, Number_Poles=None):
    all_samples = []
    for i in range(Number_Sample):
        for k_max in range(2, Number_Poles):
            target_ff_0 = torch.rand(1) * 20 - 10
            # Vary coefficient distribution
            coeff_scale = 0.5 + torch.rand(1).item() * 2.5  # 0.5 to 3.0
            c = torch.randn(k_max, dtype=torch.float64) * coeff_scale
            # Vary mass sampling
            m = torch.zeros(k_max, dtype=torch.float64)
            strategy = random.choice(['perturbed', 'exact', 'uniform'])
            noise_level = 0.05 + torch.rand(1).item() * 0.15  # 50-200 MeV
            if strategy == 'perturbed':
                for j in range(k_max):
                    center_mass = random.choice(meson_mass_list)
                    m[j] = torch.abs(torch.randn(1) * noise_level + center_mass)
            elif strategy == 'exact' and k_max <= len(meson_mass_list):
                selected = random.sample(meson_mass_list, k_max)
                m = torch.tensor(selected, dtype=torch.float64)
                m = m + torch.randn(k_max) * 0.05  # Small jitter
                m = torch.abs(m)
            else:  # uniform or fallback
                m = torch.linspace(0.5, 2.0, k_max, dtype=torch.float64)
                m = m + torch.randn(k_max) * 0.1
                m = torch.abs(m)
            # Scale to hit target
            current_ff_0 = (c / m**2).sum()
            if torch.abs(current_ff_0) > 1e-10:
                c = c * (target_ff_0 / current_ff_0)
            else:
                continue
            c = c.numpy().tolist()
            m = m.numpy().tolist()
            all_samples.append(pole_expansion(c=c, m=m, k_max=k_max))
    result = torch.stack(all_samples)
    idx = torch.randperm(result.shape[0])
    return result[idx]
'''

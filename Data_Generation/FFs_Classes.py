import torch
import numpy as np
from typing import List, Optional
from scipy.interpolate import PchipInterpolator

"""
Authors: H. Alharazin and J. Yu. Panteleeva
"""

# ─────────────────────────────────────────────────────────────────────
#  Global grid:  t ≡ −t_Mandelstam ∈ [0, 2] GeV²
# ─────────────────────────────────────────────────────────────────────
t = -torch.linspace(0.0, 2.0, 200, dtype=torch.float64)


# ═════════════════════════════════════════════════════════════════════
#  Class 1 — Multipole
#  Eq. (1);  Hackett et al., PRL 132, 251904 (2024)
# ═════════════════════════════════════════════════════════════════════
def multipole(FF0 = None, M = None, n = None):
    # FF(t) = FF(0) / (1 + t/M²)^n
    return FF0 / (1.0 - t / M**2) ** n


# ═════════════════════════════════════════════════════════════════════
#  Class 2 — z-Expansion
#  # Reference 2507.05375 mention the explicit reference for the z_expansion
#    and it also mentions something about possible constraints to be done
#    on the coefficients a.
# ═════════════════════════════════════════════════════════════════════
def z_expansion(a = None, t_cut = None, t_0 = None, k_max = None):
    #FF(t) = #sum_{k=0}^{K} a_k  z(t)^k
    #z(t) = (#sqrt{t_cut - t} - #sqrt{t_cut - t_0})
    #     / (#sqrt{t_cut - t} + #sqrt{t_cut - t_0})
    sqrt_t_cut = torch.sqrt(t_cut - t)
    sqrt_cut_t0_cut = np.sqrt(t_cut - t_0)
    z = (sqrt_t_cut - sqrt_cut_t0_cut) / (sqrt_t_cut + sqrt_cut_t0_cut)
    z_powers = torch.stack([z ** k for k in range(k_max)])
    return torch.einsum('k,kt->t', a, z_powers)


# ═════════════════════════════════════════════════════════════════════
#  Class 3 — Meson Dominance / Sum of Poles
#  It was mentioned in 2503.09297
# ═════════════════════════════════════════════════════════════════════
def pole_expansion(c = None, m = None, k_max = None):
    #FF(t) = #sum_i c_i / (m_i^2 - t)
    result = torch.stack([c[i] / (m[i]**2 - t) for i in range(k_max)])
    return result.sum(dim=0)



# ═════════════════════════════════════════════════════════════════════
#  Class 4 — Modified Exponential
#  # Careful that gamma needs to be positive here !
# ═════════════════════════════════════════════════════════════════════
def modified_exponential(FF0 = None, Lambda = None, beta =  None, gamma = None):
    #D(t) = FF(0) (1 - t/ambda^2)^#beta #exp(-amma t/ambda^2)
    return FF0 * (1.0 - t / Lambda**2) ** beta * torch.exp(np.abs(gamma) * t / Lambda**2)


# ═════════════════════════════════════════════════════════════════════
#  Class 5 — Padé Approximants [n, m]
#  Mentioned here: arxiv.org/pdf/hep-ph/0405258. m >= n + 2
# ═════════════════════════════════════════════════════════════════════
def pade_approximant(a = None, b = None, Na = None, Nb = None):
    #FF(t) = (a_0 + sum_(n=1)^(N)a_n(-t)^n) / (1 + sum_(n=1)^(M)b_n(-t)^n)
    #num = a[0] + torch.stack([a[i] * (-t) ** i for i in range(1, Na)]).sum(dim=0)
    #den = 1 + torch.stack([b[i] * (-t) ** (i+1) for i in range(0, Nb)]).sum(dim=0)
    num = a[0] + torch.zeros_like(t)
    for i in range(1, Na):
        num = num + a[i] * (-t) ** i
    den = 1.0 + torch.zeros_like(t)
    for i in range(0, Nb):
        den = den + b[i] * (-t) ** (i + 1)
    return num / den


# ═════════════════════════════════════════════════════════════════════
#  Class 6 — Dispersive / Spectral Representation
#  s_max plays a double role: upper integration limit and the damping scale in exp(−s/s_max),
#  so don't make it too small or the integral truncates physical contributions
# ═════════════════════════════════════════════════════════════════════
def dispersive_spectral(pole_masses = None, pole_couplings = None, pole_widths = None, t_cut = None,
                        threshold_power = None, s_max = None, n_s = None):
    #FF(t) = #frac{1}{#pi}#int_{t_{cut}}^{#infty} ds
    ##frac{#rho(s)}{s + t}
    #where #rho(s) is modelled as a threshold factor times
    #a sum of Breit–Wigner peaks.
    s = torch.linspace(t_cut + 1e-6, s_max, n_s, dtype=torch.float64)
    ds = s[1] - s[0]

    # Spectral function: threshold × ΣBW × high-energy damping
    threshold = (s - t_cut) ** threshold_power
    rho = torch.zeros_like(s)
    for mi, ci, gi in zip(pole_masses, pole_couplings, pole_widths):
        rho = rho + ci * gi / ((s - mi**2) ** 2 + (gi / 2.0) ** 2)
    rho = rho * threshold * torch.exp(-s / s_max)

    # D(t) = (1/π) ∫ ds ρ(s)/(s + t)   [s − t_paper = s + t]
    integrand = rho[None, :] / (s[None, :] - t[:, None])  # (N, n_s)
    return (ds / torch.pi) * integrand.sum(dim=1)


# ═════════════════════════════════════════════════════════════════════
#  Class 7 — Log-Modified Multipole (pQCD corrections)
#  Eq. (12);  Tong, Ma & Yuan, JHEP 10, 046 (2022)
# ═════════════════════════════════════════════════════════════════════
def log_modified_multipole(FF0 = None, M = None, n = None, c = None, delta = None):
    # FF(t) = #frac{FF(0)}{(1+t/M^2)^n}
    ##bigl[1 + cn(1+t/M^2)#bigr]^{#delta}"""
    u = 1.0 - t / M**2
    return FF0 / u ** n * (1.0 + c * torch.log(u)) ** delta


# ═════════════════════════════════════════════════════════════════════
#  Class 8 — Bag Model / Bessel-Type
#  Eq. (14);  Neubelt, Sampino & Schweitzer, PRD 101, 034013 (2020)
# ═════════════════════════════════════════════════════════════════════
def bag_model_bessel(FF0 = None, R = None, beta = None):
    # FF(t) = FF(0)#frac{3j_1(R#sqrt{t})}{R#sqrt{t}}
    #e^{-#betat}
    #Uses  j_1(x)/x → 1  as  x → 0.
    #"""
    x = R * torch.sqrt(-t)
    safe_x = torch.where(x < 1e-12, torch.ones_like(x), x)
    j1_over_x = (torch.sin(safe_x) / safe_x - torch.cos(safe_x)) / safe_x
    three_j1_over_x = 3.0 * j1_over_x / safe_x
    three_j1_over_x = torch.where(x < 1e-12, torch.ones_like(three_j1_over_x),
                                   three_j1_over_x)
    return FF0 * three_j1_over_x * torch.exp(beta * t)



import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_D(
    *curves: torch.Tensor,
    labels: list[str] | None = None,
    title: str = r"$FF$",
    xlabel: str = r"$-t\;[\mathrm{GeV}^2]$",
    ylabel: str = r"$FF(t)$",
    figsize: tuple = (4, 4),
    save: str | None = None,
    dpi: int = 100,
) -> None:
    """Plot one or more D(t) curves on the standard 200-point grid.
    Parameters
    ----------
    *curves  : one or more tensors returned by any Class function.
    labels   : optional list of legend labels (one per curve).
    title    : plot title.
    save     : if given, save figure to this path instead of showing.
    """
    # ── modern style ────────────────────────────────────────────────
    with mpl.rc_context({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#fafafa",
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.color":        "#e0e0e0",
        "grid.linewidth":    0.6,
        "font.family":       "serif",
        "mathtext.fontset":  "cm",
        "font.size":         13,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
    }):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        N = curves[0].shape[0]
        t_np = torch.linspace(0.0, 2.0, N).numpy()
        palette = plt.cm.Set2.colors if len(curves) <= 8 else plt.cm.tab20.colors
        for i, curve in enumerate(curves):
            lbl = labels[i] if labels else None
            ax.plot(t_np, curve.detach().numpy(),
                    color=palette[i % len(palette)],
                    linewidth=2.2, label=lbl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.set_xlim(0, 2.0)
        if labels:
            ax.legend(loc="best")
        fig.tight_layout()
        if save:
            fig.savefig(save, bbox_inches="tight")
            print(f"Saved → {save}")
        else:
            plt.show()
        plt.close(fig)
# Gravitational Form Factor Reconstruction

Training code for the paper:

> **Reconstruction of gravitational form factors using generative machine learning**
>
> 
> Herzallah Alharazin, Julia Yu. Panteleeva
>
> 
> [https://arxiv.org/abs/2602.19267v2](https://arxiv.org/abs/2602.19267v2)

## Overview

This repository provides the code used to:

1. **Generate the training data** — diverse families of physically motivated form-factor curves used to train the diffusion model.
2. **Train and fine-tune the DDPM** that learns the space of gravitational form factors.

The hadronic gravitational form factors are reconstructed over the full spacelike range $0 \le -t \le 2\;\mathrm{GeV}^2$ by conditioning the diffusion model on a small number of known data points. The framework yields non-parametric, model-independent reconstructions with quantified uncertainties.

## Companion Repository

The sampling, curve generation, and full analysis pipeline are maintained in the companion repository:

> **[GFF_Diffusion_Analysis](https://github.com/JuliaYu24/GFF_Diffusion_Analysis)** — contains the DDPM sampler, all GFF reconstruction configurations reported in the paper, and the analysis notebook for figure production and extraction of the chiral low-energy constants $c_8$ and $c_9$.

## Repository Structure

```
├── README.md
├── Data_Generation/
│   ├── Training_Data_Generator.py   # Generates 60k curves per family (10 families)
│   ├── FFs_Classes.py               # Form-factor functional forms
│   └── Random_FF_Generator.py       # Randomised parameter sampling & filtering
└── Diffusion_Source/
    ├── DiffusionModel.py            # DDPM architecture, training loop, fine-tuning
    ├── finetune.py                  # Fine-tuning entry point
    └── gff_ddpm_best.pt             # Trained checkpoint (after training)
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

## Usage

### 1. Generate training data

```bash
cd Data_Generation
python Training_Data_Generator.py --target 60000 --outdir ../data --seed 42
```

This produces one `.pt` file per family in `data/` (10 families × 60 000 curves = 600 000 total).

### 2. Normalise

```python
import torch, os

families = sorted(f for f in os.listdir("data") if f.endswith(".pt"))
X = torch.cat([torch.load(f"data/{f}", weights_only=True) for f in families], dim=0)
print(f"Combined: {X.shape}")  # (600000, 200)

mu    = X.mean(dim=0)
sigma = torch.clamp(X.std(dim=0), min=1e-6)
X_norm = (X - mu) / sigma

torch.save(X_norm, "X_norm.pt")
torch.save({"mu": mu, "sigma": sigma}, "normalization.pt")
```

### 3. Train the diffusion model

```bash
cd Diffusion_Source
python DiffusionModel.py
```

### 4. Sample and analyse

See the **[companion repository](https://github.com/JuliaYu24/GFF_Diffusion_Analysis)** for the full sampling and analysis pipeline.

## Citation

If you use this code, please cite:

```bibtex
@article{Alharazin:2026wfh,
    author        = "Alharazin, Herzallah and Panteleeva, Julia Yu.",
    title         = "{Reconstruction of Gravitational Form Factors using Generative Machine Learning}",
    eprint        = "2602.19267",
    archivePrefix = "arXiv",
    primaryClass  = "hep-ph",
    month         = "2",
    year          = "2026"
}
```

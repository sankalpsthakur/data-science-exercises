#!/usr/bin/env python
"""
pip install numpy matplotlib geomstats torch torchvision lpips tqdm scikit-learn


visualise_latent_spaces.py

Demo: Euclidean, Hypersphere, Torus, Hyperbolic (Poincaré), Learned Riemannian
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from tqdm import tqdm

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from lpips import LPIPS

# ─── Config ───────────────────────────────────────────────────────────────────
SEED       = 0
DIM        = 3          # latent dim for demo
N_SAMPLES  = 200        # how many points per topology
IMG_SIZE   = 64         # dummy images (gradient squares)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

lpips_fn = LPIPS(net='vgg').to(DEVICE).eval()  # perceptual metric

# ─── Helper: make tiny dummy “images” so LPIPS has something to chew on ───────
def make_dummy_images(n):
    imgs = []
    for i in range(n):
        g = np.linspace(0, 1, IMG_SIZE)[None, None, :]               # horizontal grad
        r = np.full_like(g, (i / n))                                # varies with idx
        b = np.flip(g, -1)
        img = np.concatenate([r, g, b], axis=1)                     # (3,1,W)
        img = np.repeat(img, IMG_SIZE, axis=2)                      # (3,1,W)->(3,1,W)
        imgs.append(torch.tensor(img, dtype=torch.float32))
    return torch.stack(imgs).to(DEVICE)

dummy_imgs = make_dummy_images(N_SAMPLES)

# ─── Geodesic distance primitives ─────────────────────────────────────────────
def dist_euclidean(a, b):        # ℝⁿ
    return np.linalg.norm(a - b, axis=-1)

sphere = Hypersphere(dim=DIM-1)
def dist_sphere(a, b):           # 𝕊ⁿ
    return sphere.metric.dist(a, b)

def dist_torus(a, b):            # (𝕊¹)ᵏ  with angles in radians
    Δ = np.abs(a - b)
    Δ = np.minimum(Δ, 2*np.pi - Δ)
    return np.linalg.norm(Δ, axis=-1)

ball = PoincareBall(dim=DIM)
def dist_poincare(a, b):         # ℍⁿ  (Poincaré ball coords)
    return ball.metric.dist(a, b)

# Learned manifold placeholder: here we just reuse Euclidean for illustration
def dist_learned(a, b):
    return dist_euclidean(a, b)  # replace with Riemannian shooting if you have a generator

# ─── Sampling functions ───────────────────────────────────────────────────────
def sample_euclidean(n):
    return np.random.randn(n, DIM)

def sample_sphere(n):
    x = np.random.randn(n, DIM)
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def sample_torus(n):
    return np.random.uniform(0, 2*np.pi, size=(n, DIM))   # angles

def sample_poincare(n, r_max=0.8):
    v = np.random.randn(n, DIM)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    radii = np.random.uniform(0, r_max, size=(n, 1))
    return v * radii

def sample_learned(n):
    return sample_euclidean(n)   # stand‑in

# ─── Pack into a registry so we can iterate ───────────────────────────────────
TOPOLOGIES = {
    "Euclidean ℝⁿ"  : (sample_euclidean,  dist_euclidean),
    "Sphere 𝕊ⁿ"     : (sample_sphere,     dist_sphere),
    "Torus (𝕊¹)ᵏ"  : (sample_torus,      dist_torus),
    "Poincaré ℍⁿ"   : (sample_poincare,   dist_poincare),
    "Learned (demo)": (sample_learned,    dist_learned),
}

# ─── Main visualisation loop ──────────────────────────────────────────────────
fig, axes = plt.subplots(len(TOPOLOGIES), 3, figsize=(15, 3*len(TOPOLOGIES)))
axes = axes if axes.ndim == 2 else axes[None, :]  # handle single row

for row, (name, (sampler, dfunc)) in enumerate(TOPOLOGIES.items()):
    Z = sampler(N_SAMPLES)
    
    # pairwise latent distances
    D_lat = np.zeros((N_SAMPLES, N_SAMPLES))
    for i in range(N_SAMPLES):
        D_lat[i] = dfunc(Z[i], Z)
    
    # perceptual distances (LPIPS)
    with torch.no_grad():
        D_perc = torch.zeros((N_SAMPLES, N_SAMPLES), device=DEVICE)
        for i in range(N_SAMPLES):
            d = lpips_fn(dummy_imgs[i].unsqueeze(0), dummy_imgs).squeeze()
            D_perc[i] = d
        D_perc = D_perc.cpu().numpy()
    
    # ── (A) Heat‑map ──────────────────────────────────────────────────────────
    ax = axes[row, 0]
    im = ax.imshow(D_lat, cmap='viridis')
    ax.set_title(f"{name}\nLatent geodesic")
    ax.set_xticks([]); ax.set_yticks([])
    
    # ── (B) MDS embedding coloured by perceptual dist to item 0 ───────────────
    X_2d = MDS(dissimilarity='precomputed', n_components=2, random_state=SEED)\
           .fit_transform(D_lat)
    ax = axes[row, 1]
    sc = ax.scatter(X_2d[:,0], X_2d[:,1], c=D_perc[0], s=40, cmap='plasma')
    ax.set_title("2‑D MDS • colour = LPIPS to #0")
    ax.axis('off')
    
    # ── (C) One geodesic path between two far points ─────────────────────────
    a, b = np.argmax(D_lat[0]), np.argmin(D_lat[0])       # farthest & nearest
    zs   = np.vstack([Z[a], Z[b]])
    
    # naïve straight line for demo; replace with shooting method if curved
    line = np.linspace(0, 1, 100)[:,None]*zs[1] + (1-line[:,None])*zs[0]
    ax = axes[row, 2]
    if DIM == 2:
        ax.plot(Z[:,0], Z[:,1], '.', alpha=.3)
        ax.plot(line[:,0], line[:,1], lw=2)
    else:
        # use first two dims
        ax.plot(Z[:,0], Z[:,1], '.', alpha=.3)
        ax.plot(line[:,0], line[:,1], lw=2)
    ax.set_title("Geodesic (proj.)")
    ax.axis('off')

fig.suptitle("Latent‑space topologies: distance structure & geodesics", y=1.02)
plt.tight_layout()
plt.show()

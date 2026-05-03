import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum
from kornia.augmentation import RandomGaussianNoise
from sklearn.decomposition import DictionaryLearning
from scipy.ndimage import label as scipy_label
import json, os

from image import Model, MNIST

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

OUT          = "exercises/priors"
ALPHA        = 0.1          # fixed — best from prior sweep
N_ATOMS_LIST = [6, 8, 10]  # sweep: forced sharing → memorization transition

# ── Build target tensor ───────────────────────────────────────────────────────
train_data = MNIST(train=True,  device=device)
test_data  = MNIST(train=False, device=device)
model = Model.from_config(epochs=20).to(device)
try:
    model.fit(train_data, test_data, RandomGaussianNoise(std=0.4))
except Exception:
    pass
print(f"val acc: {(model(test_data.x).argmax(-1) == test_data.y).float().mean().item():.4f}")

with torch.no_grad():
    wl, wr = model.w_lr[0].unbind()
    wl_px  = wl @ model.w_e
    wr_px  = wr @ model.w_e
    target = einsum(model.w_u, wl_px, wr_px, "c o, o i, o j -> c i j")
    target = 0.5 * (target + target.mT)   # (10, 784, 784)

# ── Metrics ───────────────────────────────────────────────────────────────────
def d_entropy(D):
    weights = torch.tensor(D).abs()
    weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-12)
    log_w   = weights.clamp(min=1e-12).log()
    return (-(weights * log_w).sum(dim=0)).mean().item()

def spatial_coherence(imgs_nk, threshold_pct=75):
    scores = []
    for k in range(imgs_nk.shape[1]):
        img = imgs_nk[:, k].reshape(28, 28)
        magnitude = np.abs(img)
        thresh = np.percentile(magnitude, threshold_pct)
        binary = magnitude > thresh
        n_active = binary.sum()
        if n_active == 0:
            scores.append(0.0); continue
        labeled, n_reg = scipy_label(binary)
        largest = max(np.sum(labeled == i) for i in range(1, n_reg + 1))
        scores.append(largest / n_active)
    return float(np.mean(scores))

def recon_cos_sim(B_flat_np, codes, atoms):
    B_recon = codes @ atoms
    B_t = torch.tensor(B_flat_np)
    B_r = torch.tensor(B_recon)
    return (B_t * B_r).sum() / (B_t.norm() * B_r.norm())

# ── Flatten B to (10, 784*784) ────────────────────────────────────────────────
B_flat = target.reshape(10, 784 * 784).cpu().numpy()
print(f"\nB_flat shape: {B_flat.shape}  (10 classes x 784^2 pixels)")

# ── Fit / cache per n_atoms ───────────────────────────────────────────────────
results = {}
for n_atoms in N_ATOMS_LIST:
    cache_pt   = f"{OUT}/dictlearn_cache_{n_atoms}.pt"
    cache_json = f"{OUT}/dictlearn_cache_{n_atoms}.json"

    if os.path.exists(cache_pt) and os.path.exists(cache_json):
        print(f"\n=== Loading cached n_atoms={n_atoms} ===")
        dl_tensors = torch.load(cache_pt, map_location='cpu', weights_only=False)
        with open(cache_json, encoding='utf-8') as f:
            meta = json.load(f)
        results[n_atoms] = {
            'codes':    dl_tensors['codes'].numpy(),
            'atoms':    dl_tensors['atoms'].numpy(),
            'cos_sim':  meta['cos_sim'],
            'mean_nnz': meta['mean_nnz'],
        }
        print(f"  cos_sim={meta['cos_sim']:.4f}  mean_nnz={meta['mean_nnz']:.1f}")
    else:
        print(f"\n=== Fitting n_atoms={n_atoms}, alpha={ALPHA} ===")
        dl = DictionaryLearning(
            n_components=n_atoms,
            alpha=ALPHA,
            fit_algorithm='lars',
            transform_algorithm='lasso_lars',
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )
        codes = dl.fit_transform(B_flat)   # (10, n_atoms)
        atoms = dl.components_              # (n_atoms, 784*784)

        cos_sim  = recon_cos_sim(B_flat, codes, atoms).item()
        mean_nnz = float((np.abs(codes) > 1e-6).sum(axis=1).mean())
        print(f"  cos_sim={cos_sim:.4f}  mean_nnz={mean_nnz:.1f}")

        torch.save({'codes': torch.tensor(codes), 'atoms': torch.tensor(atoms)}, cache_pt)
        with open(cache_json, 'w', encoding='utf-8') as f:
            json.dump({'cos_sim': cos_sim, 'mean_nnz': mean_nnz, 'alpha': ALPHA, 'n_atoms': n_atoms}, f, indent=2)

        results[n_atoms] = {'codes': codes, 'atoms': atoms, 'cos_sim': cos_sim, 'mean_nnz': mean_nnz}

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'n_atoms':<10}  cos_sim   mean_nnz   sharing_ratio")
print(f"{'-'*10}  -------   --------   -------------")
for n_atoms, r in results.items():
    sharing = r['mean_nnz'] / n_atoms  # fraction of atoms each class uses
    print(f"{n_atoms:<10}  {r['cos_sim']:.4f}    {r['mean_nnz']:.1f}        {sharing:.2f}")

# ── Plot each n_atoms ─────────────────────────────────────────────────────────
for n_atoms, r in results.items():
    codes = r['codes']   # (10, n_atoms)
    atoms = r['atoms']   # (n_atoms, 784*784)

    # All atoms shown (n_atoms <= 10)
    n_show = n_atoms

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 2.4, 8.5),
                             gridspec_kw={'height_ratios': [3, 3, 2]})
    if n_show == 1:
        axes = axes[:, np.newaxis]
    fig.patch.set_facecolor('#f8f8f8')

    # Sort atoms by total code weight
    atom_importance = np.abs(codes).sum(axis=0)
    sorted_idx = atom_importance.argsort()[::-1].tolist()

    for col, k in enumerate(sorted_idx):
        A = atoms[k].reshape(784, 784)
        A_sym = 0.5 * (A + A.T)
        vals_k, vecs_k = np.linalg.eigh(A_sym)

        top_pos = vecs_k[:, -1].reshape(28, 28)
        top_neg = vecs_k[:,  0].reshape(28, 28)
        vmax = max(abs(top_pos).max(), abs(top_neg).max()) * 0.9

        axes[0, col].imshow(top_pos, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[1, col].imshow(top_neg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[0, col].axis('off');  axes[1, col].axis('off')
        axes[0, col].set_title(f'atom {k}\nrank={int((np.abs(vals_k) > 0.01*np.abs(vals_k).max()).sum())}',
                               fontsize=6, pad=1)

        c_vals = codes[:, k]
        colors = ['#2980b9' if v > 0 else '#c0392b' for v in c_vals]
        ax = axes[2, col]
        ax.bar(range(10), c_vals, color=colors, width=0.7)
        ax.axhline(0, color='#888', linewidth=0.5)
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(10)], fontsize=5)
        ax.tick_params(axis='y', labelsize=5)
        ax.set_xlim(-0.5, 9.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0, 0].set_ylabel('+eig\n(support)', fontsize=8, rotation=0, ha='right', va='center', labelpad=55)
    axes[1, 0].set_ylabel('-eig\n(suppress)', fontsize=8, rotation=0, ha='right', va='center', labelpad=55)
    axes[2, 0].set_ylabel('code\n(class use)', fontsize=8, rotation=0, ha='right', va='center', labelpad=55)

    sharing = r['mean_nnz'] / n_atoms
    fig.suptitle(
        f"Dictionary learning  n_atoms={n_atoms}, alpha={ALPHA}\n"
        f"cos_sim={r['cos_sim']:.4f}   mean_nnz={r['mean_nnz']:.1f}/{n_atoms}   sharing={sharing:.2f}",
        fontsize=10, y=1.02
    )
    plt.tight_layout(w_pad=0.2, h_pad=0.5)
    out = f"{OUT}/dictlearn_atoms_{n_atoms}.png"
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")

    # Code heatmap (atoms x classes)
    fig, ax = plt.subplots(figsize=(8, max(2, n_atoms * 0.4)))
    im = ax.imshow(codes.T, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(codes).max(), vmax=np.abs(codes).max())
    ax.set_xlabel('Digit class (0-9)', fontsize=9)
    ax.set_ylabel('Atom index', fontsize=9)
    ax.set_xticks(range(10))
    ax.set_yticks(range(n_atoms))
    ax.set_title(f'Sparse codes  n_atoms={n_atoms}, alpha={ALPHA}, mean_nnz={r["mean_nnz"]:.1f}', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    out2 = f"{OUT}/dictlearn_codes_{n_atoms}.png"
    plt.savefig(out2, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out2}")

print("Done.")

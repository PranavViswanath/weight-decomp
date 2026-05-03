import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kornia.augmentation import RandomGaussianNoise
from einops import einsum
import json, os

from image import Model, MNIST

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

OUT        = "exercises/priors"
MODEL_CACHE = f"{OUT}/model.pt"
N_ATOMS    = 6   # use the interpretable regime

# ── Load or train model ───────────────────────────────────────────────────────
train_data = MNIST(train=True,  device=device)
test_data  = MNIST(train=False, device=device)
model = Model.from_config(epochs=20).to(device)
if os.path.exists(MODEL_CACHE):
    model.load_state_dict(torch.load(MODEL_CACHE, map_location=device, weights_only=False))
    print("Loaded model from cache.")
else:
    try:
        model.fit(train_data, test_data, RandomGaussianNoise(std=0.4))
    except Exception:
        pass
    torch.save(model.state_dict(), MODEL_CACHE)
    print("Trained and cached model.")
print(f"val acc: {(model(test_data.x).argmax(-1) == test_data.y).float().mean().item():.4f}")

# ── Load DL atoms and codes ───────────────────────────────────────────────────
cache_pt = f"{OUT}/dictlearn_cache_{N_ATOMS}.pt"
dl = torch.load(cache_pt, map_location='cpu', weights_only=False)
atoms = dl['atoms'].numpy()   # (n_atoms, 784*784)
codes = dl['codes'].numpy()   # (10, n_atoms)  — one code vector per digit class

print(f"\nAtoms: {atoms.shape}  Codes: {codes.shape}")
print("\nCode matrix (rows=classes, cols=atoms):")
print("       " + "  ".join(f"atm{k}" for k in range(N_ATOMS)))
for c in range(10):
    row = "  ".join(f"{codes[c,k]:+.2f}" for k in range(N_ATOMS))
    print(f"  cls{c}  {row}")

# ── Compute per-input atom activations s_k(x) = x^T A_k x ───────────────────
# x: (N, 784) test images, A_k: (784, 784) reshaped from atoms[k]
x = test_data.x.cpu().numpy().reshape(len(test_data.x), -1)  # (N, 784)
y = test_data.y.cpu().numpy()   # (N,)
N = x.shape[0]

print(f"\nComputing atom activations on {N} test images...")
S = np.zeros((N, N_ATOMS))  # s_k(x_i)
for k in range(N_ATOMS):
    A = atoms[k].reshape(784, 784)
    A_sym = 0.5 * (A + A.T)
    # s_k(x) = x^T A_sym x  for each x in batch
    Ax = x @ A_sym          # (N, 784)
    S[:, k] = (x * Ax).sum(axis=1)  # (N,)
    print(f"  atom {k} done  range=[{S[:,k].min():.3f}, {S[:,k].max():.3f}]")

# ── Per-class mean activation ─────────────────────────────────────────────────
print("\nMean atom activation per digit class:")
print("       " + "  ".join(f"atm{k}" for k in range(N_ATOMS)))
mean_S = np.zeros((10, N_ATOMS))
for c in range(10):
    mask = y == c
    mean_S[c] = S[mask].mean(axis=0)
    row = "  ".join(f"{mean_S[c,k]:+.3f}" for k in range(N_ATOMS))
    print(f"  cls{c}  {row}")

# ── Plot 1: mean activations heatmap ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

im0 = axes[0].imshow(codes, cmap='RdBu_r', aspect='auto',
                     vmin=-np.abs(codes).max(), vmax=np.abs(codes).max())
axes[0].set_title('Code matrix  α_ck  (weight-space)', fontsize=10)
axes[0].set_xlabel('Atom index'); axes[0].set_ylabel('Digit class')
axes[0].set_xticks(range(N_ATOMS)); axes[0].set_yticks(range(10))
plt.colorbar(im0, ax=axes[0], fraction=0.046)

vmax1 = np.abs(mean_S).max()
im1 = axes[1].imshow(mean_S, cmap='RdBu_r', aspect='auto', vmin=-vmax1, vmax=vmax1)
axes[1].set_title('Mean atom activation  E[s_k(x) | class c]  (forward-pass)', fontsize=10)
axes[1].set_xlabel('Atom index'); axes[1].set_ylabel('Digit class')
axes[1].set_xticks(range(N_ATOMS)); axes[1].set_yticks(range(10))
plt.colorbar(im1, ax=axes[1], fraction=0.046)

fig.suptitle(f'Weight-space codes vs forward-pass activations  (n_atoms={N_ATOMS})\n'
             f'If they agree, the decomposition predicts model behavior.',
             fontsize=10)
plt.tight_layout()
out = f"{OUT}/atom_activations_compare.png"
plt.savefig(out, dpi=140, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# ── Correlation between codes and mean activations ────────────────────────────
# Flatten both to vectors (100 entries each for 10x10 grid)
codes_flat    = codes.flatten()
mean_S_flat   = mean_S.flatten()
corr = np.corrcoef(codes_flat, mean_S_flat)[0, 1]
print(f"\nCorrelation between code matrix and mean activation matrix: r={corr:.4f}")
print("  r~1.0 => weight-space decomposition predicts forward-pass behavior")
print("  r~0.0 => atoms fire independently of class codes (decomposition is not mechanistic)")

# ── Plot 2: scatter codes vs mean activations ─────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(codes_flat, mean_S_flat, alpha=0.6, s=40)
lim = max(np.abs(codes_flat).max(), np.abs(mean_S_flat).max()) * 1.1
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.axhline(0, color='#888', lw=0.5); ax.axvline(0, color='#888', lw=0.5)
ax.set_xlabel('Code weight α_ck', fontsize=10)
ax.set_ylabel('Mean activation E[s_k(x)|class c]', fontsize=10)
ax.set_title(f'r={corr:.4f}  (n={len(codes_flat)} class×atom pairs)', fontsize=10)
plt.tight_layout()
out2 = f"{OUT}/atom_activations_scatter.png"
plt.savefig(out2, dpi=140, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ── Plot 3: per-class breakdown — predicted score vs actual logit ─────────────
# predicted score for class c on image x: sum_k alpha[c,k] * s_k(x)
pred_scores = S @ codes.T   # (N, 10)  — reconstructed logits from DL decomposition

with torch.no_grad():
    true_logits = model(test_data.x).cpu().numpy()  # (N, 10)

corr_logits = np.corrcoef(pred_scores.flatten(), true_logits.flatten())[0, 1]
print(f"\nCorrelation between DL-reconstructed logits and true model logits: r={corr_logits:.4f}")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for c in range(10):
    ax = axes[c // 5, c % 5]
    ax.scatter(true_logits[:, c], pred_scores[:, c], alpha=0.1, s=5)
    ax.set_title(f'digit {c}', fontsize=9)
    ax.set_xlabel('true logit', fontsize=7)
    ax.set_ylabel('DL pred', fontsize=7)
    r = np.corrcoef(true_logits[:, c], pred_scores[:, c])[0, 1]
    ax.text(0.05, 0.92, f'r={r:.3f}', transform=ax.transAxes, fontsize=7)
fig.suptitle(f'DL-reconstructed logits vs true model logits  (n_atoms={N_ATOMS}, cos_sim of B=0.9314)\n'
             f'Overall r={corr_logits:.4f}', fontsize=10)
plt.tight_layout()
out3 = f"{OUT}/atom_activations_logits.png"
plt.savefig(out3, dpi=140, bbox_inches='tight')
plt.close()
print(f"Saved: {out3}")
print("Done.")

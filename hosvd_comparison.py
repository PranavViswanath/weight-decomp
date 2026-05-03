import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from kornia.augmentation import RandomGaussianNoise

from image import Model, MNIST

device = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT  = "exercises/priors"
RANK = 64

# ── Train base model ──────────────────────────────────────────────────────────
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
    target = 0.5 * (target + target.mT)

# ── Plain CP (rank 64, 200 steps) — cached ───────────────────────────────────
CP_CACHE = f"{OUT}/cp_factors.pt"
import os

if os.path.exists(CP_CACHE):
    print(f"\n=== Loading cached CP factors from {CP_CACHE} ===")
    ckpt = torch.load(CP_CACHE, map_location='cpu')
    L, R, D_cp = ckpt['L'], ckpt['R'], ckpt['D']
    sim_val = ckpt['cos_sim']
    print(f"  CP cos_sim={sim_val:.4f}")
else:
    print("\n=== Training plain CP (will cache after) ===")
    torch.manual_seed(42)
    L = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
    R = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
    D = nn.Parameter(torch.randn(10,  RANK, device=device) / 10**0.5)

    try:
        opt = torch.optim.Muon([L, R, D], lr=0.02, momentum=0.95)
    except AttributeError:
        opt = torch.optim.AdamW([L, R, D], lr=0.002)
    sched = CosineAnnealingLR(opt, T_max=200)

    torch.set_grad_enabled(True)
    for _ in tqdm(range(200), desc="plain CP"):
        pred = einsum(D, L, R, "c r, i r, j r -> c i j")
        pred = 0.5 * (pred + pred.mT)
        sim  = einsum(target, pred, "c i j, c i j ->") / (target.norm() * pred.norm())
        loss = 1 - sim
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.set_grad_enabled(False)

    L, R, D_cp = L.detach().cpu(), R.detach().cpu(), D.detach().cpu()
    sim_val = sim.item()
    torch.save({'L': L, 'R': R, 'D': D_cp, 'cos_sim': sim_val}, CP_CACHE)
    print(f"  CP cos_sim={sim_val:.4f}  (cached to {CP_CACHE})")

sigma = L.norm(dim=0) * R.norm(dim=0) * D_cp.norm(dim=0)
cp_top = sigma.argsort(descending=True)[:8].tolist()

# ── HOSVD (Appendix B.2 of arXiv:2406.03947) ─────────────────────────────────
# Flatten B: (10, 784, 784) → (10, 784²), SVD, eigendecompose each Q_i
print("\n=== HOSVD (Appendix B.2) ===")
with torch.no_grad():
    B_flat = target.reshape(10, 784 * 784)           # (10, 614656)
    U, S, Vt = torch.linalg.svd(B_flat, full_matrices=False)
    # U: (10, 10), S: (10,), Vt: (10, 614656)
    # Reconstruction quality
    B_recon = (U * S) @ Vt
    cos_sim = (B_flat * B_recon).sum() / (B_flat.norm() * B_recon.norm())
    print(f"  HOSVD full-rank cos_sim={cos_sim.item():.4f}  (rank={S.shape[0]}, all singular values)")
    print(f"  Singular values: {S.cpu().numpy().round(2)}")

    hosvd_components = []  # list of (u_i, top_pos_vec, top_neg_vec, eig_vals)
    for i in range(10):
        Q = Vt[i].reshape(784, 784)
        Q = 0.5 * (Q + Q.T)          # symmetrize
        vals, vecs = torch.linalg.eigh(Q)   # ascending order
        # Project eigenvectors to pixel space (already in pixel space since target is in pixel space)
        hosvd_components.append((U[:, i].cpu(), vals.cpu(), vecs.cpu(), S[i].item()))

# ── Figure: HOSVD vs CP side by side ─────────────────────────────────────────
print("\n=== Building comparison figure ===")

N_COMP = 8

# We show: 3 rows for CP (L+R, L-R, D bars), 3 rows for HOSVD top-8 singular components
fig, axes = plt.subplots(6, N_COMP,
                         figsize=(N_COMP * 2.4, 14),
                         gridspec_kw={'height_ratios': [3, 3, 2, 3, 3, 2]})
fig.patch.set_facecolor('#f8f8f8')

# ── CP rows ───────────────────────────────────────────────────────────────────
for col, k in enumerate(cp_top):
    lk  = L[:, k]
    rk  = R[:, k]
    pos = (lk + rk).reshape(28, 28).numpy()
    neg = (lk - rk).reshape(28, 28).numpy()
    vmax = max(abs(pos).max(), abs(neg).max()) * 0.9

    axes[0, col].imshow(pos, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[1, col].imshow(neg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[0, col].axis('off'); axes[1, col].axis('off')
    axes[0, col].set_title(f'#{k}', fontsize=6, pad=1)

    d_vals = D_cp[:, k].numpy()
    colors = ['#2980b9' if v > 0 else '#c0392b' for v in d_vals]
    ax = axes[2, col]
    ax.bar(range(10), d_vals, color=colors, width=0.7)
    ax.axhline(0, color='#888', linewidth=0.5)
    ax.set_xticks(range(10)); ax.set_xticklabels([str(i) for i in range(10)], fontsize=5)
    ax.tick_params(axis='y', labelsize=5); ax.set_xlim(-0.5, 9.5)
    ax.spines[['top', 'right']].set_visible(False)

axes[0, 0].set_ylabel('CP\nL+R', fontsize=8, rotation=0, ha='right', va='center', labelpad=45)
axes[1, 0].set_ylabel('CP\nL−R', fontsize=8, rotation=0, ha='right', va='center', labelpad=45)
axes[2, 0].set_ylabel('CP\nD', fontsize=8, rotation=0, ha='right', va='center', labelpad=45)

# ── HOSVD rows ────────────────────────────────────────────────────────────────
# For each of top-8 singular components, show top positive eigenvector of Q_i
for col in range(N_COMP):
    u_i, vals_i, vecs_i, sigma_i = hosvd_components[col]

    # top positive and negative eigenvectors
    top_pos = vecs_i[:, -1].numpy().reshape(28, 28)   # largest eigenvalue
    top_neg = vecs_i[:,  0].numpy().reshape(28, 28)   # most negative eigenvalue
    vmax = max(abs(top_pos).max(), abs(top_neg).max()) * 0.9

    axes[3, col].imshow(top_pos, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[4, col].imshow(top_neg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[3, col].axis('off'); axes[4, col].axis('off')
    axes[3, col].set_title(f'σ={sigma_i:.1f}', fontsize=6, pad=1)

    u_vals = u_i.numpy()
    colors = ['#2980b9' if v > 0 else '#c0392b' for v in u_vals]
    ax = axes[5, col]
    ax.bar(range(10), u_vals, color=colors, width=0.7)
    ax.axhline(0, color='#888', linewidth=0.5)
    ax.set_xticks(range(10)); ax.set_xticklabels([str(i) for i in range(10)], fontsize=5)
    ax.tick_params(axis='y', labelsize=5); ax.set_xlim(-0.5, 9.5)
    ax.spines[['top', 'right']].set_visible(False)

axes[3, 0].set_ylabel('HOSVD\n+eig', fontsize=8, rotation=0, ha='right', va='center', labelpad=45)
axes[4, 0].set_ylabel('HOSVD\n−eig', fontsize=8, rotation=0, ha='right', va='center', labelpad=45)
axes[5, 0].set_ylabel('HOSVD\nU', fontsize=8, rotation=0, ha='right', va='center', labelpad=45)

fig.suptitle("CP decomposition vs HOSVD (Appendix B.2, arXiv:2406.03947)\n"
             "CP: non-orthogonal components via gradient descent  |  "
             "HOSVD: orthogonal singular components via SVD + eigendecomposition",
             fontsize=9, y=1.002)
plt.tight_layout(w_pad=0.2, h_pad=0.8)
out = f"{OUT}/hosvd_vs_cp.png"
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")
print("Done.")

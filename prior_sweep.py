import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from kornia.augmentation import RandomGaussianNoise

from image import Model, MNIST

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0) if device.startswith('cuda') else 'n/a'}")

OUT   = "exercises/priors"
RANK  = 64
N_SHOW = 8   # components to visualize per condition

# ── Train base bilinear model ─────────────────────────────────────────────────
print("\n=== Training base bilinear model (20 epochs) ===")
train_data = MNIST(train=True,  device=device)
test_data  = MNIST(train=False, device=device)

model = Model.from_config(epochs=20).to(device)
try:
    model.fit(train_data, test_data, RandomGaussianNoise(std=0.4))
except Exception:
    pass
val_acc = (model(test_data.x).argmax(-1) == test_data.y).float().mean().item()
print(f"  val acc: {val_acc:.4f}")

# ── Precompute target tensor (same for all conditions) ────────────────────────
with torch.no_grad():
    wl, wr = model.w_lr[0].unbind()
    wl_px  = wl @ model.w_e    # (256,784)
    wr_px  = wr @ model.w_e
    target = einsum(model.w_u, wl_px, wr_px, "c o, o i, o j -> c i j")
    target = 0.5 * (target + target.mT)   # (10,784,784) symmetrized

def tensor_similarity(L, R, D):
    """Cosine similarity of CP reconstruction to target tensor. Returns tensor (keeps grad)."""
    pred = einsum(D, L, R, "c r, i r, j r -> c i j")
    pred = 0.5 * (pred + pred.mT)
    return einsum(target, pred, "c i j, c i j ->") / (target.norm() * pred.norm())

def cp_accuracy(L, R, D):
    with torch.no_grad():
        xf = test_data.x.flatten(start_dim=1)
        logits = ((xf @ L) * (xf @ R)) @ D.T
        return (logits.argmax(-1) == test_data.y).float().mean().item()

def tv_reg(M):
    """Total variation of M's columns reshaped to 28×28."""
    imgs = M.T.reshape(RANK, 28, 28)   # (64, 28, 28)
    dx   = (imgs[:, :, 1:] - imgs[:, :, :-1]).pow(2).sum()
    dy   = (imgs[:, 1:, :] - imgs[:, :-1, :]).pow(2).sum()
    return (dx + dy) / imgs.numel()

def top_components(L, R, D, k=N_SHOW):
    """Return indices of top-k components by σ = ‖L_k‖·‖R_k‖·‖D[:,k]‖."""
    sigma = L.norm(dim=0) * R.norm(dim=0) * D.norm(dim=0)
    return sigma.argsort(descending=True)[:k]

def get_optimizer(params):
    try:
        return torch.optim.Muon(params, lr=0.02, momentum=0.95), "Muon"
    except AttributeError:
        return torch.optim.AdamW(params, lr=0.002), "AdamW"

# ────────────────────────────────────────────────────────────────────────────
# Model classes
# ────────────────────────────────────────────────────────────────────────────

class PlainCP(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.left  = nn.Parameter(torch.randn(784, RANK) / 784**0.5)
        self.right = nn.Parameter(torch.randn(784, RANK) / 784**0.5)
        self.down  = nn.Parameter(torch.randn(10,  RANK) / 10**0.5)

    def forward(self, x):
        xf = x.flatten(start_dim=1)
        return ((xf @ self.left) * (xf @ self.right)) @ self.down.T

    def recon_loss(self):
        return 1 - tensor_similarity(self.left, self.right, self.down)

    def reg_loss(self):
        return torch.tensor(0., device=device)

class L1CP(PlainCP):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def reg_loss(self):
        return self.alpha * self.down.abs().mean()

class TVCP(PlainCP):
    def __init__(self, lam_tv=5e-3, alpha=0.02):
        super().__init__()
        self.lam_tv = lam_tv
        self.alpha  = alpha

    def reg_loss(self):
        return (self.lam_tv * (tv_reg(self.left) + tv_reg(self.right))
                + self.alpha * self.down.abs().mean())

class NonnegCP(nn.Module):
    """Non-negative CP via softplus reparameterization of L and R."""
    def __init__(self, alpha=0.02):
        super().__init__()
        torch.manual_seed(42)
        # init raw params so softplus(raw) ≈ N(0, 1/d) magnitude
        self.left_raw  = nn.Parameter(torch.randn(784, RANK) / 784**0.5)
        self.right_raw = nn.Parameter(torch.randn(784, RANK) / 784**0.5)
        self.down      = nn.Parameter(torch.randn(10,  RANK) / 10**0.5)
        self.alpha     = alpha

    @property
    def left(self):
        return F.softplus(self.left_raw)

    @property
    def right(self):
        return F.softplus(self.right_raw)

    def forward(self, x):
        xf = x.flatten(start_dim=1)
        return ((xf @ self.left) * (xf @ self.right)) @ self.down.T

    def recon_loss(self):
        return 1 - tensor_similarity(self.left, self.right, self.down)

    def reg_loss(self):
        return self.alpha * self.down.abs().mean()

# ────────────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────────────

def train(sparse, steps=200, label=""):
    opt, opt_name = get_optimizer(sparse.parameters())
    sched = CosineAnnealingLR(opt, T_max=steps)
    torch.set_grad_enabled(True)
    for _ in tqdm(range(steps), desc=f"{label} ({opt_name})"):
        loss = sparse.recon_loss() + sparse.reg_loss()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.set_grad_enabled(False)
    sim = tensor_similarity(sparse.left.detach(), sparse.right.detach(), sparse.down.detach()).item()
    acc = cp_accuracy(sparse.left.detach(), sparse.right.detach(), sparse.down.detach())
    print(f"  [{label}]  cos_sim={sim:.4f}  acc={acc:.4f}")
    return sim, acc

# ── Train all four conditions ─────────────────────────────────────────────────
print("\n=== Training CP conditions ===")

conditions = {}

plain = PlainCP().to(device)
sim_plain, acc_plain = train(plain, label="plain CP")
conditions["plain CP"]    = (plain, sim_plain, acc_plain)

l1 = L1CP(alpha=0.05).to(device)
sim_l1, acc_l1 = train(l1, label="L1 (α=0.05)")
conditions["L1 (α=0.05)"] = (l1, sim_l1, acc_l1)

tv = TVCP(lam_tv=5e-3, alpha=0.02).to(device)
sim_tv, acc_tv = train(tv, label="TV+L1 (λ=5e-3)")
conditions["TV+L1 (λ=5e-3)"] = (tv, sim_tv, acc_tv)

nonneg = NonnegCP(alpha=0.02).to(device)
sim_nn, acc_nn = train(nonneg, label="non-neg+L1")
conditions["non-neg+L1"] = (nonneg, sim_nn, acc_nn)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"\n  {'Condition':<22s}  cos_sim   task_acc")
print(f"  {'─'*22}  ───────   ────────")
for name, (_, sim, acc) in conditions.items():
    print(f"  {name:<22s}  {sim:.4f}    {acc:.4f}")

# ── Build figure ──────────────────────────────────────────────────────────────
print("\n=== Building figure ===")

N_COND = 4
fig, axes = plt.subplots(N_COND, N_SHOW * 2, figsize=(N_SHOW * 3.2, N_COND * 3.2))
fig.patch.set_facecolor('#f8f8f8')

row_labels = list(conditions.keys())

for row_idx, (cond_name, (sparse, sim, acc)) in enumerate(conditions.items()):
    L = sparse.left.detach().cpu()
    R = sparse.right.detach().cpu()
    D = sparse.down.detach().cpu()

    top_idx = top_components(L, R, D)   # (8,) indices of top components
    vmax    = max(L[:, top_idx].abs().max().item(),
                  R[:, top_idx].abs().max().item())
    vmax    = vmax * 0.9   # slight clip for contrast

    for col_pair, k in enumerate(top_idx):
        k = k.item()
        l_img = L[:, k].reshape(28, 28).numpy()
        r_img = R[:, k].reshape(28, 28).numpy()

        ax_l = axes[row_idx, col_pair * 2]
        ax_r = axes[row_idx, col_pair * 2 + 1]

        ax_l.imshow(l_img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax_r.imshow(r_img, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')

        # thin border to visually group L/R pairs
        for ax, side in [(ax_l, 'L'), (ax_r, 'R')]:
            ax.axis('off')
            if col_pair == 0 and side == 'L':
                ax.set_title(f'#{k}\nL', fontsize=6.5, pad=1, color='#333')
            elif col_pair == 0 and side == 'R':
                ax.set_title(f'\nR', fontsize=6.5, pad=1, color='#555')
            else:
                ax.set_title(f'#{k}\n{side}', fontsize=6.5, pad=1,
                             color='#333' if side == 'L' else '#555')

        # faint vertical separator between pairs (except last)
        if col_pair < N_SHOW - 1:
            ax_r.spines['right'].set_visible(True)
            ax_r.spines['right'].set_color('#cccccc')
            ax_r.spines['right'].set_linewidth(1.5)

    # row label on left
    axes[row_idx, 0].set_ylabel(
        f"{cond_name}\nsim={sim:.3f} acc={acc:.3f}",
        fontsize=8.5, rotation=0, ha='right', va='center',
        labelpad=80, color='#222'
    )

fig.suptitle(
    "CP decomposition priors — top-8 components by σ = ‖L_k‖·‖R_k‖·‖D[:,k]‖\n"
    "Each pair: L_k (left) · R_k (right), reshaped to 28×28. Diverging colormap (red=+, blue=−).",
    fontsize=10, y=1.01
)

plt.tight_layout(w_pad=0.1, h_pad=1.5)
out_path = f"{OUT}/prior_comparison.png"
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {out_path}")
print("\nDone.")

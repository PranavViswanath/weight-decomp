import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
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

def recon_loss(L, R, D):
    pred = einsum(D, L, R, "c r, i r, j r -> c i j")
    pred = 0.5 * (pred + pred.mT)
    return 1 - einsum(target, pred, "c i j, c i j ->") / (target.norm() * pred.norm())

def tv_reg(M):
    imgs = M.T.reshape(RANK, 28, 28)
    dx = (imgs[:, :, 1:] - imgs[:, :, :-1]).pow(2).sum()
    dy = (imgs[:, 1:, :] - imgs[:, :-1, :]).pow(2).sum()
    return (dx + dy) / imgs.numel()

def cp_accuracy(L, R, D):
    with torch.no_grad():
        xf = test_data.x.flatten(start_dim=1)
        return ((xf @ L) * (xf @ R) @ D.T).argmax(-1).eq(test_data.y).float().mean().item()

def top_components(L, R, D, k=8):
    sigma = L.norm(dim=0) * R.norm(dim=0) * D.norm(dim=0)
    return sigma.argsort(descending=True)[:k].tolist()

def make_params():
    torch.manual_seed(42)
    L = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
    R = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
    D = nn.Parameter(torch.randn(10,  RANK, device=device) / 10**0.5)
    return L, R, D

def get_opt(params):
    try:
        return torch.optim.Muon(params, lr=0.02, momentum=0.95)
    except AttributeError:
        return torch.optim.AdamW(params, lr=0.002)

def train(label, alpha_l1=0.0, lam_tv=0.0, nonneg=False, steps=200):
    if nonneg:
        torch.manual_seed(42)
        Lr = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
        Rr = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
        D  = nn.Parameter(torch.randn(10,  RANK, device=device) / 10**0.5)
        params = [Lr, Rr, D]
        get_L = lambda: F.softplus(Lr)
        get_R = lambda: F.softplus(Rr)
    else:
        Lp, Rp, D = make_params()
        params = [Lp, Rp, D]
        get_L = lambda: Lp
        get_R = lambda: Rp

    opt   = get_opt(params)
    sched = CosineAnnealingLR(opt, T_max=steps)
    torch.set_grad_enabled(True)
    for _ in tqdm(range(steps), desc=label):
        L, R = get_L(), get_R()
        rl = recon_loss(L, R, D)
        ll = alpha_l1 * (L.abs().mean() + R.abs().mean() + D.abs().mean()) if alpha_l1 > 0 else torch.tensor(0., device=device)
        tl = lam_tv * (tv_reg(L) + tv_reg(R)) if lam_tv > 0 else torch.tensor(0., device=device)
        loss = rl + ll + tl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.set_grad_enabled(False)
    Ld, Rd, Dd = get_L().detach().cpu(), get_R().detach().cpu(), D.detach().cpu()
    sim = (1 - recon_loss(Ld.to(device), Rd.to(device), Dd.to(device))).item()
    acc = cp_accuracy(Ld.to(device), Rd.to(device), Dd.to(device))
    print(f"  [{label}] cos_sim={sim:.4f} acc={acc:.4f}")
    return Ld, Rd, Dd, sim, acc

# ── Run all conditions ────────────────────────────────────────────────────────
print("\n=== Training all prior conditions ===")
results = {}
results["plain CP"]            = train("plain CP")
results["L1 α=0.05"]          = train("L1 α=0.05",          alpha_l1=0.05)
results["L1 α=0.5"]           = train("L1 α=0.5",           alpha_l1=0.5)
results["L1 α=1.0"]           = train("L1 α=1.0",           alpha_l1=1.0)
results["TV λ=0.05 pure"]     = train("TV λ=0.05 pure",     lam_tv=0.05)
results["TV λ=0.2 pure"]      = train("TV λ=0.2 pure",      lam_tv=0.2)
results["TV+L1 λ=0.05 α=0.5"] = train("TV+L1 λ=0.05 α=0.5", lam_tv=0.05, alpha_l1=0.5)
results["non-neg+L1 α=0.02"]  = train("non-neg+L1 α=0.02",  alpha_l1=0.02, nonneg=True)

# ── Build figure ──────────────────────────────────────────────────────────────
print("\n=== Building figure ===")
N_COND = len(results)
N_COMP = 8

fig, axes = plt.subplots(N_COND * 3, N_COMP,
                         figsize=(N_COMP * 2.4, N_COND * 7.0),
                         gridspec_kw={'height_ratios': [3, 3, 2] * N_COND})
fig.patch.set_facecolor('#f8f8f8')

for cond_idx, (cond_name, (L, R, D, sim, acc)) in enumerate(results.items()):
    top_idx = top_components(L, R, D)
    row_pos = cond_idx * 3
    row_neg = cond_idx * 3 + 1
    row_d   = cond_idx * 3 + 2

    for col, k in enumerate(top_idx):
        lk  = L[:, k]
        rk  = R[:, k]
        pos = (lk + rk).reshape(28, 28).numpy()
        neg = (lk - rk).reshape(28, 28).numpy()
        vmax = max(abs(pos).max(), abs(neg).max()) * 0.9

        axes[row_pos, col].imshow(pos, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[row_neg, col].imshow(neg, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
        axes[row_pos, col].axis('off')
        axes[row_neg, col].axis('off')
        axes[row_pos, col].set_title(f'#{k}', fontsize=6, pad=1)

        d_vals = D[:, k].numpy()
        colors = ['#2980b9' if v > 0 else '#c0392b' for v in d_vals]
        ax = axes[row_d, col]
        ax.bar(range(10), d_vals, color=colors, width=0.7)
        ax.axhline(0, color='#888', linewidth=0.5)
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(10)], fontsize=5)
        ax.tick_params(axis='y', labelsize=5)
        ax.set_xlim(-0.5, 9.5)
        ax.spines[['top', 'right']].set_visible(False)

    axes[row_pos, 0].set_ylabel(
        f"{cond_name}\nsim={sim:.3f} acc={acc:.3f}",
        fontsize=7.5, rotation=0, ha='right', va='center', labelpad=90
    )

fig.suptitle("All prior conditions — L+R (symmetric), L−R (antisymmetric), D weights\n"
             "Blue = positive contribution to digit, Red = negative",
             fontsize=10, y=1.001)
plt.tight_layout(w_pad=0.2, h_pad=0.5)
out = f"{OUT}/revis_all_priors.png"
plt.savefig(out, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")
print("Done.")

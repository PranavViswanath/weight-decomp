import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum
from scipy.ndimage import label as scipy_label
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from kornia.augmentation import RandomGaussianNoise

from image import Model, MNIST

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0) if device.startswith('cuda') else 'n/a'}")

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

# ── Metrics ───────────────────────────────────────────────────────────────────

def d_entropy(D):
    """Mean Shannon entropy of |D[:,k]| across all components.
    Low = each component votes for few digits. Max = log(10) ≈ 2.30."""
    weights = D.abs()                         # (10, rank)
    weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-12)
    log_w   = weights.clamp(min=1e-12).log()
    entropy = -(weights * log_w).sum(dim=0)   # (rank,)
    return entropy.mean().item()

def spatial_coherence(L, R, threshold_pct=75):
    """Mean over components of (largest connected region) / (total active pixels)
    in the L+R image. High = one connected blob. Low = scattered pixels."""
    scores = []
    for k in range(L.shape[1]):
        img = (L[:, k] + R[:, k]).reshape(28, 28).numpy()
        magnitude = np.abs(img)
        thresh = np.percentile(magnitude, threshold_pct)
        binary = magnitude > thresh
        n_active = binary.sum()
        if n_active == 0:
            scores.append(0.0)
            continue
        labeled, n_regions = scipy_label(binary)
        largest = max(np.sum(labeled == i) for i in range(1, n_regions + 1))
        scores.append(largest / n_active)
    return float(np.mean(scores))

def recon_loss(L, R, D):
    pred = einsum(D, L, R, "c r, i r, j r -> c i j")
    pred = 0.5 * (pred + pred.mT)
    return 1 - einsum(target, pred, "c i j, c i j ->") / (target.norm() * pred.norm())

def cp_accuracy(L, R, D):
    with torch.no_grad():
        xf = test_data.x.flatten(start_dim=1)
        return ((xf @ L) * (xf @ R) @ D.T).argmax(-1).eq(test_data.y).float().mean().item()

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

def train(label, alpha_l1=0.0, lam_tv=0.0, lam_hoyer=0.0, nonneg=False, steps=200):
    if nonneg:
        torch.manual_seed(42)
        Lr = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
        Rr = nn.Parameter(torch.randn(784, RANK, device=device) / 784**0.5)
        Dp = nn.Parameter(torch.randn(10,  RANK, device=device) / 10**0.5)
        params = [Lr, Rr, Dp]
        get_L = lambda: F.softplus(Lr)
        get_R = lambda: F.softplus(Rr)
        get_D = lambda: Dp
    else:
        Lp, Rp, Dp = make_params()
        params = [Lp, Rp, Dp]
        get_L = lambda: Lp
        get_R = lambda: Rp
        get_D = lambda: Dp

    opt   = get_opt(params)
    sched = CosineAnnealingLR(opt, T_max=steps)
    torch.set_grad_enabled(True)
    for _ in tqdm(range(steps), desc=label):
        L, R, D = get_L(), get_R(), get_D()
        imgs = L.T.reshape(RANK, 28, 28)
        rl = recon_loss(L, R, D)
        ll = alpha_l1 * (L.abs().mean() + R.abs().mean() + D.abs().mean()) if alpha_l1 > 0 else torch.tensor(0., device=device)
        hl = lam_hoyer * ((L.norm(p=1, dim=0) / L.norm(p=2, dim=0).clamp(1e-8)).mean()
                        + (R.norm(p=1, dim=0) / R.norm(p=2, dim=0).clamp(1e-8)).mean()) if lam_hoyer > 0 else torch.tensor(0., device=device)
        tl = (lam_tv * ((imgs[:, :, 1:] - imgs[:, :, :-1]).pow(2).sum()
                      + (imgs[:, 1:, :] - imgs[:, :-1, :]).pow(2).sum()) / imgs.numel()) if lam_tv > 0 else torch.tensor(0., device=device)
        loss = rl + ll + hl + tl
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.set_grad_enabled(False)

    Ld = get_L().detach().cpu()
    Rd = get_R().detach().cpu()
    Dd = get_D().detach().cpu()
    sim  = (1 - recon_loss(Ld.to(device), Rd.to(device), Dd.to(device))).item()
    acc  = cp_accuracy(Ld.to(device), Rd.to(device), Dd.to(device))
    ent  = d_entropy(Dd)
    coh  = spatial_coherence(Ld, Rd)
    return Ld, Rd, Dd, sim, acc, ent, coh

# ── HOSVD (for comparison) ─────────────────────────────────────────────────────
def hosvd_metrics():
    with torch.no_grad():
        B_flat = target.reshape(10, 784 * 784)
        U, S, Vt = torch.linalg.svd(B_flat, full_matrices=False)
        cos_sim = (B_flat * ((U * S) @ Vt)).sum() / (B_flat.norm() * ((U * S) @ Vt).norm())

        # Build L, R, D analogues: top eigenvector of each Q_i as a "component"
        # Use top positive eigenvector of each Q_i as the "feature"
        L_h = torch.zeros(784, 10)
        R_h = torch.zeros(784, 10)
        D_h = torch.zeros(10, 10)
        for i in range(10):
            Q = Vt[i].reshape(784, 784)
            Q = 0.5 * (Q + Q.T)
            vals, vecs = torch.linalg.eigh(Q)
            L_h[:, i] = vecs[:, -1]
            R_h[:, i] = vecs[:, -1]
            D_h[:, i] = U[:, i] * S[i]
        ent = d_entropy(D_h)
        coh = spatial_coherence(L_h, R_h)
    return cos_sim.item(), ent, coh

# ── Run all conditions (with disk cache) ──────────────────────────────────────
import os, json

CACHE_FILE = f"{OUT}/metrics_cache.pt"
CACHE_META = f"{OUT}/metrics_cache.json"

CONDITION_SPECS = [
    ("plain CP",             dict()),
    ("L1 a=0.05",            dict(alpha_l1=0.05)),
    ("L1 a=0.5",             dict(alpha_l1=0.5)),
    ("L1 a=1.0",             dict(alpha_l1=1.0)),
    ("TV l=0.05",            dict(lam_tv=0.05)),
    ("TV l=0.2",             dict(lam_tv=0.2)),
    ("TV+L1 l=0.05 a=0.5",  dict(lam_tv=0.05, alpha_l1=0.5)),
    ("Hoyer l=0.3",          dict(lam_hoyer=0.3)),
    ("non-neg+L1",           dict(alpha_l1=0.02, nonneg=True)),
]

if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_META):
    print(f"\n=== Loading cached factors from {CACHE_FILE} ===")
    cache_tensors = torch.load(CACHE_FILE, map_location='cpu')
    with open(CACHE_META, encoding="utf-8") as f:
        cache_meta = json.load(f)
    results = {}
    for name, _ in CONDITION_SPECS:
        key = name.replace(" ", "_")
        L = cache_tensors[f"{key}_L"]
        R = cache_tensors[f"{key}_R"]
        D = cache_tensors[f"{key}_D"]
        sim, acc = cache_meta[name]["sim"], cache_meta[name]["acc"]
        ent = d_entropy(D)
        coh = spatial_coherence(L, R)
        results[name] = (L, R, D, sim, acc, ent, coh)
        print(f"  loaded: {name}  cos_sim={sim:.4f}  acc={acc:.4f}  D_ent={ent:.4f}  coh={coh:.4f}")
else:
    print("\n=== Training all conditions (will cache after) ===")
    results = {}
    for name, kwargs in CONDITION_SPECS:
        results[name] = train(name, **kwargs)

    # Save to disk
    save_tensors = {}
    save_meta = {}
    for name, (L, R, D, sim, acc, ent, coh) in results.items():
        key = name.replace(" ", "_")
        save_tensors[f"{key}_L"] = L
        save_tensors[f"{key}_R"] = R
        save_tensors[f"{key}_D"] = D
        save_meta[name] = {"sim": sim, "acc": acc}
    torch.save(save_tensors, CACHE_FILE)
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump(save_meta, f, indent=2)
    print(f"\nCached to {CACHE_FILE}")

hosvd_sim, hosvd_ent, hosvd_coh = hosvd_metrics()

# ── Print summary ──────────────────────────────────────────────────────────────
print("\n=== Results ===")
print(f"\n  {'Condition':<26s}  cos_sim   acc     D_entropy  spatial_coh")
print(f"  {'-'*26}  -------   ----    ---------  -----------")
for name, (_, _, _, sim, acc, ent, coh) in results.items():
    flag = " *" if name == "plain CP" else ""
    print(f"  {name:<26s}  {sim:.4f}    {acc:.4f}  {ent:.4f}     {coh:.4f}{flag}")
print(f"  {'HOSVD':<26s}  {hosvd_sim:.4f}    {'n/a':<6s}  {hosvd_ent:.4f}     {hosvd_coh:.4f}")
print(f"\n  D_entropy: lower is better (max = log(10) = {np.log(10):.3f})")
print(f"  spatial_coh: higher is better (max = 1.0, one connected blob)")

# ── Plot: scatter of D_entropy vs spatial_coherence ───────────────────────────
print("\n=== Building figure ===")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#f8f8f8')

names  = list(results.keys()) + ["HOSVD"]
sims   = [v[3] for v in results.values()] + [hosvd_sim]
ents   = [v[5] for v in results.values()] + [hosvd_ent]
cohs   = [v[6] for v in results.values()] + [hosvd_coh]
colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

ax = axes[0]
for i, (name, ent, coh, c) in enumerate(zip(names, ents, cohs, colors)):
    ax.scatter(ent, coh, color=c, s=80, zorder=3)
    ax.annotate(name, (ent, coh), fontsize=7, xytext=(4, 2),
                textcoords='offset points', color='#333')
ax.set_xlabel("D entropy (↓ better: votes for fewer digits)", fontsize=9)
ax.set_ylabel("Spatial coherence (↑ better: one connected blob)", fontsize=9)
ax.set_title("Interpretability metrics: all conditions", fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#fafafa')

ax = axes[1]
x = np.arange(len(names))
width = 0.35
bars1 = ax.bar(x - width/2, ents, width, color=[c for c in colors], alpha=0.85, label='D entropy (↓)')
bars2 = ax.bar(x + width/2, cohs, width, color=[c for c in colors], alpha=0.45, label='spatial coh (↑)')
ax.axhline(np.log(10), color='#999', linewidth=0.8, linestyle='--', label=f'max entropy = {np.log(10):.2f}')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7)
ax.set_ylabel("Value", fontsize=9)
ax.set_title("Per-condition comparison", fontsize=10)
ax.legend(fontsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.set_facecolor('#fafafa')

fig.suptitle("Decomposition quality: D entropy (class specificity) and spatial coherence (blob-ness)\n"
             "Better = lower D entropy + higher spatial coherence, without sacrificing cos_sim",
             fontsize=9, y=1.02)
plt.tight_layout()
out = f"{OUT}/metrics_comparison.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")
print("Done.")

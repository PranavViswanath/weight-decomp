import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import einsum
from kornia.augmentation import RandomGaussianNoise
from image import Model, MNIST

device = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = "exercises/priors"
MODEL_CACHE = f"{OUT}/model.pt"

# ── Hand-written codes ────────────────────────────────────────────────────────
# Atom names (from visual inspection):
#   0: upper-curve    1: open-curve/S    2: closed-loop
#   3: hook-end       4: diagonal        5: double-loop
#
#               atm0   atm1   atm2   atm3   atm4   atm5
HAND = np.array([
    [ 0.0,  -0.1,  +1.0,   0.0,   0.0,   0.0],  # 0: round loop
    [-0.8,   0.0,  -0.5,   0.0,   0.0,  -0.3],  # 1: straight, no curves
    [-0.1,  +1.0,   0.0,   0.0,   0.0,   0.0],  # 2: S-curve
    [+0.7,   0.0,   0.0,  +0.7,   0.0,   0.0],  # 3: curves + hooks
    [-0.8,  -0.5,  -0.5,   0.0,  -0.6,   0.0],  # 4: angular, suppresses all
    [+1.1,   0.0,  -0.4,  -0.3,   0.0,   0.0],  # 5: top-arc, not loop
    [+0.2,   0.0,  +0.6,  -0.8,  -0.1,   0.0],  # 6: loop + no hook
    [-0.2,   0.0,   0.0,   0.0,  +1.0,   0.0],  # 7: diagonal
    [ 0.0,   0.0,   0.0,   0.0,   0.0,  +1.0],  # 8: double-loop
    [ 0.0,  -0.7,   0.0,  +0.2,   0.0,   0.0],  # 9: not S-curve + slight hook (loop not discriminative vs 0/6)
])

# ── Load model and atoms ──────────────────────────────────────────────────────
train_data = MNIST(train=True,  device=device)
test_data  = MNIST(train=False, device=device)
model = Model.from_config(epochs=20).to(device)
if os.path.exists(MODEL_CACHE):
    model.load_state_dict(torch.load(MODEL_CACHE, map_location=device, weights_only=False))
else:
    try:
        model.fit(train_data, test_data, RandomGaussianNoise(std=0.4))
    except Exception:
        pass
    torch.save(model.state_dict(), MODEL_CACHE)

dl = torch.load(f"{OUT}/dictlearn_cache_6.pt", map_location='cpu', weights_only=False)
atoms = dl['atoms'].numpy()   # (6, 614656)
true_codes = dl['codes'].numpy()  # (10, 6)

# ── Compare codes ─────────────────────────────────────────────────────────────
print(f"{'digit':<6} {'hand code':<46} {'true code':<46} {'cos_sim':>7}")
print("-" * 110)
for c in range(10):
    h = HAND[c]
    t = true_codes[c]
    cos = np.dot(h, t) / (np.linalg.norm(h) * np.linalg.norm(t) + 1e-12)
    print(f"  {c}    {str(np.round(h,1)):<46} {str(np.round(t,2)):<46} {cos:>7.4f}")

# ── Build all B matrices from model ──────────────────────────────────────────
with torch.no_grad():
    wl, wr = model.w_lr[0].unbind()
    wl_px  = wl @ model.w_e
    wr_px  = wr @ model.w_e
    B_model = einsum(model.w_u, wl_px, wr_px, "c o, o i, o j -> c i j")
    B_model = 0.5 * (B_model + B_model.mT)
    B_model = B_model.cpu().numpy()  # (10, 784, 784)

x = test_data.x.cpu().numpy().reshape(len(test_data.x), -1)
y = test_data.y.cpu().numpy()

# Baseline: model's own B matrices
scores_model = np.zeros((len(x), 10))
for c in range(10):
    Ax = x @ B_model[c]
    scores_model[:, c] = (x * Ax).sum(axis=1)

# Hand-written: replace ALL 10 B_c with hand-constructed versions
B_hand = np.stack([sum(HAND[c, k] * atoms[k].reshape(784, 784)
                       for k in range(6)) for c in range(10)])  # (10, 784, 784)

scores_hand = np.zeros((len(x), 10))
for c in range(10):
    Ax = x @ B_hand[c]
    scores_hand[:, c] = (x * Ax).sum(axis=1)

# ── Per-class accuracy ────────────────────────────────────────────────────────
preds_model = scores_model.argmax(axis=1)
preds_hand  = scores_hand.argmax(axis=1)

print(f"\n{'digit':<6} {'model acc':>10} {'hand acc':>10} {'delta':>8}")
print("-" * 40)
for c in range(10):
    mask = y == c
    acc_m = (preds_model[mask] == c).mean()
    acc_h = (preds_hand[mask]  == c).mean()
    print(f"  {c}    {acc_m:>10.4f} {acc_h:>10.4f} {acc_h - acc_m:>+8.4f}")

overall_model = (preds_model == y).mean()
overall_hand  = (preds_hand  == y).mean()
print(f"\n  overall: model={overall_model:.4f}  hand={overall_hand:.4f}  delta={overall_hand-overall_model:+.4f}")
print(f"\nChance = 10%. Hand classifier built entirely from visual reasoning.")

# ── Figure: hand vs true codes + per-class accuracy ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

atom_labels = ['atm0\nupper-curve', 'atm1\nopen-curve', 'atm2\nclosed-loop',
               'atm3\nhook-end', 'atm4\ndiagonal', 'atm5\ndbl-loop']

vmax = max(np.abs(HAND).max(), np.abs(true_codes).max())
im0 = axes[0].imshow(HAND, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
axes[0].set_title('Hand-written codes', fontsize=11)
axes[0].set_xticks(range(6)); axes[0].set_xticklabels(atom_labels, fontsize=7)
axes[0].set_yticks(range(10)); axes[0].set_yticklabels([str(i) for i in range(10)])
axes[0].set_ylabel('Digit class')
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(true_codes, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
axes[1].set_title('True DL codes (learned)', fontsize=11)
axes[1].set_xticks(range(6)); axes[1].set_xticklabels(atom_labels, fontsize=7)
axes[1].set_yticks(range(10)); axes[1].set_yticklabels([str(i) for i in range(10)])
plt.colorbar(im1, ax=axes[1], fraction=0.046)

accs_model = [(preds_model[y==c]==c).mean() for c in range(10)]
accs_hand  = [(preds_hand[y==c]==c).mean()  for c in range(10)]
x_pos = np.arange(10)
w = 0.35
axes[2].bar(x_pos - w/2, accs_model, w, label='model', color='#2980b9', alpha=0.8)
axes[2].bar(x_pos + w/2, accs_hand,  w, label='hand-written', color='#e74c3c', alpha=0.8)
axes[2].axhline(0.1, color='gray', linestyle='--', linewidth=0.8, label='chance (10%)')
axes[2].set_xticks(x_pos); axes[2].set_xticklabels([str(i) for i in range(10)])
axes[2].set_ylabel('Accuracy'); axes[2].set_ylim(0, 1.05)
axes[2].set_title('Per-class accuracy', fontsize=11)
axes[2].legend(fontsize=8)

fig.suptitle('Human-specified classifier via atom composition\n'
             '6 named atoms + hand-written codes → full 10-class classifier',
             fontsize=11)
plt.tight_layout()
out = f"{OUT}/handwrite_all10.png"
plt.savefig(out, dpi=140, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

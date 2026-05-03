import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'src')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = "exercises/priors"

# ── Load dict learning cache (n_atoms=6) ─────────────────────────────────────
dl = torch.load(f"{OUT}/dictlearn_cache_6.pt", map_location='cpu', weights_only=False)
atoms_raw = dl['atoms'].numpy()   # (6, 784*784)
true_codes = dl['codes'].numpy()  # (10, 6)

# ── Load model and test data ──────────────────────────────────────────────────
from image import Model, MNIST
model = Model.from_config(epochs=20).to(device)
model.load_state_dict(torch.load(f"{OUT}/model.pt", map_location=device, weights_only=False))
test_data = MNIST(train=False, device=device)

x_np = test_data.x.cpu().numpy()
y_np = test_data.y.cpu().numpy()
x_imgs = x_np.reshape(len(x_np), 28, 28)
# Normalize to [0,1] for display
if x_imgs.max() > 1.5:
    x_imgs = x_imgs / 255.0

# ── Get leading eigenvectors of each atom (matches dictlearn.py row 0) ────────
def get_top_eigvec(atom_flat):
    A = atom_flat.reshape(784, 784)
    A_sym = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A_sym)
    return vecs[:, -1].reshape(28, 28)   # largest eigenvalue = dominant pattern

eigvecs = [get_top_eigvec(atoms_raw[k]) for k in range(6)]

# ── Digit 6 examples ──────────────────────────────────────────────────────────
idx6 = np.where(y_np == 6)[0][:3]
digit6_imgs = x_imgs[idx6]

# ── Atom names and hand-written code for digit 6 (from handwrite_all10.py) ───
ATOM_NAMES  = ['upper-\ncurve', 'open-\ncurve', 'closed-\nloop',
               'hook-\nend', 'diag-\nonal', 'dbl-\nloop']
HAND_6 = np.array([+0.2, 0.0, +0.6, -0.8, -0.1, 0.0])

def code_style(c):
    """Border color, linewidth, title color, label string."""
    if abs(c) < 0.05:
        return '#aaaaaa', 1.2, '#888888', f'≈ 0\n(unused)'
    elif c > 0:
        strong = c >= 0.4
        ec = '#1a5276' if strong else '#2e86c1'
        lw = 3.8 if strong else 2.2
        tc = '#1a5276' if strong else '#2e86c1'
        return ec, lw, tc, f'+{c:.1f}\n(fires{"!" if strong else ""})'
    else:
        strong = abs(c) >= 0.4
        ec = '#7b241c' if strong else '#c0392b'
        lw = 3.8 if strong else 2.2
        tc = '#7b241c' if strong else '#c0392b'
        return ec, lw, tc, f'{c:.1f}\n(suppressed{"!" if strong else ""})'

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.5, 4.2), facecolor='white')
# Layout: [digit-6 examples | gap | 6 atom panels]
gs = GridSpec(1, 8, figure=fig, wspace=0.10,
              width_ratios=[1.5, 0.15, 1, 1, 1, 1, 1, 1])

# ── Panel 0: real digit 6 images stacked ────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
combined = np.concatenate([
    digit6_imgs[0],
    np.full((3, 28), 0.88),
    digit6_imgs[1],
    np.full((3, 28), 0.88),
    digit6_imgs[2],
], axis=0)
ax0.imshow(combined, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
ax0.set_title('digit 6\n(test set)', fontsize=9.5, fontweight='bold',
              color='#2c3e50', pad=5)
ax0.axis('off')
rect = plt.Rectangle((0, 0), 1, 1, transform=ax0.transAxes,
                      fill=False, edgecolor='#cccccc', linewidth=1.2)
ax0.add_patch(rect)

# ── Panel 1: arrow ───────────────────────────────────────────────────────────
ax_arr = fig.add_subplot(gs[0, 1])
ax_arr.axis('off')
ax_arr.text(0.5, 0.52, '→', ha='center', va='center',
            fontsize=22, color='#555555', transform=ax_arr.transAxes)

# ── Panels 2-7: atom eigenvectors ────────────────────────────────────────────
for k in range(6):
    ax = fig.add_subplot(gs[0, k + 2])
    v = eigvecs[k]
    vmax = np.abs(v).max() * 0.96
    ax.imshow(v, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    ec, lw, tc, label = code_style(HAND_6[k])
    for spine in ax.spines.values():
        spine.set_edgecolor(ec)
        spine.set_linewidth(lw)

    fw = 'bold' if abs(HAND_6[k]) >= 0.4 else 'normal'
    ax.set_title(f'atm {k}\n{ATOM_NAMES[k]}\n{label}',
                 fontsize=8, pad=4, color=tc, fontweight=fw,
                 linespacing=1.35)

# ── Legend ───────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color='#1a5276', label='fires (positive code)'),
    mpatches.Patch(color='#7b241c', label='suppressed (negative code)'),
    mpatches.Patch(color='#aaaaaa', label='inactive (≈ 0)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.62, -0.05), framealpha=0.92,
           edgecolor='#cccccc')

# ── Formula annotation ───────────────────────────────────────────────────────
fig.text(
    0.62, -0.14,
    r'digit 6  $\approx$  closed-loop ($\times$+0.6)  +  upper-curve ($\times$+0.2)  $-$  hook-end ($\times$0.8)',
    ha='center', fontsize=10.5, style='italic', color='#1c2833',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#eaf4fb',
              edgecolor='#aed6f1', alpha=0.95, linewidth=1.2)
)

fig.suptitle(
    'Decomposing digit 6 into learned atoms',
    fontsize=12, fontweight='bold', y=1.04, color='#1c2833'
)

out = f'{OUT}/digit6_decomposition.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')

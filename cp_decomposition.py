import sys
sys.path.insert(0, 'src')

import torch
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from kornia.augmentation import RandomGaussianNoise

from image import Model, MNIST, plot_eigenspectrum
from image.sparse import Model as Sparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
train, test = MNIST(train=True, device=device), MNIST(train=False, device=device)

# ── Train base bilinear model ─────────────────────────────────────────────────
print("\n=== Training bilinear MNIST model ===")
model = Model.from_config(epochs=20).to(device)
model.fit(train, test, RandomGaussianNoise(std=0.4))

# ── Eigendecomposition baseline ───────────────────────────────────────────────
print("\n=== Eigendecomposition baseline ===")
vals, vecs = model.decompose()
top_vecs = vecs[:, -1, :].cpu()

fig = px.imshow(
    top_vecs.view(10, 28, 28),
    color_continuous_midpoint=0, color_continuous_scale="RdBu",
    facet_col=0, title="Eigendecomposition: top positive eigenvector per class",
    facet_col_spacing=0.01,
)
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
fig.update_coloraxes(showscale=False)
fig.for_each_annotation(lambda a: a.update(text=str(int(a.text.split("=")[1]))))
fig.update_layout(template="plotly_white", width=1100, height=160, margin=dict(l=0,r=0,b=0,t=30))
fig.write_html("exercises/out_0_eigendecomposition.html")
print("  saved: out_0_eigendecomposition.html")

fig2 = plot_eigenspectrum(model, digit=5)
fig2.write_html("exercises/out_0_eigenspectrum_digit5.html")
print("  saved: out_0_eigenspectrum_digit5.html")

# ── Helpers ───────────────────────────────────────────────────────────────────

def train_sparse(model, rank=64, alpha=0.0, beta=0.0, steps=200, label=""):
    sparse = Sparse.from_config(rank=rank).to(device)
    try:
        optimizer = torch.optim.Muon(sparse.parameters(), lr=0.02, momentum=0.95)
    except AttributeError:
        optimizer = torch.optim.AdamW(sparse.parameters(), lr=0.002)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    torch.set_grad_enabled(True)
    for step in tqdm(range(steps), desc=label):
        recon   = 1 - sparse.similarity(model)
        d_loss  = alpha * sparse.down.abs().mean()
        in_loss = beta  * (sparse.left.abs().mean() + sparse.right.abs().mean())
        loss = recon + d_loss + in_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    torch.set_grad_enabled(False)

    with torch.no_grad():
        orig_acc   = (model(test.x).argmax(-1) == test.y).float().mean()
        sparse_acc = (sparse(test.x).argmax(-1) == test.y).float().mean()
        print(f"  [{label}] original: {orig_acc:.3f} | CP approx: {sparse_acc:.3f}")
    return sparse


def plot_neurons(sparse, k=8, title=""):
    plus, minus, down, sigma = sparse.decompose()
    plus, minus, down = plus.cpu(), minus.cpu(), down.cpu()
    fig = make_subplots(rows=3, cols=k,
        row_titles=["L+R (excites)", "L-R (inhibits)", "logit weights"],
        vertical_spacing=0.06)
    for i in range(k):
        hm = dict(showscale=False, colorscale="RdBu", zmid=0)
        fig.add_heatmap(z=plus[:, i].view(28, 28).flip(0), **hm, row=1, col=i+1)
        fig.add_heatmap(z=minus[:, i].view(28, 28).flip(0), **hm, row=2, col=i+1)
        fig.add_bar(x=list(range(10)), y=down[:, i],
                    marker_color=["gray"]*10, showlegend=False, row=3, col=i+1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=list(range(10)), row=3)
    fig.update_layout(width=k*120, height=430, margin=dict(l=70,r=0,b=0,t=30),
                      template="plotly_white", title_text=title)
    return fig


def plot_down_matrix(sparse, title=""):
    _, _, down, _ = sparse.decompose()
    down = down.cpu()
    fig = px.imshow(down, color_continuous_midpoint=0, color_continuous_scale="RdBu",
        labels=dict(x="component (sorted by σ)", y="digit class"),
        title=title, template="plotly_white")
    fig.update_layout(width=700, height=300, margin=dict(l=0,r=0,b=40,t=40))
    return fig


def down_entropy(sparse):
    _, _, down, _ = sparse.decompose()
    p = down.abs() / down.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
    return (-(p * (p + 1e-8).log()).sum(dim=0)).mean().item()


def find_shared_neurons(sparse, min_classes=3, threshold=0.15, k=6):
    _, _, down, sigma = sparse.decompose()
    down = down.cpu()
    normed = down.abs() / down.abs().max(dim=0, keepdim=True).values.clamp(min=1e-8)
    active = (normed > threshold).sum(dim=0)
    shared_sigma = sigma.cpu() * (active >= min_classes).float()
    return shared_sigma.argsort(descending=True)[:k]


def plot_shared_neurons(sparse, title=""):
    idxs = find_shared_neurons(sparse)
    plus, minus, down, sigma = sparse.decompose()
    plus, minus, down = plus.cpu(), minus.cpu(), down.cpu()
    k = len(idxs)
    fig = make_subplots(rows=3, cols=k,
        row_titles=["L+R", "L-R", "logit weights"], vertical_spacing=0.06)
    for col, idx in enumerate(idxs):
        hm = dict(showscale=False, colorscale="RdBu", zmid=0)
        fig.add_heatmap(z=plus[:, idx].view(28, 28).flip(0), **hm, row=1, col=col+1)
        fig.add_heatmap(z=minus[:, idx].view(28, 28).flip(0), **hm, row=2, col=col+1)
        d = down[:, idx]
        thresh = 0.15 * d.abs().max()
        colors = ["steelblue" if abs(v) > thresh else "lightgray" for v in d]
        fig.add_bar(x=list(range(10)), y=d, marker_color=colors,
                    showlegend=False, row=3, col=col+1)
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=list(range(10)), row=3)
    fig.update_layout(width=k*130, height=430, margin=dict(l=70,r=0,b=0,t=40),
                      template="plotly_white", title_text=title)
    return fig


# ── Experiment 1: Plain CP ────────────────────────────────────────────────────
print("\n=== Experiment 1: Plain CP (alpha=0, beta=0) ===")
cp_plain = train_sparse(model, rank=64, alpha=0.0, beta=0.0, steps=200, label="Exp1 plain")

plot_neurons(cp_plain, k=8, title="Exp 1: Plain CP — top 8 components").write_html("exercises/out_1_plain_neurons.html")
plot_down_matrix(cp_plain, title="Exp 1: Down matrix (plain CP)").write_html("exercises/out_1_plain_down.html")
plot_shared_neurons(cp_plain, title="Exp 1: Most shared neurons (≥3 classes)").write_html("exercises/out_1_plain_shared.html")
print(f"  down entropy: {down_entropy(cp_plain):.4f}")
print("  saved: out_1_plain_*.html")

# ── Experiment 2: CP + down sparsity ─────────────────────────────────────────
print("\n=== Experiment 2: CP + down sparsity ===")
cp_d_low  = train_sparse(model, rank=64, alpha=0.02, beta=0.0, steps=200, label="Exp2 alpha=0.02")
cp_d_med  = train_sparse(model, rank=64, alpha=0.05, beta=0.0, steps=200, label="Exp2 alpha=0.05")
cp_d_high = train_sparse(model, rank=64, alpha=0.15, beta=0.0, steps=200, label="Exp2 alpha=0.15")

for cp, a, tag in [(cp_d_low,"0.02","low"),(cp_d_med,"0.05","med"),(cp_d_high,"0.15","high")]:
    plot_neurons(cp, k=8, title=f"Exp 2: down L1 alpha={a}").write_html(f"exercises/out_2_{tag}_neurons.html")
    plot_down_matrix(cp, title=f"Exp 2: down matrix alpha={a}").write_html(f"exercises/out_2_{tag}_down.html")
    print(f"  alpha={a} down entropy: {down_entropy(cp):.4f}")
print("  saved: out_2_*.html")

# ── Experiment 3: CP + down + input sparsity ──────────────────────────────────
print("\n=== Experiment 3: CP + down + input sparsity ===")
cp_b_low  = train_sparse(model, rank=64, alpha=0.05, beta=0.005, steps=200, label="Exp3 beta=0.005")
cp_b_med  = train_sparse(model, rank=64, alpha=0.05, beta=0.02,  steps=200, label="Exp3 beta=0.02")
cp_b_high = train_sparse(model, rank=64, alpha=0.05, beta=0.05,  steps=200, label="Exp3 beta=0.05")

for cp, b, tag in [(cp_b_low,"0.005","low"),(cp_b_med,"0.02","med"),(cp_b_high,"0.05","high")]:
    plot_neurons(cp, k=8, title=f"Exp 3: down(0.05)+input beta={b}").write_html(f"exercises/out_3_{tag}_neurons.html")
    plot_down_matrix(cp, title=f"Exp 3: down matrix beta={b}").write_html(f"exercises/out_3_{tag}_down.html")
    print(f"  beta={b} down entropy: {down_entropy(cp):.4f}")
print("  saved: out_3_*.html")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
rows = [
    ("Exp 1: plain CP",         0.00, 0.000, down_entropy(cp_plain)),
    ("Exp 2a: down alpha=0.02",     0.02, 0.000, down_entropy(cp_d_low)),
    ("Exp 2b: down alpha=0.05",     0.05, 0.000, down_entropy(cp_d_med)),
    ("Exp 2c: down alpha=0.15",     0.15, 0.000, down_entropy(cp_d_high)),
    ("Exp 3a: both beta=0.005",    0.05, 0.005, down_entropy(cp_b_low)),
    ("Exp 3b: both beta=0.02",     0.05, 0.020, down_entropy(cp_b_med)),
    ("Exp 3c: both beta=0.05",     0.05, 0.050, down_entropy(cp_b_high)),
]
df = pd.DataFrame(rows, columns=["Experiment", "alpha (down)", "beta (input)", "Down entropy"])
print(df.to_string(index=False))
df.to_csv("exercises/out_summary.csv", index=False)

fig = px.line(df, x="Experiment", y="Down entropy", markers=True,
              title="Down entropy vs regularization (higher = more shared across classes)",
              template="plotly_white")
fig.update_layout(width=800, height=400)
fig.write_html("exercises/out_summary_entropy.html")
print("\nAll done. Open exercises/out_*.html in a browser.")

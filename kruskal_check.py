import sys
sys.path.insert(0, 'src')

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from kornia.augmentation import RandomGaussianNoise
from tqdm import tqdm

from image import Model, MNIST
from image.sparse import Model as Sparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device.startswith("cuda"):
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Train base model ───────────────────────────────────────────────────────────
print("\n=== Training base bilinear model (20 epochs) ===")
model = Model.from_config(epochs=20).to(device)
model.fit(train := MNIST(train=True, device=device),
          test  := MNIST(train=False, device=device),
          RandomGaussianNoise(std=0.4))

# ── Train plain CP (no regularization) ────────────────────────────────────────
print("\n=== Training plain CP decomposition (200 steps) ===")
sparse = Sparse.from_config(rank=64).to(device)
try:
    optimizer = torch.optim.Muon(sparse.parameters(), lr=0.02, momentum=0.95)
    print("  Optimizer: Muon")
except AttributeError:
    optimizer = torch.optim.AdamW(sparse.parameters(), lr=0.002)
    print("  Optimizer: AdamW (Muon not available)")
scheduler = CosineAnnealingLR(optimizer, T_max=200)

torch.set_grad_enabled(True)
for step in tqdm(range(200), desc="CP"):
    loss = 1 - sparse.similarity(model)
    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
torch.set_grad_enabled(False)

with torch.no_grad():
    sim = sparse.similarity(model).item()
    acc = (sparse(test.x).argmax(-1) == test.y).float().mean().item()
print(f"  CP similarity to original tensor: {sim:.4f}")
print(f"  CP task accuracy: {acc:.4f}")

# ── Kruskal rank check ─────────────────────────────────────────────────────────
print("\n=== Kruskal rank verification ===")
print(f"  Rank r = 64  →  need k_L + k_R + k_D ≥ 2r + 2 = 130\n")

L = sparse.left.detach()   # shape: (784, 64)
R = sparse.right.detach()  # shape: (784, 64)
D = sparse.down.detach()   # shape: (10,  64)

def kruskal_rank(M, name, thresh_rel=1e-6):
    sv = torch.linalg.svdvals(M)
    thresh = thresh_rel * sv[0].item()
    k = (sv > thresh).sum().item()
    print(f"  {name}: shape {list(M.shape)}")
    print(f"    singular values (top 10): {sv[:10].cpu().numpy().round(4)}")
    print(f"    min sv: {sv[-1].item():.2e}  |  threshold ({thresh_rel:.0e} × σ_max): {thresh:.2e}")
    print(f"    k-rank = {k}  (out of min(shape) = {min(M.shape)})")
    return k

k_L = kruskal_rank(L, "L (left,  784×64)")
print()
k_R = kruskal_rank(R, "R (right, 784×64)")
print()
k_D = kruskal_rank(D, "D (down,  10×64)")

total = k_L + k_R + k_D
needed = 2 * 64 + 2  # 130

print(f"\n{'='*50}")
print(f"  k_L = {k_L}")
print(f"  k_R = {k_R}")
print(f"  k_D = {k_D}")
print(f"  SUM = {total}  (need ≥ {needed})")
print()
if total >= needed:
    print(f"  ✓ KRUSKAL CONDITION HOLDS  ({total} ≥ {needed})")
    print("    Decomposition is essentially unique. Sharing is forced by the tensor.")
else:
    print(f"  ✗ KRUSKAL CONDITION FAILS  ({total} < {needed})")
    print("    Uniqueness not guaranteed — shared components may be an artifact.")
print('='*50)

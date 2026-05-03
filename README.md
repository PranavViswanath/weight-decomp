# Decomposing Weights into Human Concepts

**Dictionary learning on bilinear MLP weights yields a human-programmable visual vocabulary**

*Pranav Viswanath — MARS V Application, 2026*

📄 **[Read the report (PDF)](report.pdf)**

---

## What this is

Standard MLPs apply a nonlinearity after each linear layer, making the weights uninterpretable in isolation. This project uses a **bilinear MLP** (Dooms & Gauderis, ICLR 2025), where the output logit for each class is an exact quadratic form in the input — fully determined by a symmetric 784×784 weight matrix, no activations needed.

We decompose that weight tensor directly using tensor methods and find that **dictionary learning recovers a human-readable visual vocabulary**: upper-curve, open-curve, closed-loop, hook-end, diagonal, double-loop. These six atoms are interpretable enough that you can write the full 10-class classifier by hand and get **90.1% test accuracy** (vs. the model's trained 96.8%).

The weights are not just readable — they're writable.

---

## Key finding

Given only the six atom visualizations (no training data, no test images), I hand-specified a code vector for each digit using visual reasoning:

- **Digit 0:** closed-loop (+1.0)
- **Digit 1:** suppress all curve atoms
- **Digit 6:** closed-loop (+0.6) + upper-curve (+0.2) − hook-end (−0.8)
- **Digit 8:** double-loop (+1.0)

Running this hand-written classifier on 10,000 test images: **90.1% accuracy**.

---

## Structure

```
weight-decomp/
├── src/
│   ├── model.py            # Bilinear MLP architecture (Dooms & Gauderis)
│   ├── datasets.py         # MNIST loading
│   ├── cp_sparse.py        # CP decomposition model
│   └── metrics.py          # D-entropy, spatial coherence
│
├── cp_decomposition.py     # Run CP rank-64 decomposition
├── kruskal_check.py        # Verify algebraic uniqueness (Kruskal's theorem)
├── hosvd_comparison.py     # HOSVD vs CP comparison
├── prior_sweep.py          # Sweep L1 / TV / Hoyer / non-neg priors
├── dictionary_learning.py  # Main DL experiment: K=6,8,10 atoms
├── atom_activations.py     # Validate weight-space codes vs forward-pass
├── handwrite_classifier.py # THE FINDING: hand-written codes → 90.1% accuracy
│
├── figures/
│   ├── visualize_cp.py         # CP component visualizations
│   └── digit6_decomposition.py # Digit 6 atom decomposition figure
│
└── report.pdf              # Full write-up
```

Run scripts in order: `cp_decomposition` → `kruskal_check` → `prior_sweep` → `dictionary_learning` → `handwrite_classifier`.

---

## Dependencies

```
torch >= 2.0
numpy
matplotlib
scikit-learn      # DictionaryLearning
scipy
einops
kornia
```

The bilinear MLP architecture and training code are from [tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition) (Dooms & Gauderis, arXiv:2410.08417, ICLR 2025). The decomposition experiments, dictionary learning analysis, and hand-written classifier are original work.

---

## Citation

```bibtex
@article{dooms2024bilinear,
  title   = {Bilinear {MLP}s enable weight-based mechanistic interpretability},
  author  = {Dooms, Thomas and Gauderis, Ward},
  journal = {arXiv preprint arXiv:2410.08417},
  year    = {2024},
  note    = {Accepted at ICLR 2025}
}
```

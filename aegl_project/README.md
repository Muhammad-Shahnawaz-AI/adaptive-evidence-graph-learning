# AEGL — Adaptive Evidence Graph Learning

A working reference implementation of the framework described in the proposal
*"Adaptive Evidence Graph Learning (AEGL): A Variational Framework for
Open-World Visual Recognition via End-to-End Differentiable Memory
Topologies."*

This turns the proposal's math (Chapter 4) and pseudocode (Algorithm 1) into
runnable, trainable PyTorch code, and reproduces the four ablation protocols
from Section 5.1 so they can actually be run and compared.

## Proposal → Code Map

| Proposal element | File | Notes |
|---|---|---|
| Vision Encoder φ, `Zx ← VisionEncoder_φ(X)` | `aegl/encoder.py` | Compact CNN by default; swap in ViT/CLIP/MAE for scale. |
| Memory repository `M` | `aegl/memory.py` | Non-parametric buffer with EMA refresh, not backprop-trained directly. |
| Router `q(G\|x,M)`, Eq. 4.3 | `aegl/router.py` | `ω_ij = softmax(z_x·M_j/√d)`, top-k retrieval, Gumbel-Softmax relaxation `A_ij`. |
| Graph Transformer θ | `aegl/graph_transformer.py` | Multi-head self-attention over `{query} ∪ V_retrieved`, with `A` injected as a topology bias. |
| ELBO objective, Eq. 4.2 | `aegl/losses.py` | `L_CE + β·D_KL(q‖p)` with a uniform prior `p(G)` over retrieved edges. |
| Epistemic bound `Ue`, Alg. 1 lines 9–10 | `aegl/uncertainty.py` | Predictive entropy + variance of updated node states. |
| Causal counterfactual verification | `aegl/causal_verification.py` | Randomly perturbs `A`, measures KL between original and counterfactual predictions. |
| Algorithm 1 (full loop) | `aegl/model.py` (`AEGLModel.forward`) + `train.py` | End-to-end forward/backward/Adam loop. |
| Table 5.1 metrics | `aegl/metrics.py` | Top-1/5 accuracy, ECE, macro-F1, MAE. |
| Section 5.1 Ablation Matrix | `aegl/config.py` (`AEGLConfig.ablation(...)`) | 4 presets, see below. |

## Ablation Matrix Protocols (Section 5.1)

Run any of these directly:

```bash
python train.py --ablation baseline          # 1. backbone only, no router/graph
python train.py --ablation static_topology   # 2. frozen k-NN, hard adjacency
python train.py --ablation no_elbo           # 3. ELBO KL term disabled
python train.py --ablation no_causal         # 4. causal verification disabled
python train.py --ablation full              # complete AEGL
```

Or run all five back-to-back with a comparison table:

```bash
python evaluate.py --epochs 5
```

## Quickstart

```bash
pip install -r requirements.txt
python train.py --ablation full --epochs 5
```

This prints per-epoch CE / KL / causal-divergence, then reports Top-1/5
accuracy, ECE, and macro-F1 on both an in-distribution test split and an
out-of-distribution (shifted) split.

## Important limitation: data

Table 5.1 targets ImageNet-A/R, iNaturalist 2024, MIMIC-IV/CMU-MOSEI, and
PEMS — all multi-gigabyte datasets hosted outside this environment's
network allowlist (only package registries like PyPI are reachable here).

To keep the pipeline fully runnable, `aegl/data.py` ships a lightweight
**synthetic open-world proxy**: fixed per-class spatial templates for
in-distribution data, plus a rotated/rescaled/noise-heavy variant for the
"OOD test" split (standing in for the ImageNet-A/R style shift). It exists
solely so every module — encoder, memory, router, graph transformer, ELBO,
uncertainty, causal verification — can be exercised end-to-end without
external downloads.

**To run on a real benchmark**, replace `build_dataloaders` in
`aegl/data.py` with a loader (e.g. `torchvision.datasets.ImageFolder`, or a
custom iNaturalist/MIMIC-IV reader) that yields `(image_tensor, label)`
batches — no other file needs to change, since the rest of the framework
only depends on that contract.

## Scaling up

For results suitable for the target venues (TPAMI / IJCV), the following
are the natural next steps, matching the 36-month roadmap in Section 6.2:

- Swap `VisionEncoder` for a pretrained ViT/CLIP/MAE backbone (`aegl/encoder.py`).
- Increase `memory_size` and `top_k` in `aegl/config.py` substantially, and
  consider approximate nearest-neighbor search (e.g. FAISS) before the
  differentiable top-k step, since a full `z_x @ M.T` is O(N_mem) per query.
- Replace the synthetic dataset with the real benchmark suite from Table 5.1.
- Extend `CausalVerifier` with a learned counterfactual generator instead of
  random edge dropout, for closer alignment with Research Objective 4.

## Repository layout

```
aegl/
  config.py               # AEGLConfig + ablation presets
  encoder.py               # Vision Encoder φ
  memory.py                # Non-parametric memory M
  router.py                # Differentiable router q(G|x,M), Eq. 4.3
  graph_transformer.py     # Graph Transformer θ
  uncertainty.py            # Epistemic bound Ue
  causal_verification.py   # Counterfactual verification module
  losses.py                 # ELBO loss, Eq. 4.2
  metrics.py                 # Top-1/5, ECE, macro-F1, MAE
  model.py                   # AEGLModel (Algorithm 1)
  data.py                     # Dataset utilities (synthetic proxy)
train.py                     # Single-config training/eval entry point
evaluate.py                  # Full ablation matrix comparison
requirements.txt
README.md
```

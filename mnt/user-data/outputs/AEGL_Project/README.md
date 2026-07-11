# AEGL — Adaptive Evidence Graph Learning

A completed, runnable implementation of the architecture described in
`Project Concept Note | AEGL`, plus a full-stack console (FastAPI backend +
dashboard frontend) so you can actually watch the system train and inspect
its evidence graphs, rather than just read code in a notebook.

```
aegl/
├── backend/
│   ├── aegl_model.py   # the completed model: encoder, router, graph transformer,
│   │                   # ELBO loss, uncertainty engine, causal verifier, ECE
│   ├── dataset.py      # real long-tail image dataset + CSV upload support
│   └── main.py         # FastAPI server (training jobs, polling, results)
├── frontend/
│   └── index.html      # single-file dashboard (served by the backend)
├── requirements.txt
└── README.md
```

## Run it

```bash
cd aegl
pip install -r requirements.txt
cd backend
python main.py
```

Open **http://localhost:8000** — that's the whole app; the backend also serves
the frontend as static files, so there's nothing else to stand up.

## What was in the original notebook, and what was actually missing

The uploaded notebook (`Adaptive_Evidence_Graph_Learning.ipynb`) had the right
ideas sketched across ten cells, but it wasn't a working system:

1. **It didn't run.** Cell 0 started with a stray `+import torch`, an invalid
   Python token that raises `SyntaxError` before anything else executes.
2. **Three incompatible copies of the same classes.** `AEGLSystem`,
   `DynamicGraphTransformerLayer`, `AEGLLossEngine`, and `AEGLCausalVerifier`
   were each redefined 2–3 times across different cells, with different bugs
   fixed in some copies but not others (the last cell even leaves `# FIX:`
   comments pointing at bugs in earlier cells that were never applied
   upstream).
3. **The masking bug.** Earlier versions multiplied the adjacency matrix into
   attention *after* softmax, which doesn't zero out unrouted edges — it just
   down-weights them, so the model still attends over the entire memory bank
   regardless of what the router selected. This silently defeats the whole
   point of "differentiable topology routing." Fixed by masking scores to
   `-inf` *before* softmax (`aegl_model.py`, `DynamicGraphTransformerLayer`).
4. **No aleatoric/epistemic decomposition**, despite the proposal explicitly
   promising to "distinguish between epistemic and aleatoric uncertainty."
   The notebook only ever computed one ad-hoc number (entropy + variance of
   graph entropy) and called it "epistemic." There was no aleatoric term
   anywhere. Replaced with a proper MC-Dropout / BALD decomposition
   (`UncertaintyEngine` in `aegl_model.py`):
   `total = H[E_t[p_t]]`, `aleatoric = E_t[H[p_t]]`, `epistemic = total − aleatoric`.
5. **No calibration metric**, despite "reducing expected calibration error"
   being named as a target benchmark. Implemented `expected_calibration_error()`
   (standard binned ECE) plus a reliability diagram in the dashboard.
6. **No real data.** Every training/eval loop ran on `torch.randn(...)` dummy
   tensors — there was never an actual image, so nothing about the "heavy-tail
   benchmark" claims could be observed. Added `LongTailDigitsDataset`: real
   8×8 handwritten-digit images (scikit-learn's `digits`, no network download
   needed) resampled into a genuine long-tail class distribution, plus a CSV
   upload path for bringing your own tabular/pixel data.
7. **No way to see any of it working.** Everything printed to stdout in a
   notebook cell and vanished. Added the FastAPI backend + dashboard so
   training curves, calibration, causal attribution, and per-sample evidence
   graphs render live and are inspectable.

## How the pieces map to the proposal

| Proposal concept | Where it lives now |
|---|---|
| Vision Encoder → `z_x` | `VisionEncoder` (small CNN) in `aegl_model.py` |
| Differentiable Router `q(G\|x,M)` | `AEGLSystem.differentiable_router` — Gumbel-Softmax relaxation during training, sparse top-k masked routing at inference |
| Dynamic Graph Transformer | `DynamicGraphTransformerLayer` — pre-softmax structural masking, multi-head, residual + FFN |
| Evidence Lower Bound (ELBO) | `AEGLLossEngine`: `L = CE + β·KL(q(G\|x,M) ‖ p(G))` |
| Epistemic vs. aleatoric uncertainty | `UncertaintyEngine` (MC-Dropout / BALD decomposition) |
| Causal Counterfactual Verification | `AEGLCausalVerifier` — edge-masking stress test, reports KL deflection as a causal attribution score |
| Open-world / long-tail benchmarking | `LongTailDigitsDataset` + Expected Calibration Error + reliability diagram |

## Dashboard walkthrough

- **Left panel** — pick the built-in long-tail digit benchmark (adjustable
  imbalance ratio) or upload your own CSV (label column + flattened pixel/feature
  columns), then tune the router/graph hyperparameters (feature dimension,
  memory repository size, top-k routed edges, Gumbel-Softmax τ, KL weight β,
  epochs, batch size, learning rate).
- **Run the pipeline** — kicks off a real background training job; the two
  live charts stream the ELBO components (total / cross-entropy / KL) and
  training accuracy per epoch as they happen.
- **Results** — test accuracy, Expected Calibration Error, and the mean
  epistemic/aleatoric uncertainty across a held-out sample; a reliability
  diagram; the actual long-tail class distribution the run trained on.
- **Sample gallery + Evidence Graph Inspector** — click any test image to see
  its real prediction, confidence, uncertainty split, and — this is the
  proposal's "auditable structural reasoning" made concrete — a bar chart of
  exactly which memory-repository nodes the router routed that sample through,
  i.e. the literal subgraph `G` the Graph Transformer reasoned over.

## Honest caveats

- The vision encoder is a small CNN trained from scratch on 8×8 images for
  a fast, dependency-light demo — not a frozen CLIP/DINOv2 backbone. Swap
  `VisionEncoder` for a real backbone's frozen features to scale this up.
  The router / graph transformer / ELBO / uncertainty / causal-verification
  machinery around it is unchanged either way.
- "Causal" here means the proposal's own definition: how much a prediction
  deflects when you remove the graph's strongest edges. It is a counterfactual
  sensitivity/attribution measure, not causal inference in the Pearl sense —
  worth knowing before quoting the number as a formal causal effect.

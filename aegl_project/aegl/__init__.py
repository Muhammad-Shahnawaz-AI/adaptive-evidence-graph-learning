"""
AEGL: Adaptive Evidence Graph Learning
======================================
Reference implementation of the framework described in the doctoral
research proposal "Adaptive Evidence Graph Learning (AEGL): A Variational
Framework for Open-World Visual Recognition via End-to-End Differentiable
Memory Topologies".

Modules
-------
encoder.py              -> Vision Encoder (phi), produces z_x
memory.py                -> Non-parametric Memory repository M
router.py                -> Differentiable variational router q(G|x,M)
                             (Gumbel-Softmax relaxation, Eq. 4.3)
graph_transformer.py     -> Dynamic Graph Transformer (theta)
uncertainty.py            -> Epistemic / aleatoric uncertainty (Ue)
causal_verification.py   -> Counterfactual causal verification module
losses.py                -> ELBO objective (Eq. 4.2)
metrics.py                -> Top-1/5 accuracy, ECE, macro-F1, MAE
model.py                  -> Full AEGL model (Algorithm 1) + ablation configs
data.py                   -> Dataset utilities (synthetic open-world proxy)
"""

from .config import AEGLConfig
from .model import AEGLModel

__all__ = ["AEGLConfig", "AEGLModel"]

"""
ELBO training objective -- Section 4.2, Eq. (4.2) and Algorithm 1, line 11:
    L_ELBO = L_CE( p(y|x,G), y_true ) + beta * D_KL( q(G|x,M) || p(G) )

Ablation ("Uncertainty De-calibration", Section 5.1 item 3): setting
`use_elbo=False` (equivalently beta=0 at the call site) removes the KL
regularizer entirely, leaving a plain cross-entropy objective with no
variational calibration of the router's edge distribution.
"""
import torch
import torch.nn.functional as F


def elbo_loss(logits: torch.Tensor, targets: torch.Tensor, kl: torch.Tensor, beta: float, use_elbo: bool):
    ce = F.cross_entropy(logits, targets)
    if not use_elbo or kl is None:
        return ce, ce.detach(), torch.zeros_like(ce)
    kl_term = beta * kl.mean()
    return ce + kl_term, ce.detach(), kl_term.detach()

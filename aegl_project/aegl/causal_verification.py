"""
Adversarial causal verification -- Research Objective 4 / Section 5.1 item 4.

Stress-tests the derived subgraph by generating a counterfactual version
of the adjacency (randomly dropping a fraction of retrieved edges) and
re-running the same classification head on the resulting graph
representation. A large prediction shift under an otherwise-benign
perturbation indicates the causal structure is *not* robust / auditable;
we summarize this as a scalar "causal attribution divergence" that can be
logged for auditing or added as an auxiliary consistency penalty.

Ablation ("Causal Verification Removal", Section 5.1 item 4): the model
can simply skip calling this module (`use_causal=False`), which removes
both the auditability signal and any associated training penalty.
"""
import torch
import torch.nn.functional as F


class CausalVerifier:
    def __init__(self, perturb_ratio: float = 0.3):
        self.perturb_ratio = perturb_ratio

    @torch.no_grad()
    def counterfactual_adjacency(self, A: torch.Tensor) -> torch.Tensor:
        """Randomly zero out a fraction of edges and renormalize -> A_cf."""
        mask = (torch.rand_like(A) > self.perturb_ratio).float()
        A_cf = A * mask
        A_cf = A_cf / A_cf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return A_cf

    def verify(self, graph_transformer, classifier, z_x, retrieved_nodes, A):
        """
        Returns:
            probs_cf: (B, C) counterfactual predictive distribution
            divergence: (B,) KL(p(y|x,G) || p(y|x,G_cf)) causal attribution
                        divergence -- lower is more robust/auditable.
        """
        A_cf = self.counterfactual_adjacency(A)
        h_cf, _ = graph_transformer(z_x, retrieved_nodes, A_cf)
        logits_cf = classifier(h_cf)
        probs_cf = F.softmax(logits_cf, dim=-1)
        return probs_cf, A_cf

    @staticmethod
    def attribution_divergence(probs: torch.Tensor, probs_cf: torch.Tensor) -> torch.Tensor:
        p = probs.clamp_min(1e-12)
        q = probs_cf.clamp_min(1e-12)
        return (p * (p.log() - q.log())).sum(dim=-1)

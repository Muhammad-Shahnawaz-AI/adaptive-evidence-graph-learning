"""
Differentiable Router q(G|x, M)  -- Section 4.2, Eq. (4.3), Algorithm 1 lines 3-6.

Steps implemented:
  1. Compatibility weights:      omega_ij = softmax_j( z_x^T M_j / sqrt(d) )
  2. Top-k candidate selection:  restrict to the k highest-compatibility
                                 memory nodes (this indexing step is the
                                 one non-differentiable operation, exactly
                                 as flagged in Section 3.1 / 6.1 -- gradients
                                 still flow through the *weights* on the
                                 selected edges).
  3. Gumbel-Softmax relaxation:  A_ij = softmax( (log omega_ij + g_ij) / tau )
                                 over the retrieved candidates, giving a
                                 continuous, differentiable soft adjacency.

Ablation ("Static vs. Dynamic Topology", Section 5.1, item 2): when
`static_topology=True` the router instead performs a frozen top-k k-NN
lookup and returns a hard, uniform 0/1 adjacency with no learned routing
signal -- isolating the contribution of *dynamic, differentiable* edge
learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableRouter(nn.Module):
    def __init__(self, embed_dim: int, top_k: int, gumbel_tau: float, static_topology: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.tau = gumbel_tau
        self.static_topology = static_topology

    def forward(self, z_x: torch.Tensor, memory_keys: torch.Tensor):
        """
        Args:
            z_x: (B, d) query encoder features
            memory_keys: (N_mem, d) memory node features M_j
        Returns:
            retrieved_idx: (B, k) indices of retrieved memory nodes (V_retrieved)
            A: (B, k) soft (or hard, if static) adjacency / edge weights
            kl: (B,) per-sample KL(q(G|x,M) || p(G)) against a uniform prior p(G)
        """
        d = z_x.size(-1)
        # omega_ij = softmax_j( z_x^T M_j / sqrt(d) )      -- Algorithm 1, line 3
        logits = (z_x @ memory_keys.t()) / (d ** 0.5)          # (B, N_mem)
        omega = F.softmax(logits, dim=-1)

        # Restrict to top-k candidates (the retrieval step)
        topk_omega, topk_idx = torch.topk(omega, self.top_k, dim=-1)   # (B, k)

        if self.static_topology:
            # Ablation 2: frozen k-NN flat lookup, hard uniform adjacency,
            # no gradient path through edge weights.
            A = torch.full_like(topk_omega, 1.0 / self.top_k)
            A = A.detach()
        else:
            # Gumbel-Softmax relaxation over the retrieved candidates -- Eq. 4.3
            log_w = torch.log(topk_omega.clamp_min(1e-12))
            u = torch.rand_like(log_w).clamp_min(1e-12)
            gumbel_noise = -torch.log((-torch.log(u)).clamp_min(1e-12))
            A = F.softmax((log_w + gumbel_noise) / self.tau, dim=-1)

        # KL(q(G|x,M) || p(G)) with an isotropic (uniform) categorical prior p(G)
        # over the k retrieved edges -- Eq. 4.2 second term.
        q = topk_omega / topk_omega.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p = torch.full_like(q, 1.0 / self.top_k)
        kl = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(p))).sum(dim=-1)

        return topk_idx, A, kl

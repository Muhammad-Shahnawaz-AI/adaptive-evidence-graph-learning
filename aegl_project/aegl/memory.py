"""
Non-parametric memory repository M (Problem Statement, Section 3.1).

M holds `memory_size` structural feature vectors plus verified downstream
semantic labels. It is intentionally kept as a persistent buffer (not a
plain nn.Parameter) to reflect the proposal's framing of M as a
"non-parametric memory repository" that is *queried* rather than directly
back-propagated into like ordinary weights -- gradients instead flow
through the router's compatibility weights omega_ij.

An EMA (exponential moving average) update rule is provided so the memory
can be slowly refreshed with encoder features seen during training,
without turning M into a fully parametric embedding table.
"""
import torch
import torch.nn as nn


class MemoryRepository(nn.Module):
    def __init__(self, memory_size: int, embed_dim: int, num_classes: int, ema_momentum: float = 0.99):
        super().__init__()
        self.memory_size = memory_size
        self.embed_dim = embed_dim
        self.ema_momentum = ema_momentum

        # Structural feature vectors M_j
        self.register_buffer("keys", torch.randn(memory_size, embed_dim) * 0.02)
        # Verified downstream semantic definitions (soft label distribution per node)
        self.register_buffer("labels", torch.randint(0, num_classes, (memory_size,)))

    @torch.no_grad()
    def update(self, z_x: torch.Tensor, y: torch.Tensor, hit_idx: torch.Tensor):
        """EMA-refresh the retrieved memory nodes toward the batch features
        that matched them, keeping M non-parametric (no gradient)."""
        for b in range(z_x.size(0)):
            idx = hit_idx[b]
            self.keys[idx] = self.ema_momentum * self.keys[idx] + (1 - self.ema_momentum) * z_x[b].detach()
            # Occasionally refresh the stored label toward the most confident observed class
            self.labels[idx[0]] = y[b]

    def as_nodes(self) -> torch.Tensor:
        return self.keys  # (memory_size, embed_dim)

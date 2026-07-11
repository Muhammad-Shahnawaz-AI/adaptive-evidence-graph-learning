"""
Full AEGL model -- implements Algorithm 1 ("Adaptive Evidence Graph
Learning (AEGL) Core Optimization Engine") end to end, with the four
ablation switches from Section 5.1 wired into the config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AEGLConfig
from .encoder import VisionEncoder
from .memory import MemoryRepository
from .router import DifferentiableRouter
from .graph_transformer import GraphTransformer
from .uncertainty import epistemic_bound
from .causal_verification import CausalVerifier


class AEGLModel(nn.Module):
    def __init__(self, cfg: AEGLConfig):
        super().__init__()
        self.cfg = cfg

        # phi: Vision Encoder
        self.encoder = VisionEncoder(cfg.in_channels, cfg.embed_dim)

        # M: non-parametric memory repository
        self.memory = MemoryRepository(cfg.memory_size, cfg.embed_dim, cfg.num_classes)

        # q(G|x,M): differentiable router (only built if the ablation uses it)
        if cfg.use_router and cfg.use_graph:
            self.router = DifferentiableRouter(
                cfg.embed_dim, cfg.top_k, cfg.gumbel_tau, static_topology=cfg.static_topology
            )
            self.graph_transformer = GraphTransformer(cfg.embed_dim, cfg.graph_layers, cfg.graph_heads)
        else:
            self.router = None
            self.graph_transformer = None

        # Classification head: p(y|x,G)
        self.classifier = nn.Linear(cfg.embed_dim, cfg.num_classes)

        # Causal verification module
        self.causal_verifier = CausalVerifier(cfg.causal_perturb_ratio) if cfg.use_causal else None

    def forward(self, x: torch.Tensor, run_causal: bool = True):
        """
        Returns a dict with:
            logits, probs, kl, Ue (epistemic bound),
            causal_divergence (None if disabled or run_causal=False)
        """
        # Algorithm 1, line 1
        z_x = self.encoder(x)

        if self.router is None:
            # Ablation 1: Baseline Validation -- backbone only, no retrieval/graph.
            logits = self.classifier(z_x)
            probs = F.softmax(logits, dim=-1)
            Ue = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
            return {
                "logits": logits, "probs": probs, "kl": None,
                "Ue": Ue, "causal_divergence": None, "retrieved_idx": None,
            }

        # Algorithm 1, lines 3-6: compatibility weights + Gumbel-Softmax routing
        retrieved_idx, A, kl = self.router(z_x, self.memory.as_nodes())
        retrieved_nodes = self.memory.as_nodes()[retrieved_idx]  # (B, k, d)

        # Algorithm 1, line 7: Graph Transformer forward pass
        h_query, all_nodes = self.graph_transformer(z_x, retrieved_nodes, A)

        # Algorithm 1, line 8: classification head
        logits = self.classifier(h_query)
        probs = F.softmax(logits, dim=-1)

        # Algorithm 1, lines 9-10: epistemic bound
        Ue = epistemic_bound(probs, all_nodes)

        causal_divergence = None
        if self.causal_verifier is not None and run_causal:
            probs_cf, _ = self.causal_verifier.verify(
                self.graph_transformer, self.classifier, z_x, retrieved_nodes, A
            )
            causal_divergence = self.causal_verifier.attribution_divergence(probs, probs_cf)

        return {
            "logits": logits, "probs": probs, "kl": kl,
            "Ue": Ue, "causal_divergence": causal_divergence,
            "retrieved_idx": retrieved_idx,
        }

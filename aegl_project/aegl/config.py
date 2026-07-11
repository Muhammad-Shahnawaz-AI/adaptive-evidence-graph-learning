"""
Central configuration for AEGL, including the four ablation switches
defined in Section 5.1 of the proposal ("Ablation Matrix Protocols").
"""
from dataclasses import dataclass


@dataclass
class AEGLConfig:
    # ---- data / task ----
    num_classes: int = 10
    in_channels: int = 3
    image_size: int = 32

    # ---- encoder (phi) ----
    embed_dim: int = 128

    # ---- memory (M) ----
    memory_size: int = 512          # number of non-parametric memory nodes
    top_k: int = 8                  # |V_retrieved| per sample

    # ---- router q(G|x,M), Eq. 4.3 ----
    gumbel_tau: float = 0.5         # relaxation temperature tau

    # ---- graph transformer (theta) ----
    graph_layers: int = 2
    graph_heads: int = 4

    # ---- ELBO, Eq. 4.2 ----
    kl_beta: float = 0.1            # beta weighting of D_KL(q||p)

    # ---- causal verification ----
    causal_perturb_ratio: float = 0.3   # fraction of edges dropped for the counterfactual

    # ---- Ablation Matrix Protocols (Section 5.1) ----
    # 1. Baseline Validation: backbone only, no router / graph module.
    use_router: bool = True
    use_graph: bool = True
    # 2. Static vs. Dynamic Topology: replace the differentiable router
    #    with a frozen top-k k-NN flat lookup (hard 0/1 adjacency, no
    #    gradient through edge weights).
    static_topology: bool = False
    # 3. Uncertainty De-calibration: disable the ELBO KL term.
    use_elbo: bool = True
    # 4. Causal Verification Removal.
    use_causal: bool = True

    @classmethod
    def ablation(cls, name: str, **overrides) -> "AEGLConfig":
        """Convenience constructor for the four protocols in Section 5.1."""
        presets = {
            "full": dict(),
            "baseline": dict(use_router=False, use_graph=False, use_causal=False),
            "static_topology": dict(static_topology=True),
            "no_elbo": dict(use_elbo=False),
            "no_causal": dict(use_causal=False),
        }
        if name not in presets:
            raise ValueError(f"Unknown ablation '{name}'. Choose from {list(presets)}")
        cfg = cls(**presets[name])
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

"""
Epistemic bound via variational entropy -- Algorithm 1, line 9-10:
    Ue <- -sum p(y|x,G) log p(y|x,G) + Variance(H_G)

`Variance(H_G)` is computed as the mean feature-wise variance across the
updated graph node states (query + retrieved nodes), used as a proxy for
how much the representation disagrees across the retrieved topological
neighborhood -- high disagreement under OOD inputs signals epistemic
uncertainty, while the predictive entropy term captures aleatoric-style
ambiguity in the output distribution itself.
"""
import torch


def epistemic_bound(probs: torch.Tensor, all_nodes: torch.Tensor) -> torch.Tensor:
    """
    Args:
        probs: (B, C) predictive class probabilities p(y|x,G)
        all_nodes: (B, 1+k, d) all updated node representations H_G
    Returns:
        Ue: (B,) scalar epistemic/aleatoric uncertainty bound per sample
    """
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    node_variance = all_nodes.var(dim=1, unbiased=False).mean(dim=-1)
    return entropy + node_variance

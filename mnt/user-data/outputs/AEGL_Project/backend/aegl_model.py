"""
Adaptive Evidence Graph Learning (AEGL) — Core Engine
=======================================================
This module is the completed, consolidated implementation of the architecture
described in the AEGL concept note. It reconciles the multiple draft variants
found in the original notebook into a single, correct, runnable system and
fills in the pieces the proposal called for but that were never implemented:

  * A real (small) vision encoder over image tensors (not just flat vectors).
  * A differentiable structural router: soft compatibility scores -> Gumbel-Softmax
    relaxation during training, sparse top-k masked-attention routing at inference.
  * A Dynamic Graph Transformer that performs masked message passing strictly
    over the routed topology (the earlier "multiply-after-softmax" gating leaked
    attention mass to unrouted nodes — fixed here with a pre-softmax mask, matching
    the "FIX" already sketched in the last notebook cell).
  * An explicit Evidence Lower Bound (ELBO) loss: L = CE + beta * KL(q(G|x,M) || p(G)).
  * A proper epistemic/aleatoric uncertainty DECOMPOSITION via MC-Dropout (BALD):
        total (predictive) entropy = H[ E_t[p(y|x,G_t)] ]
        aleatoric                  = E_t[ H[p(y|x,G_t)] ]
        epistemic (mutual info)    = total - aleatoric
    The original notebook only ever computed a single ad-hoc "epistemic" number
    (predictive entropy + graph-entropy variance) and never separated aleatoric
    uncertainty at all, despite the proposal explicitly promising both.
  * Expected Calibration Error (ECE) — the proposal's stated success metric
    ("reducing expected calibration error") was never implemented anywhere.
  * A Causal Counterfactual Verifier that measures how much masking the top
    routed evidence edges deflects the prediction (causal attribution score).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. VISION ENCODER
# ============================================================================
class VisionEncoder(nn.Module):
    """Small convolutional encoder producing a latent feature token z_x.

    Standing in for a frozen foundation backbone (CLIP / DINOv2) as described
    in the proposal — swap this module out for a real frozen backbone without
    touching anything downstream.
    """

    def __init__(self, in_channels: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 2. DYNAMIC GRAPH TRANSFORMER (masked message passing over routed topology)
# ============================================================================
class DynamicGraphTransformerLayer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, adjacency_matrix):
        """
        nodes: [B, N, D]   adjacency_matrix: [B, N, N] (soft, in [0,1], 0 = no edge)
        """
        batch_size, num_nodes, _ = nodes.size()
        residual = nodes

        q = self.q_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Structural mask: suppress (not merely down-weight) attention to unrouted edges,
        # applied BEFORE softmax so probability mass cannot leak to non-routed nodes.
        gated_adjacency = adjacency_matrix.unsqueeze(1)  # [B, 1, N, N]
        structural_mask = (gated_adjacency <= 1e-8)
        attn_scores = attn_scores.masked_fill(structural_mask, float("-1e9"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.feature_dim)

        x = self.norm1(residual + self.out_proj(context))
        ffn_out = self.ffn(x)
        refined_nodes = self.norm2(x + self.dropout(ffn_out))
        return refined_nodes


# ============================================================================
# 3. THE UNIFIED AEGL SYSTEM (encoder -> differentiable router -> graph transformer -> head)
# ============================================================================
class AEGLSystem(nn.Module):
    def __init__(self, in_channels, feature_dim, memory_size, num_classes,
                 tau=1.0, top_k=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.tau = tau
        self.top_k = min(top_k, memory_size)

        self.vision_encoder = VisionEncoder(in_channels, feature_dim, dropout=dropout)
        self.memory_repository = nn.Parameter(torch.randn(memory_size, feature_dim) * 0.5)
        self.graph_transformer = DynamicGraphTransformerLayer(feature_dim, dropout=dropout)
        self.classification_head = nn.Linear(feature_dim, num_classes)

    def differentiable_router(self, z_x):
        """q(G | x, M): soft compatibility -> Gumbel-Softmax relaxation (train) /
        sparse top-k masked routing (eval)."""
        scores = torch.matmul(z_x, self.memory_repository.t()) / (self.feature_dim ** 0.5)
        omega = F.softmax(scores, dim=-1)

        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
            logits = (torch.log(omega + 1e-20) + gumbel_noise) / self.tau
            a_route = F.softmax(logits, dim=-1)
        else:
            topk_vals, topk_idx = torch.topk(omega, self.top_k, dim=-1)
            sparse_mask = torch.zeros_like(omega).scatter_(-1, topk_idx, 1.0)
            a_route = F.softmax(scores.masked_fill(sparse_mask == 0, float("-1e9")), dim=-1)

        return a_route

    def forward(self, x, return_nodes=False):
        batch_size = x.size(0)
        z_x = self.vision_encoder(x)
        a_route = self.differentiable_router(z_x)

        image_token = z_x.unsqueeze(1)
        expanded_memory = self.memory_repository.unsqueeze(0).expand(batch_size, -1, -1)
        combined_nodes = torch.cat([image_token, expanded_memory], dim=1)

        full_adj = torch.zeros(batch_size, 1 + self.memory_size, 1 + self.memory_size, device=x.device)
        full_adj[:, 0, 1:] = a_route
        full_adj[:, 1:, 0] = a_route
        # every node needs a self-loop or fully-masked rows produce NaNs in softmax
        idx = torch.arange(1 + self.memory_size, device=x.device)
        full_adj[:, idx, idx] = 1.0

        refined_nodes = self.graph_transformer(combined_nodes, full_adj)
        final_image_state = refined_nodes[:, 0, :]
        logits = self.classification_head(final_image_state)

        if return_nodes:
            return logits, a_route, refined_nodes
        return logits, a_route


# ============================================================================
# 4. VARIATIONAL ELBO LOSS ENGINE
# ============================================================================
class AEGLLossEngine(nn.Module):
    """L_ELBO = CrossEntropy(y, y_hat) + beta * D_KL( q(G|x,M) || p(G) )"""

    def __init__(self, beta=1.0, epsilon=1e-10):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon

    def compute_kl_divergence(self, q_adjacency, p_prior=None):
        if p_prior is None:
            num_nodes = q_adjacency.size(-1)
            p_prior = torch.full_like(q_adjacency, 1.0 / num_nodes)
        kl_val = q_adjacency * (torch.log(q_adjacency + self.epsilon) - torch.log(p_prior + self.epsilon))
        return torch.mean(torch.sum(kl_val, dim=-1))

    def forward(self, logits, targets, adjacency_matrix, p_prior=None):
        ce_loss = F.cross_entropy(logits, targets)
        kl_loss = self.compute_kl_divergence(adjacency_matrix, p_prior)
        total_loss = ce_loss + self.beta * kl_loss
        return total_loss, ce_loss, kl_loss


# ============================================================================
# 5. UNCERTAINTY ENGINE — proper epistemic / aleatoric decomposition (BALD)
# ============================================================================
class UncertaintyEngine:
    """MC-Dropout based decomposition of predictive uncertainty.

    total (predictive) entropy = H[ mean_t p_t ]
    aleatoric                  = mean_t H[ p_t ]
    epistemic (mutual info)    = total - aleatoric
    """

    def __init__(self, num_samples: int = 12, epsilon: float = 1e-10):
        self.num_samples = num_samples
        self.epsilon = epsilon

    @staticmethod
    def _entropy(probs, eps):
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    @torch.no_grad()
    def estimate(self, model: AEGLSystem, x: torch.Tensor):
        was_training = model.training
        model.train()  # enable dropout stochasticity for MC sampling
        # keep BatchNorm-free architecture so train() only toggles dropout masks

        prob_samples = []
        adj_samples = []
        for _ in range(self.num_samples):
            logits, adj = model(x)
            prob_samples.append(F.softmax(logits, dim=-1))
            adj_samples.append(adj)

        model.train(was_training)

        probs_stack = torch.stack(prob_samples, dim=0)      # [T, B, C]
        mean_probs = probs_stack.mean(dim=0)                # [B, C]

        total_entropy = self._entropy(mean_probs, self.epsilon)                     # [B]
        per_sample_entropy = self._entropy(probs_stack, self.epsilon)               # [T, B]
        aleatoric = per_sample_entropy.mean(dim=0)                                  # [B]
        epistemic = (total_entropy - aleatoric).clamp(min=0.0)                       # [B]

        mean_adjacency = torch.stack(adj_samples, dim=0).mean(dim=0)  # [B, memory_size]

        return {
            "mean_probs": mean_probs,
            "total_uncertainty": total_entropy,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "mean_adjacency": mean_adjacency,
        }


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10):
    """Standard binned ECE — the calibration metric the proposal names as a
    target benchmark but never implements."""
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_stats.append({
                "bin_lower": lo.item(), "bin_upper": hi.item(),
                "accuracy": acc_in_bin.item(), "confidence": conf_in_bin.item(),
                "proportion": prop_in_bin.item(),
            })
        else:
            bin_stats.append({
                "bin_lower": lo.item(), "bin_upper": hi.item(),
                "accuracy": 0.0, "confidence": 0.0, "proportion": 0.0,
            })

    return ece.item(), bin_stats


# ============================================================================
# 6. CAUSAL COUNTERFACTUAL VERIFIER
# ============================================================================
class AEGLCausalVerifier(nn.Module):
    """Stress-tests the routed subgraph: masks out the top evidence edges and
    measures how far predictions deflect (KL between original and counterfactual
    predictive distributions). A high score means the routed topology was
    causally load-bearing for the decision — the "auditable structural
    reasoning" the proposal calls for."""

    def __init__(self, alpha=0.15):
        super().__init__()
        self.alpha = alpha

    def generate_counterfactual_topology(self, adjacency_matrix, perturbation_type="edge_mask"):
        counterfactual_adj = adjacency_matrix.clone()
        if perturbation_type == "edge_mask":
            threshold = torch.quantile(adjacency_matrix, 1.0 - self.alpha)
            mask = adjacency_matrix < threshold
            counterfactual_adj = counterfactual_adj * mask.float()
        elif perturbation_type == "gaussian_noise":
            noise = torch.randn_like(adjacency_matrix) * self.alpha
            counterfactual_adj = torch.clamp(adjacency_matrix + noise, min=0.0, max=1.0)
        return counterfactual_adj

    @torch.no_grad()
    def forward(self, aegl_system: AEGLSystem, inputs, base_logits, base_adjacency):
        was_training = aegl_system.training
        aegl_system.eval()

        cf_adjacency = self.generate_counterfactual_topology(base_adjacency, "edge_mask")

        z_x = aegl_system.vision_encoder(inputs)
        batch_size = inputs.size(0)
        image_token = z_x.unsqueeze(1)
        expanded_memory = aegl_system.memory_repository.unsqueeze(0).expand(batch_size, -1, -1)
        combined_nodes = torch.cat([image_token, expanded_memory], dim=1)

        cf_full_adj = torch.zeros(batch_size, 1 + aegl_system.memory_size,
                                   1 + aegl_system.memory_size, device=inputs.device)
        cf_full_adj[:, 0, 1:] = cf_adjacency
        cf_full_adj[:, 1:, 0] = cf_adjacency
        idx = torch.arange(1 + aegl_system.memory_size, device=inputs.device)
        cf_full_adj[:, idx, idx] = 1.0

        cf_refined_nodes = aegl_system.graph_transformer(combined_nodes, cf_full_adj)
        cf_logits = aegl_system.classification_head(cf_refined_nodes[:, 0, :])

        base_probs = F.softmax(base_logits, dim=-1)
        cf_probs = F.softmax(cf_logits, dim=-1)
        causal_attribution_score = torch.sum(
            base_probs * (torch.log(base_probs + 1e-10) - torch.log(cf_probs + 1e-10)), dim=-1
        )

        aegl_system.train(was_training)
        return causal_attribution_score

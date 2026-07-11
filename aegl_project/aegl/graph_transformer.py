"""
Dynamic Graph Transformer theta -- Section 4.1/4.3, Algorithm 1 line 7:
    H_G <- GraphTransformer_theta(G, z_x)

The subgraph G = {V_retrieved, A} consists of the query node z_x plus the
top-k retrieved memory nodes, connected by the soft adjacency A produced
by the router. Message passing is implemented as multi-head self-attention
over this small node set, with the router's edge weights A injected as an
additive attention bias so the *learned topology* (not just content
similarity) shapes information flow -- matching the proposal's claim that
node characteristics are adjusted "on the fly" based on derived topology.
"""
import torch
import torch.nn as nn


class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, nodes: torch.Tensor, attn_bias: torch.Tensor):
        # nodes: (B, 1+k, d); attn_bias: (B, 1+k, 1+k) additive bias encoding A
        attn_out, _ = self.attn(nodes, nodes, nodes, attn_mask=None, need_weights=False)
        # Inject topology bias as a residual re-weighting of the attended messages
        biased = attn_out + torch.bmm(attn_bias, nodes)
        nodes = self.norm1(nodes + biased)
        nodes = self.norm2(nodes + self.ff(nodes))
        return nodes


class GraphTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, z_x: torch.Tensor, retrieved_nodes: torch.Tensor, A: torch.Tensor):
        """
        Args:
            z_x: (B, d) query node
            retrieved_nodes: (B, k, d) V_retrieved node features
            A: (B, k) soft adjacency edge weights from query -> retrieved nodes
        Returns:
            h_query: (B, d) updated query representation H_G
            all_nodes: (B, 1+k, d) all updated node states (used for epistemic variance)
        """
        B, k, d = retrieved_nodes.shape
        nodes = torch.cat([z_x.unsqueeze(1), retrieved_nodes], dim=1)  # (B, 1+k, d)

        # Build a (1+k, 1+k) attention bias matrix: only the query<->memory
        # edges carry the learned topology weight A; other entries are 0.
        bias = torch.zeros(B, 1 + k, 1 + k, device=z_x.device, dtype=z_x.dtype)
        bias[:, 0, 1:] = A
        bias[:, 1:, 0] = A

        for layer in self.layers:
            nodes = layer(nodes, bias)

        h_query = nodes[:, 0, :]
        return h_query, nodes

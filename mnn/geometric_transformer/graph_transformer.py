"""mnn.geometric_transformer.graph_transformer — Theorem Graph Transformer.

Part 7: Transformers that reason over mathematical knowledge graphs.
Nodes = definitions, lemmas, theorems. Edges = implication, dependency, equivalence.
A mathematical reasoning network, not merely a language model.

Part 8: Hierarchical Geometric Reasoning — reason across scales:
local (equations) → mid (proofs) → large (theorem networks) → global (domains).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class EdgeType(Enum):
    IMPLIES = "implies"
    DEPENDS = "depends"
    EQUIVALENT = "equivalent"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    USES = "uses"


class GraphAttentionLayer(nn.Module):
    """Graph attention over theorem knowledge graphs.

    Nodes carry theorem embeddings; edges carry relation types.
    Attention is edge-type aware.
    """
    def __init__(self, embed_dim: int, n_heads: int = 4,
                 n_edge_types: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

        # Edge-type specific bias
        self.edge_bias = nn.Embedding(n_edge_types, n_heads)
        self.edge_transform = nn.Embedding(n_edge_types, embed_dim)

        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, n_nodes, embed_dim)
        adj: (batch, n_nodes, n_nodes) binary adjacency
        edge_types: (batch, n_nodes, n_nodes) int edge type ids
        """
        B, N, D = x.shape
        residual = x

        Q = self.Wq(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Edge-type bias
        if edge_types is not None:
            edge_b = self.edge_bias(edge_types)  # (B, N, N, n_heads)
            scores = scores + edge_b.permute(0, 3, 1, 2)

        # Mask non-edges (but keep self-loops)
        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        graph_mask = (adj + eye).clamp(max=1.0)
        scores = scores.masked_fill(graph_mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.Wo(out)
        return self.norm(out + residual)


class TheoremGraphTransformer(nn.Module):
    """Full Graph Transformer for mathematical knowledge graphs.

    Stacks graph attention layers with feed-forward networks.
    """
    def __init__(self, embed_dim: int = 64, n_heads: int = 4,
                 n_layers: int = 4, n_edge_types: int = 6,
                 ff_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TheoremGraphBlock(embed_dim, n_heads, n_edge_types, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj, edge_types)
        return self.final_norm(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TheoremGraphBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_edge_types, ff_dim, dropout):
        super().__init__()
        self.attn = GraphAttentionLayer(embed_dim, n_heads, n_edge_types, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, adj, edge_types=None):
        x = self.attn(x, adj, edge_types)
        return self.norm(self.ff(x) + x)


# ---- Part 8: Hierarchical Geometric Reasoning ----

class HierarchicalPooling(nn.Module):
    """Pool nodes into higher-level clusters for multi-scale reasoning."""
    def __init__(self, embed_dim: int, pool_ratio: float = 0.5):
        super().__init__()
        self.score_fn = nn.Linear(embed_dim, 1)
        self.pool_ratio = pool_ratio

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x: (batch, n_nodes, dim)
        Returns: pooled_x, pooled_adj, pool_indices
        """
        B, N, D = x.shape
        k = max(1, int(N * self.pool_ratio))

        scores = self.score_fn(x).squeeze(-1)  # (B, N)
        _, top_idx = torch.topk(scores, k, dim=-1)  # (B, k)

        # Gather top nodes
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)
        pooled_x = torch.gather(x, 1, top_idx_exp) * torch.sigmoid(
            torch.gather(scores, 1, top_idx)).unsqueeze(-1)

        # Reduce adjacency
        idx_i = top_idx.unsqueeze(-1).expand(-1, -1, N)
        pooled_adj_rows = torch.gather(adj, 1, idx_i)
        idx_j = top_idx.unsqueeze(1).expand(-1, k, -1)
        pooled_adj = torch.gather(pooled_adj_rows, 2, idx_j)

        return pooled_x, pooled_adj, top_idx


class HierarchicalGeometricTransformer(nn.Module):
    """Multi-scale geometric reasoning across hierarchical levels.

    Level 0: Equations (local)
    Level 1: Proofs (mid)
    Level 2: Theorem networks (large)
    Level 3: Mathematical domains (global)
    """
    def __init__(self, embed_dim: int = 64, n_heads: int = 4,
                 n_levels: int = 3, pool_ratio: float = 0.5,
                 n_edge_types: int = 6, dropout: float = 0.1):
        super().__init__()
        self.n_levels = n_levels

        # Transformer at each level
        self.level_transformers = nn.ModuleList([
            TheoremGraphTransformer(embed_dim, n_heads, n_layers=2,
                                     n_edge_types=n_edge_types, dropout=dropout)
            for _ in range(n_levels)
        ])

        # Pooling between levels
        self.pooling_layers = nn.ModuleList([
            HierarchicalPooling(embed_dim, pool_ratio)
            for _ in range(n_levels - 1)
        ])

        # Cross-level communication (top-down)
        self.unpool_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim)
            for _ in range(n_levels - 1)
        ])

        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None) -> Dict:
        """Multi-scale forward pass. Returns per-level representations."""
        level_outputs = []
        pool_indices_list = []

        current_x = x
        current_adj = adj

        # Bottom-up: process each level
        for level in range(self.n_levels):
            current_x = self.level_transformers[level](current_x, current_adj, edge_types)
            level_outputs.append(current_x)

            if level < self.n_levels - 1:
                current_x, current_adj, pool_idx = self.pooling_layers[level](
                    current_x, current_adj)
                pool_indices_list.append(pool_idx)
                edge_types = None  # edge types not preserved after pooling

        # Top-down: propagate global context
        for level in range(self.n_levels - 2, -1, -1):
            global_context = self.unpool_projections[level](level_outputs[level + 1])
            # Broadcast global mean to lower level
            global_mean = global_context.mean(dim=1, keepdim=True)
            level_outputs[level] = level_outputs[level] + global_mean

        return {
            "levels": level_outputs,
            "finest": self.final_proj(level_outputs[0]),
            "coarsest": level_outputs[-1],
            "pool_indices": pool_indices_list,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

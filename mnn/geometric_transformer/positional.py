"""mnn.geometric_transformer.positional — Manifold Positional Encodings.

Part 3: Replace linear positional embeddings with:
  - Laplacian eigenvectors (spectral coordinates)
  - Graph-based embeddings (structural position)
  - Random walk encodings (topological role)

Part 4: Category-Theoretic Attention — attention as morphism interaction
and compositional reasoning over mathematical structures.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class SpectralPositionalEncoding(nn.Module):
    """Positional encoding using graph Laplacian eigenvectors.

    L = D - A, use smallest eigenvectors as structural coordinates.
    """
    def __init__(self, embed_dim: int, max_nodes: int = 512, n_eigvecs: int = 16):
        super().__init__()
        self.n_eigvecs = n_eigvecs
        self.proj = nn.Linear(n_eigvecs, embed_dim)
        # Fallback: learnable sinusoidal for sequential data
        self.seq_pe = nn.Parameter(torch.randn(1, max_nodes, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add positional encoding. If adjacency given, use spectral; else sequential."""
        if adjacency is not None:
            pe = self._spectral_encode(adjacency, x.device)
            return x + self.proj(pe)
        return x + self.seq_pe[:, :x.size(1), :]

    def _spectral_encode(self, adj: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute Laplacian eigenvectors from adjacency matrix."""
        B, N, _ = adj.shape
        encodings = []
        for b in range(B):
            A = adj[b].float()
            D = torch.diag(A.sum(dim=-1))
            L = D - A
            try:
                evals, evecs = torch.linalg.eigh(L)
                pe = evecs[:, 1:self.n_eigvecs + 1]  # skip trivial eigenvector
                if pe.size(1) < self.n_eigvecs:
                    pad = torch.zeros(N, self.n_eigvecs - pe.size(1), device=device)
                    pe = torch.cat([pe, pad], dim=-1)
            except Exception:
                pe = torch.zeros(N, self.n_eigvecs, device=device)
            encodings.append(pe)
        return torch.stack(encodings)


class RandomWalkEncoding(nn.Module):
    """Positional encoding from random walk landing probabilities."""
    def __init__(self, embed_dim: int, walk_length: int = 8):
        super().__init__()
        self.walk_length = walk_length
        self.proj = nn.Linear(walk_length, embed_dim)

    def forward(self, x: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        if adjacency is None:
            return x
        rw = self._random_walk_pe(adjacency)
        return x + self.proj(rw)

    def _random_walk_pe(self, adj: torch.Tensor) -> torch.Tensor:
        """Compute random walk landing probabilities."""
        B, N, _ = adj.shape
        # Normalize adjacency to transition matrix
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        T = adj / deg

        pe_list = []
        power = T.clone()
        for _ in range(self.walk_length):
            pe_list.append(torch.diagonal(power, dim1=-2, dim2=-1))
            power = torch.bmm(power, T)
        return torch.stack(pe_list, dim=-1)


# ---- Part 4: Category-Theoretic Attention ----

class CategoricalAttention(nn.Module):
    """Attention as morphism interaction in a category.

    Objects = theorem states, Morphisms = transformations.
    Attention scores = morphism relevance (compositional reasoning).
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        # Object and morphism projections
        self.W_obj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_mor = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_val = nn.Linear(embed_dim, embed_dim, bias=False)

        # Composition-aware scoring
        self.compose_score = nn.Sequential(
            nn.Linear(2 * self.d_head, self.d_head),
            nn.GELU(),
            nn.Linear(self.d_head, 1),
        )
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                morphism_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x

        # Objects and morphisms from same representation
        obj = self.W_obj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        mor = self.W_mor(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        val = self.W_val(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Compositional attention: score(i,j) = f(obj_i, mor_j)
        obj_exp = obj.unsqueeze(-2).expand(-1, -1, -1, T, -1)
        mor_exp = mor.unsqueeze(-3).expand(-1, -1, T, -1, -1)
        pair = torch.cat([obj_exp, mor_exp], dim=-1)
        scores = self.compose_score(pair).squeeze(-1) / (self.d_head ** 0.5)

        if morphism_mask is not None:
            scores = scores.masked_fill(morphism_mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, val)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.Wo(out)
        return self.norm(out + residual)

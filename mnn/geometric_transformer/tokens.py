"""mnn.geometric_transformer.tokens — Geometric Token Representation.

Part 1: Tokens are no longer flat vectors. They represent theorems,
operators, PDE states, morphisms — each living on a geometric manifold.

Part 2: Geometric Attention — attention using geodesic similarity,
manifold distance, and curvature-aware interactions.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TokenType(Enum):
    THEOREM = "theorem"
    OPERATOR = "operator"
    PDE_STATE = "pde_state"
    MORPHISM = "morphism"
    MANIFOLD_POINT = "manifold_point"
    EQUATION = "equation"
    GENERIC = "generic"


class GeometricToken:
    """A token living on a geometric manifold, not in flat space."""
    def __init__(self, embedding: np.ndarray, token_type: TokenType = TokenType.GENERIC,
                 curvature: float = 0.0, metadata: Optional[Dict] = None):
        self.embedding = np.asarray(embedding, dtype=float)
        self.token_type = token_type
        self.curvature = curvature
        self.metadata = metadata or {}
        self.dim = len(self.embedding)

    def geodesic_distance(self, other: "GeometricToken") -> float:
        """Distance accounting for curvature."""
        if abs(self.curvature) < 1e-10:
            return float(np.linalg.norm(self.embedding - other.embedding))
        elif self.curvature > 0:
            # Spherical: arccos of normalized dot product
            n1 = self.embedding / (np.linalg.norm(self.embedding) + 1e-15)
            n2 = other.embedding / (np.linalg.norm(other.embedding) + 1e-15)
            cos_d = np.clip(np.dot(n1, n2), -1, 1)
            return float(np.arccos(cos_d) / np.sqrt(self.curvature))
        else:
            # Hyperbolic (Poincaré): use Lorentzian distance
            diff = self.embedding - other.embedding
            return float(np.sqrt(abs(self.curvature)) * np.linalg.norm(diff))

    def parallel_transport(self, vector: np.ndarray, target: "GeometricToken") -> np.ndarray:
        """Approximate parallel transport of a vector to another point."""
        if abs(self.curvature) < 1e-10:
            return vector.copy()
        direction = target.embedding - self.embedding
        norm = np.linalg.norm(direction) + 1e-15
        direction /= norm
        proj = np.dot(vector, direction) * direction
        return vector - self.curvature * proj * norm

    def exponential_map(self, tangent: np.ndarray) -> "GeometricToken":
        """Map tangent vector to manifold point."""
        new_emb = self.embedding + tangent
        if self.curvature > 0:
            norm = np.linalg.norm(new_emb) + 1e-15
            new_emb = new_emb / norm / np.sqrt(self.curvature)
        return GeometricToken(new_emb, self.token_type, self.curvature)

    def logarithmic_map(self, target: "GeometricToken") -> np.ndarray:
        """Map manifold point to tangent vector at self."""
        return target.embedding - self.embedding


class ManifoldEmbedding(nn.Module):
    """Embed tokens into a curved manifold space."""
    def __init__(self, vocab_size: int, embed_dim: int, curvature: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.type_embedding = nn.Embedding(len(TokenType), embed_dim)
        self.curvature_param = nn.Parameter(torch.tensor(float(curvature)))
        nn.init.normal_(self.embedding.weight, 0, 0.05)

    def forward(self, token_ids: torch.Tensor,
                type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.embedding(token_ids)
        if type_ids is not None:
            emb = emb + self.type_embedding(type_ids)
        if abs(self.curvature) > 1e-10:
            emb = self._project_to_manifold(emb)
        return emb

    def _project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        if self.curvature_param > 0:
            return F.normalize(x, dim=-1) / torch.sqrt(torch.abs(self.curvature_param) + 1e-8)
        return x


# ---- Part 2: Geometric Attention ----

class GeometricAttention(nn.Module):
    """Attention using geodesic distances and curvature-aware scoring.

    Instead of dot-product attention in flat space, computes attention
    weights based on manifold geometry.
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, curvature: float = 0.0,
                 dropout: float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads
        self.curvature = nn.Parameter(torch.tensor(curvature))

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x

        Q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = self._geometric_scores(Q, K)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.Wo(out)
        return self.norm(out + residual)

    def _geometric_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using geodesic-aware similarity."""
        # Standard dot product component
        dot = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Curvature correction: penalize by squared distance
        if abs(self.curvature.item()) > 1e-10:
            Q_norm = Q.unsqueeze(-2)
            K_norm = K.unsqueeze(-3)
            dist_sq = ((Q_norm - K_norm) ** 2).sum(dim=-1)
            curvature_correction = -0.5 * self.curvature * dist_sq
            dot = dot + curvature_correction

        return dot

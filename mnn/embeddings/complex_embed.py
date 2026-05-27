"""mnn.embeddings.complex_embed — Complex-Valued Theorem Embeddings.

Part 3: Embed theorems as complex vectors encoding magnitude (importance)
and phase (relational structure). Enables interference, duality, symmetry,
and equivalence representation.

Part 4: Theorem Similarity via quantum inner products <T1|T2>.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class ComplexEmbeddingLayer(nn.Module):
    """Complex-valued embedding: each token -> (real, imag) vector.

    Encodes both magnitude (importance) and phase (relational role).
    """
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.real_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.imag_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self._init()

    def _init(self):
        nn.init.normal_(self.real_embed.weight, 0, 0.1)
        nn.init.normal_(self.imag_embed.weight, 0, 0.1)

    def forward(self, token_ids: torch.Tensor):
        """Returns (real, imag) embeddings."""
        return self.real_embed(token_ids), self.imag_embed(token_ids)

    def magnitude(self, token_ids: torch.Tensor) -> torch.Tensor:
        r, i = self.forward(token_ids)
        return torch.sqrt(r**2 + i**2 + 1e-8)

    def phase(self, token_ids: torch.Tensor) -> torch.Tensor:
        r, i = self.forward(token_ids)
        return torch.atan2(i, r)


class PositionalPhaseEncoding(nn.Module):
    """Position encoding via complex phases: exp(i * theta * pos)."""
    def __init__(self, embed_dim: int, max_len: int = 256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        dim = torch.arange(embed_dim).unsqueeze(0).float()
        angles = pos / (10000 ** (2 * dim / embed_dim))
        self.register_buffer("cos_pos", torch.cos(angles))
        self.register_buffer("sin_pos", torch.sin(angles))

    def forward(self, real: torch.Tensor, imag: torch.Tensor):
        seq_len = real.size(-2)
        cos_p = self.cos_pos[:seq_len]
        sin_p = self.sin_pos[:seq_len]
        # Complex multiplication: (a+bi)(cos+isin) = (a*cos - b*sin) + i(a*sin + b*cos)
        new_real = real * cos_p - imag * sin_p
        new_imag = real * sin_p + imag * cos_p
        return new_real, new_imag


class TheoremEncoder(nn.Module):
    """Encode tokenized theorems into complex-valued Hilbert space vectors.

    Architecture: embed tokens -> positional phase -> complex attention -> pool -> normalize.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, max_len: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = ComplexEmbeddingLayer(vocab_size, embed_dim)
        self.pos_encoding = PositionalPhaseEncoding(embed_dim, max_len)
        self.attn_layers = nn.ModuleList([
            ComplexSelfAttention(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.out_proj_r = nn.Linear(embed_dim, embed_dim)
        self.out_proj_i = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor):
        """token_ids: (batch, seq_len) -> (batch, embed_dim) complex vector."""
        r, i = self.embedding(token_ids)
        r, i = self.pos_encoding(r, i)

        mask = (token_ids != 0).float().unsqueeze(-1)
        for attn in self.attn_layers:
            r, i = attn(r, i, mask)

        # Masked mean pooling
        r = (r * mask).sum(dim=-2) / (mask.sum(dim=-2) + 1e-8)
        i = (i * mask).sum(dim=-2) / (mask.sum(dim=-2) + 1e-8)

        r = self.out_proj_r(r)
        i = self.out_proj_i(i)

        # Normalize to unit vector in complex space
        norm = torch.sqrt((r**2 + i**2).sum(dim=-1, keepdim=True) + 1e-8)
        return r / norm, i / norm

    def encode_numpy(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            r, i = self.forward(torch.tensor(token_ids, dtype=torch.long))
            return r.numpy(), i.numpy()


class ComplexSelfAttention(nn.Module):
    """Self-attention in complex space for theorem encoding."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.Wq_r = nn.Linear(dim, dim, bias=False)
        self.Wq_i = nn.Linear(dim, dim, bias=False)
        self.Wk_r = nn.Linear(dim, dim, bias=False)
        self.Wk_i = nn.Linear(dim, dim, bias=False)
        self.Wv_r = nn.Linear(dim, dim, bias=False)
        self.Wv_i = nn.Linear(dim, dim, bias=False)
        self.out_r = nn.Linear(dim, dim)
        self.out_i = nn.Linear(dim, dim)
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)

    def forward(self, r: torch.Tensor, i: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        B, T, D = r.shape
        # Q, K, V in complex space
        qr, qi = self.Wq_r(r), self.Wq_i(i)
        kr, ki = self.Wk_r(r), self.Wk_i(i)
        vr, vi = self.Wv_r(r), self.Wv_i(i)

        # Complex dot product: Re(<q|k>) for attention scores
        attn_scores = (torch.matmul(qr, kr.transpose(-2, -1))
                       + torch.matmul(qi, ki.transpose(-2, -1))) / (D ** 0.5)
        if mask is not None:
            pad_mask = mask.squeeze(-1).unsqueeze(-2)
            attn_scores = attn_scores.masked_fill(pad_mask == 0, -1e9)
        attn = F.softmax(attn_scores, dim=-1)

        out_r = self.out_r(torch.matmul(attn, vr))
        out_i = self.out_i(torch.matmul(attn, vi))

        # Residual + norm
        out_r = self.norm_r(out_r + r)
        out_i = self.norm_i(out_i + i)
        return out_r, out_i


class TheoremSimilarity:
    """Part 4: Quantum-style similarity between theorem embeddings.

    <T1|T2> = complex inner product (amplitude overlap).
    |<T1|T2>|^2 = overlap probability.
    """
    @staticmethod
    def inner_product(r1: np.ndarray, i1: np.ndarray,
                      r2: np.ndarray, i2: np.ndarray) -> complex:
        """<T1|T2> = sum(conj(T1) * T2)."""
        t1 = r1 + 1j * i1
        t2 = r2 + 1j * i2
        return complex(np.vdot(t1, t2))

    @staticmethod
    def overlap(r1: np.ndarray, i1: np.ndarray,
                r2: np.ndarray, i2: np.ndarray) -> float:
        """|<T1|T2>|^2"""
        return abs(TheoremSimilarity.inner_product(r1, i1, r2, i2)) ** 2

    @staticmethod
    def similarity_matrix(reals: np.ndarray, imags: np.ndarray) -> np.ndarray:
        """Pairwise similarity matrix for a batch of embeddings."""
        Z = reals + 1j * imags  # (n, dim)
        G = Z.conj() @ Z.T
        return np.abs(G) ** 2

    @staticmethod
    def nearest_theorems(query_r: np.ndarray, query_i: np.ndarray,
                          db_r: np.ndarray, db_i: np.ndarray,
                          names: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar theorems to a query."""
        q = query_r + 1j * query_i
        db = db_r + 1j * db_i
        sims = np.abs(db.conj() @ q) ** 2
        top_idx = np.argsort(-sims)[:top_k]
        return [(names[i], float(sims[i])) for i in top_idx]

    @staticmethod
    def interference_term(r1: np.ndarray, i1: np.ndarray,
                          r2: np.ndarray, i2: np.ndarray) -> float:
        """Quantum interference: 2*Re(<T1|T2>) — constructive/destructive."""
        ip = TheoremSimilarity.inner_product(r1, i1, r2, i2)
        return float(2 * ip.real)

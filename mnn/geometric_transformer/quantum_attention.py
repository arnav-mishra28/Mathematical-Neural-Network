"""mnn.geometric_transformer.quantum_attention — Quantum & PDE-Aware Attention.

Part 5: Quantum Geometric Attention — complex inner products ⟨ψ_i|ψ_j⟩
in Hilbert space for phase-sensitive, geometry-aware attention.

Part 6: PDE-Aware Transformers — reason over differential operators,
solution manifolds, and tensor flows.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class QuantumGeometricAttention(nn.Module):
    """Attention via complex inner products in Hilbert space.

    Instead of Q·K^T, use ⟨ψ_Q|ψ_K⟩ = Re(Q_r K_r + Q_i K_i) + Im(...)
    Phase-sensitive, interference-aware, topology-preserving.
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        # Complex Q, K, V projections (real + imaginary)
        self.Wq_r = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_i = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk_r = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk_i = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_r = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv_i = nn.Linear(embed_dim, embed_dim, bias=False)

        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.phase_gate = nn.Parameter(torch.zeros(n_heads))  # learnable phase

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x

        # Complex projections
        Qr = self.Wq_r(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Qi = self.Wq_i(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Kr = self.Wk_r(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Ki = self.Wk_i(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Vr = self.Wv_r(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Vi = self.Wv_i(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Complex inner product: Re(⟨Q|K⟩) = Q_r·K_r + Q_i·K_i
        real_part = torch.matmul(Qr, Kr.transpose(-2, -1)) + torch.matmul(Qi, Ki.transpose(-2, -1))
        # Im(⟨Q|K⟩) for interference
        imag_part = torch.matmul(Qi, Kr.transpose(-2, -1)) - torch.matmul(Qr, Ki.transpose(-2, -1))

        # Phase-modulated scores
        phase = self.phase_gate.view(1, self.n_heads, 1, 1)
        scores = (real_part * torch.cos(phase) - imag_part * torch.sin(phase)) / (self.d_head ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply to real part of values
        out = torch.matmul(attn, Vr)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.Wo(out)
        return self.norm(out + residual)


class PDEAwareAttention(nn.Module):
    """Part 6: Attention that reasons over PDE structure.

    Integrates differential operator information into attention:
    tokens carry both value AND operator context (gradient, Laplacian).
    """
    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

        # Differential operator projections
        self.W_grad = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_lap = nn.Linear(embed_dim, embed_dim, bias=False)
        self.operator_gate = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x

        # Approximate spatial derivatives via differences
        grad_x = torch.zeros_like(x)
        lap_x = torch.zeros_like(x)
        if T > 2:
            grad_x[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
            lap_x[:, 1:-1] = x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2]

        # Operator-enriched representation
        grad_feat = self.W_grad(grad_x)
        lap_feat = self.W_lap(lap_x)
        gate = self.operator_gate(torch.cat([x, grad_feat, lap_feat], dim=-1))
        x_enriched = x + gate * (grad_feat + lap_feat)

        Q = self.Wq(x_enriched).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(x_enriched).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(x_enriched).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.Wo(out)
        return self.norm(out + residual)

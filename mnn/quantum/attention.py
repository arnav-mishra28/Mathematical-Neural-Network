"""mnn.quantum.attention — Quantum-inspired attention mechanisms.

Part 4: Attention where Q, K, V are quantum states in Hilbert space.
Similarity is the inner product <phi|psi> instead of plain dot products.
This gives amplitude overlap, interference effects, and geometric resonance.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
import math


class QuantumAttentionHead(nn.Module):
    """Single head of quantum-inspired attention.

    Q, K, V are projected into complex space (real + imaginary parts).
    Attention scores use the squared modulus of the complex inner product:
      alpha_{ij} = |<q_i | k_j>|^2 / Z
    This captures amplitude overlap + phase interference.
    """
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        # Project to complex Q, K, V (real + imag parts)
        self.Wq_r = nn.Linear(d_model, d_head, bias=False)
        self.Wq_i = nn.Linear(d_model, d_head, bias=False)
        self.Wk_r = nn.Linear(d_model, d_head, bias=False)
        self.Wk_i = nn.Linear(d_model, d_head, bias=False)
        self.Wv_r = nn.Linear(d_model, d_head, bias=False)
        self.Wv_i = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Complex projections
        qr, qi = self.Wq_r(x), self.Wq_i(x)
        kr, ki = self.Wk_r(x), self.Wk_i(x)
        vr, vi = self.Wv_r(x), self.Wv_i(x)

        # Normalize to unit vectors in complex space
        q_norm = torch.sqrt(qr**2 + qi**2 + 1e-8)
        qr, qi = qr / q_norm, qi / q_norm
        k_norm = torch.sqrt(kr**2 + ki**2 + 1e-8)
        kr, ki = kr / k_norm, ki / k_norm

        # Complex inner product: <q|k> = (qr + i*qi)^dag (kr + i*ki)
        # Real part: qr*kr + qi*ki, Imag part: qr*ki - qi*kr
        attn_real = torch.matmul(qr, kr.transpose(-2, -1)) + torch.matmul(qi, ki.transpose(-2, -1))
        attn_imag = torch.matmul(qr, ki.transpose(-2, -1)) - torch.matmul(qi, kr.transpose(-2, -1))

        # |<q|k>|^2 = real^2 + imag^2
        attn_scores = (attn_real**2 + attn_imag**2) / math.sqrt(self.d_head)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply to complex values
        out_r = torch.matmul(attn_weights, vr)
        out_i = torch.matmul(attn_weights, vi)

        return out_r, out_i, attn_weights


class QuantumMultiHeadAttention(nn.Module):
    """Multi-head quantum attention with complex-valued projections.

    Combines n_heads quantum attention heads, concatenates their
    complex outputs, and projects back to d_model.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.heads = nn.ModuleList([
            QuantumAttentionHead(d_model, self.d_head) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        head_outputs_r = []
        head_outputs_i = []
        all_weights = []

        for head in self.heads:
            r, i, w = head(x, mask)
            head_outputs_r.append(r)
            head_outputs_i.append(i)
            all_weights.append(w)

        concat_r = torch.cat(head_outputs_r, dim=-1)
        concat_i = torch.cat(head_outputs_i, dim=-1)
        combined = torch.cat([concat_r, concat_i], dim=-1)

        out = self.out_proj(combined)
        out = self.dropout(out)
        out = self.norm(out + x)  # residual

        return out, all_weights


class QuantumPhaseAttention(nn.Module):
    """Phase-only quantum attention.

    Uses relative phase differences for attention scores:
      alpha_{ij} = cos(phase_q_i - phase_k_j)
    Captures rotational alignment in complex space.
    """
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.Wq = nn.Linear(d_model, d_head, bias=False)
        self.Wk = nn.Linear(d_model, d_head, bias=False)
        self.Wv = nn.Linear(d_model, d_head, bias=False)
        self.phase_q = nn.Linear(d_model, d_head, bias=False)
        self.phase_k = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        pq = self.phase_q(x)
        pk = self.phase_k(x)

        # Phase difference attention
        phase_diff = pq.unsqueeze(-2) - pk.unsqueeze(-3)  # (B, T, T, d)
        attn = torch.cos(phase_diff).mean(dim=-1) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v), attn


class QuantumInterferenceAttention(nn.Module):
    """Attention via quantum interference.

    Models constructive/destructive interference between query and key amplitudes.
    Score = |psi_q + psi_k|^2 - |psi_q|^2 - |psi_k|^2 = 2 Re(<q|k>)
    """
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.Wqr = nn.Linear(d_model, d_head, bias=False)
        self.Wqi = nn.Linear(d_model, d_head, bias=False)
        self.Wkr = nn.Linear(d_model, d_head, bias=False)
        self.Wki = nn.Linear(d_model, d_head, bias=False)
        self.Wv = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor):
        qr, qi = self.Wqr(x), self.Wqi(x)
        kr, ki = self.Wkr(x), self.Wki(x)
        v = self.Wv(x)

        # Interference term: 2 Re(<q|k>) = 2(qr*kr + qi*ki)
        interference = 2 * (torch.matmul(qr, kr.transpose(-2, -1))
                           + torch.matmul(qi, ki.transpose(-2, -1)))
        scores = interference / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn


class QuantumTransformerBlock(nn.Module):
    """Transformer block using quantum-inspired attention + feedforward."""
    def __init__(self, d_model: int, n_heads: int = 4,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = QuantumMultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out, weights = self.attn(x, mask)
        x = self.norm(self.ff(attn_out) + attn_out)
        return x, weights


class QuantumTransformer(nn.Module):
    """Full quantum-inspired transformer encoder."""
    def __init__(self, input_dim: int, output_dim: int,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor):
        h = self.embed(x)
        if h.dim() == 2:
            h = h.unsqueeze(1)
        all_weights = []
        for block in self.blocks:
            h, w = block(h)
            all_weights.append(w)
        return self.head(h.squeeze(1)), all_weights

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

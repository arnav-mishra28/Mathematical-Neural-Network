"""mnn.spectral_manifold.spectral_attention — Spectral & Quantum Attention.

Part 6: Spectral Attention — attention in eigenfunction space using
manifold harmonics and global topology.

Part 7: Quantum Spectral Geometry — combine quantum embeddings with
spectral theory: |ψ⟩ = Σ c_i φ_i (spectral wavefunctions).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SpectralAttention(nn.Module):
    """Part 6: Attention in eigenfunction/harmonic space.

    Instead of attending in raw coordinates, project Q,K,V into
    spectral domain, attend there, then reconstruct.
    """
    def __init__(self, embed_dim: int, n_harmonics: int = 16,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_harmonics = n_harmonics
        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

        # Learnable spectral basis (approximation)
        self.spectral_basis = nn.Parameter(
            torch.randn(n_harmonics, embed_dim) * 0.05)

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

        # Spectral filter (learnable frequency response)
        self.freq_filter = nn.Parameter(torch.ones(n_heads, n_harmonics))
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                eigenvectors: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        residual = x

        # Project to spectral domain
        if eigenvectors is not None and eigenvectors.shape[-2] == T:
            basis = eigenvectors  # (B, T, n_harmonics)
        else:
            # Use learnable basis: approximate spectral projection
            basis = F.softmax(x @ self.spectral_basis.T, dim=-1)  # (B, T, n_harmonics)

        Q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Spectral modulation: weight attention by harmonic overlap
        spectral_sim = torch.matmul(basis, basis.transpose(-2, -1))  # (B, T, T)
        # Apply learnable frequency filter
        freq_weight = self.freq_filter.mean(dim=0)  # average across heads
        filtered_basis = basis * freq_weight.unsqueeze(0).unsqueeze(0)
        spectral_mod = torch.matmul(filtered_basis, basis.transpose(-2, -1))
        scores = scores + spectral_mod.unsqueeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.Wo(out)
        return self.norm(out + residual)


class QuantumSpectralLayer(nn.Module):
    """Part 7: Quantum spectral geometry.

    |ψ⟩ = Σ c_i φ_i — states as superpositions of manifold harmonics.
    Complex-valued spectral decomposition with phase structure.
    """
    def __init__(self, state_dim: int, n_harmonics: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.n_harmonics = n_harmonics

        # Learnable harmonic basis (complex)
        self.basis_r = nn.Parameter(torch.randn(n_harmonics, state_dim) * 0.05)
        self.basis_i = nn.Parameter(torch.randn(n_harmonics, state_dim) * 0.05)

        # Spectral filter (complex frequency response)
        self.filter_r = nn.Parameter(torch.ones(n_harmonics))
        self.filter_i = nn.Parameter(torch.zeros(n_harmonics))

        self.reconstruct_r = nn.Linear(n_harmonics, state_dim)
        self.reconstruct_i = nn.Linear(n_harmonics, state_dim)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        """Spectral transform, filter, reconstruct in complex domain.

        x_r, x_i: (batch, state_dim) real and imaginary parts.
        """
        # Project onto spectral basis: c_i = <φ_i | ψ>
        # Re(<φ|ψ>) = φ_r·ψ_r + φ_i·ψ_i
        coeffs_r = x_r @ self.basis_r.T + x_i @ self.basis_i.T
        coeffs_i = x_i @ self.basis_r.T - x_r @ self.basis_i.T

        # Apply spectral filter: c_i * h(λ_i)
        fr = self.filter_r
        fi = self.filter_i
        filtered_r = coeffs_r * fr - coeffs_i * fi
        filtered_i = coeffs_r * fi + coeffs_i * fr

        # Reconstruct
        out_r = self.reconstruct_r(filtered_r)
        out_i = self.reconstruct_i(filtered_i)

        # Normalize
        norm = torch.sqrt((out_r**2 + out_i**2).sum(dim=-1, keepdim=True) + 1e-8)
        return out_r / norm, out_i / norm

    def spectral_energy(self, x_r: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """Energy at each harmonic: |c_i|^2."""
        coeffs_r = x_r @ self.basis_r.T + x_i @ self.basis_i.T
        coeffs_i = x_i @ self.basis_r.T - x_r @ self.basis_i.T
        return coeffs_r**2 + coeffs_i**2

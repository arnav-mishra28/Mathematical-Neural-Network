"""
mnn.spectral_manifold — Spectral Manifold Learning

Learn the intrinsic frequencies and harmonics of mathematical structures.
Geometry has hidden frequencies — spectral theory reveals global manifold structure.

Submodules:
  laplacian          — Manifold Laplacian + spectral decomposition (Parts 1-2)
  embeddings         — Graph spectral analysis, spectral embeddings, manifold harmonics (Parts 3-5)
  spectral_attention — Spectral attention + quantum spectral geometry (Parts 6-7)
  pde_spectral       — PDE spectral solvers + theorem spectral topology (Parts 8-9)
"""
from .laplacian import ManifoldLaplacian, SpectralDecomposition
from .embeddings import GraphSpectralAnalyzer, SpectralEmbedding, ManifoldHarmonics
from .spectral_attention import SpectralAttention, QuantumSpectralLayer
from .pde_spectral import SpectralPDEEvolver, TheoremSpectralTopology

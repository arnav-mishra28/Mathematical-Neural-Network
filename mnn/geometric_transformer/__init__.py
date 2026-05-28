"""
mnn.geometric_transformer — Geometric Transformer Architectures

Geometry-aware reasoning architectures for mathematical intelligence.
Attention over manifolds, graphs, curved spaces, and theorem geometries.

Submodules:
  tokens              — Geometric tokens + geodesic attention (Parts 1-2)
  positional           — Spectral/graph positional encodings + categorical attention (Parts 3-4)
  quantum_attention    — Quantum geometric + PDE-aware attention (Parts 5-6)
  graph_transformer    — Theorem graph transformers + hierarchical reasoning (Parts 7-8)
"""
from .tokens import (
    TokenType, GeometricToken, ManifoldEmbedding, GeometricAttention,
)
from .positional import (
    SpectralPositionalEncoding, RandomWalkEncoding, CategoricalAttention,
)
from .quantum_attention import QuantumGeometricAttention, PDEAwareAttention
from .graph_transformer import (
    EdgeType, GraphAttentionLayer, TheoremGraphTransformer,
    HierarchicalPooling, HierarchicalGeometricTransformer,
)

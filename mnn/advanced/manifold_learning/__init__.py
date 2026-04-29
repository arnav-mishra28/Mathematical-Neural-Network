"""
mnn.advanced.manifold_learning
================================
Topology + Manifold Learning System (Phase 4).

Your MNN now learns the STRUCTURE OF SPACE ITSELF, not just functions on it.

Parts
-----
  datasets      — Manifold dataset generators (S¹, S², Sⁿ, torus, Klein bottle, etc.)
  autoencoder   — Manifold-aware autoencoders with topological constraints
  constraints   — Topological constraint losses (on-manifold, geodesic, curvature)
  geometry      — Intrinsic geometry computations (geodesics, curvature, connection)
  analysis      — Manifold analysis tools (intrinsic dim, Betti numbers, persistence)
  visualization — Manifold visualization (2D/3D projections, latent space plots)
"""
from .datasets     import ManifoldDataset
from .autoencoder  import ManifoldAutoencoder, ManifoldVAE
from .constraints  import ManifoldConstraints
from .geometry     import IntrinsicGeometry
from .analysis     import ManifoldAnalyzer

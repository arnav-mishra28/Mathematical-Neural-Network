"""
mnn.quantum_pde — Quantum PDE Solvers

Learn PDE solution spaces geometrically using quantum-inspired representations.
Intersection of PDEs, quantum computing, and geometric deep learning.

Submodules:
  states          — Quantum state PDE representation + operator formulation (Parts 1-2)
  neural_operator — Neural operators + quantum evolution (Parts 3-4)
  spectral        — Spectral PDE solving + geometric regularization (Parts 5-6)
  pde_discovery   — Discover governing PDEs from data (Part 7)
  categorical_pde — Category-theoretic PDE structure (Part 8)
"""
from .states import QuantumPDEState, PDEOperator
from .neural_operator import (
    NeuralPDEOperator, IntegralKernelLayer,
    QuantumEvolutionLayer, QuantumPDENet,
)
from .spectral import (
    SpectralPDESolver, SpectralLayer, SpectralPDENet,
    GeometricPDERegularizer,
)
from .pde_discovery import PDELibrary, SparsePDERegressor, PDEDiscoveryEngine
from .categorical_pde import (
    SolutionSpace, PDEMorphism, PDECategory, DiscretizationFunctor,
)

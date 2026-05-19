"""
MNN Phase 3 — Advanced Mathematical Intelligence Layer

Four tightly-integrated pillars:
  1. dynamical     — Dynamical + Nonlinear Systems (flow maps, stability, bifurcations)
  2. group_algebra — Abstract Algebra Engine (symmetry-aware neural group operations)
  3. neural_pde    — Neural PDE Solvers (generalised PINN for arbitrary PDEs)
  4. discovery     — Scientific Discovery Engine (hybrid neural-symbolic regression)
"""
from .dynamical     import FlowMapLearner, StabilityAnalyzer, BifurcationDetector
from .group_algebra import NeuralGroupOperator, EquivariantNetwork, InvariantLearner
from .neural_pde    import NeuralPDESolver, GeneralizedPINN, PDEProblem
from .discovery     import ScientificDiscoveryEngine, SparseRegressor, HybridDiscovery

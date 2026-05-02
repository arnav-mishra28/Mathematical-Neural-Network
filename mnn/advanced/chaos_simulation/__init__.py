"""
mnn.advanced.chaos_simulation
================================
Chaos Simulation + Neural Discovery Engine (Phase 5).

Your MNN now LEARNS CHAOTIC DYNAMICS and attempts to DISCOVER
the underlying equations governing them.

Parts
-----
  simulator     — Lorenz + 6 other chaotic system simulators
  learner       — Neural dynamics learner (state → derivative prediction)
  predictor     — Short-term trajectory predictor with uncertainty
  discovery     — Sparse equation discovery (SINDy-style symbolic regression)
  analyzer      — Chaos diagnostics (Lyapunov, predictability horizon, etc.)
"""
from .simulator  import ChaosSimulator, LorenzSimulator, RosslerSimulator
from .learner    import DynamicsNet, DynamicsTrainer, DynamicsResult
from .predictor  import ChaosPredictor, EnsemblePredictor
from .discovery  import EquationDiscovery, SINDyEngine, SymbolicTerm
from .analyzer   import ChaosNeuralAnalyzer

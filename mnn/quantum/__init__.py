"""
mnn.quantum — Quantum-Inspired Neural Mathematics

Borrows mathematical structures from quantum mechanics to create
richer neural representations. NOT a quantum computer simulator —
uses complex-valued state spaces, unitary dynamics, and quantum geometry
to extend classical neural computation.

Submodules:
  hilbert    — Hilbert spaces, quantum states, operator algebra, quantum gates
  complex_nn — Complex-valued neural networks (magnitude + phase processing)
  unitary    — Unitary transformations, norm-preserving layers, reversible nets
  attention  — Quantum-inspired attention (amplitude overlap, interference)
  geometric  — Quantum geometric learning (Fubini-Study manifold, curvature)
  chaos      — Quantum chaos (random matrices, spectral statistics, OTOC)
"""
from .hilbert import QuantumState, HilbertSpace, QuantumGates
from .complex_nn import (
    ComplexLinear, ComplexActivation, ComplexBlock,
    ComplexNeuralNetwork, ComplexTrainer,
)
from .unitary import (
    UnitaryLayer, UnitaryBlock, UnitaryNetwork,
    UnitaryConstraintLoss, ParameterizedUnitary, UnitaryTrainer,
)
from .attention import (
    QuantumAttentionHead, QuantumMultiHeadAttention,
    QuantumPhaseAttention, QuantumInterferenceAttention,
    QuantumTransformerBlock, QuantumTransformer,
)
from .geometric import (
    FubiniStudyMetric, QuantumEmbedding,
    QuantumGeometricLayer, QuantumGeometricNetwork,
)
from .chaos import (
    RandomMatrixEnsemble, SpectralAnalyzer,
    QuantumKickedTop, QuantumLyapunov, QuantumEntanglementDynamics,
)

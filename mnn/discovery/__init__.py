"""
mnn.discovery — Mathematical Discovery Engine

Automated theorem discovery: observe structures, detect patterns,
generate hypotheses, and validate them.

Engines:
  representation  — Expression trees, math object encoding
  patterns        — Symmetry detection, invariant finding, sequence analysis
  conjectures     — Hypothesis generation (algebraic, geometric, dynamical)
  validation      — Numerical verification, counterexample search, symbolic checks
  neural_search   — Neural-guided theorem search (beam search + learned heuristics)
  categorical     — Category-theoretic discovery (morphism invariants, equivalences)
"""
from .representation import ExprNode, MathObject, MathEncoder
from .patterns import SymmetryDetector, InvariantFinder, SequenceAnalyzer
from .conjectures import (
    Conjecture, ConjectureType, ConjectureStatus, ConjectureGenerator,
)
from .validation import (
    NumericalValidator, CounterexampleSearcher,
    SymbolicValidator, ConjectureValidator,
)
from .neural_search import (
    TheoremStep, ProofPath, TransformationRule,
    TransformationLibrary, NeuralTheoremSearcher,
)
from .categorical import (
    MorphismInvariantFinder, EquivalenceDiscovery,
    FunctorialDiscovery, CategoricalDiscoveryEngine,
)

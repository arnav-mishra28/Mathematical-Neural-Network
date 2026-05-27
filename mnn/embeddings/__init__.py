"""
mnn.embeddings — Quantum-Inspired Theorem Embeddings

Bridge between symbolic mathematics, geometric learning, and
quantum-inspired representation. Represents theorems, proofs,
and conjectures as points in a complex Hilbert space.

Submodules:
  theorem_states      — Quantum state representation of theorems (Part 1)
  tokenizer           — Mathematical tokenization with type annotations (Part 2)
  complex_embed       — Complex-valued embeddings + similarity (Parts 3-4)
  proof_trajectories  — Proof paths through theorem space (Part 5)
  categorical_embed   — Category-theoretic embeddings (Part 6)
  training            — Training objectives (4 losses) (Part 7)
"""
from .theorem_states import MathConcept, ConceptBasis, TheoremState
from .tokenizer import MathToken, MathVocabulary, MathTokenizer
from .complex_embed import (
    ComplexEmbeddingLayer, TheoremEncoder,
    TheoremSimilarity, ComplexSelfAttention,
)
from .proof_trajectories import (
    ProofStep, ProofTrajectory,
    ProofPathPredictor, ProofNavigator,
)
from .categorical_embed import CategoricalTheoremSpace, TheoremFunctor
from .training import (
    StructuralSimilarityLoss, ProofContinuityLoss,
    AlgebraicConsistencyLoss, TopologicalRegularizer,
    TheoremEmbeddingTrainer,
)

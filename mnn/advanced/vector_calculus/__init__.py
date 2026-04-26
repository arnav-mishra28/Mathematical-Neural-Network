"""
mnn.advanced.vector_calculus
==============================
Vector + Tensor Calculus Neural Network Engine (Phase 3).

Modules
-------
  field_networks        — Neural networks for scalar, vector, tensor fields
  operators             — Autograd-based ∇, ∇·, ∇×, ∇², ∇∇ (all via torch.autograd)
  constraints           — Divergence-free, curl-free, harmonic, solenoidal constraints
  trainer               — Field trainer: supervised + multi-constraint optimization
  symbolic_validation   — SymPy symbolic cross-validation of learned fields
  tensor_fields         — Rank-2 and rank-3 tensor field networks + covariant ops

Quick usage
-----------
  from mnn.advanced.vector_calculus import (
      ScalarFieldNet, VectorFieldNet, TensorFieldNet,
      FieldOperators, FieldConstraints, FieldTrainer,
      SymbolicValidator
  )
"""
from .field_networks       import ScalarFieldNet, VectorFieldNet, TensorFieldNet
from .operators            import FieldOperators
from .constraints          import FieldConstraints
from .trainer              import FieldTrainer, FieldTrainingResult
from .symbolic_validation  import SymbolicValidator
from .tensor_fields        import TensorFieldEngine

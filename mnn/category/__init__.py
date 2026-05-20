"""
mnn.category — Category Theory Engine

Foundational abstraction layer that unifies all MNN modules through
categorical structure: objects, morphisms, functors, natural transformations.

Submodules:
  core     — Category, Object, Morphism, composition, identity, products, coproducts
  functors — Functor, NaturalTransformation, MNN-specific bridge functors
  neural   — NeuralMorphism, LearnableFunctor, categorical neural computation
"""
from .core import (
    CatObject, Morphism, Category, IdentityMorphism,
    ProductCategory, OppositeCategory, SliceCategory,
    CompositionError,
)
from .functors import (
    Functor, NaturalTransformation, ContravariantFunctor,
    ForgetfulFunctor, FreeObjectFunctor,
    GeometryToAlgebraFunctor, AlgebraToComputationFunctor,
    DynamicsToLearningFunctor, UniversalBridgeFunctor,
)
from .neural import (
    NeuralMorphism, NeuralCategory, LearnableFunctor,
    CategoricalPipeline, NeuralNaturalTransformation,
)

"""
MNN Phase 2 — Advanced Mathematical Modules

Submodules (import individually to avoid optional-dependency issues):
  mnn.advanced.group_theory    — Deep group theory engine
  mnn.advanced.abelian         — Abelian theory + elliptic curves
  mnn.advanced.chaos_advanced  — Advanced chaos + fractal analysis
  mnn.advanced.prototype       — Derivative-constrained neural network (requires torch)
"""
from .group_theory   import (FiniteGroupAnalyzer, RepresentationTheory,
                              GroupHomomorphism, GroupExtension)
from .abelian        import (EllipticCurve, JacobiVariety,
                              AbelianVariety, ThetaDivisor)
from .chaos_advanced import (FractalAnalyzer, MultifractalSpectrum,
                              ChaoticMap, CoupledOscillators)
# Prototype is NOT auto-imported here because it requires torch.
# Import manually: from mnn.advanced.prototype import run_prototype

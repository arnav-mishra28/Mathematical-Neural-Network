"""Example 04 — Tensor Calculus"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.algebra.tensors   import TensorEngine
from mnn.core.math_engine  import MathEngine, TensorObject
from mnn.core.operators    import DifferentialOperators
eng = MathEngine()
te  = TensorEngine()
T1  = TensorObject(np.array([[1,2],[3,4]]),name="T1")
T2  = TensorObject(np.array([[5,6],[7,8]]),name="T2")
print(f"T1@T2 =\n{TensorEngine.einsum('ij,jk->ik',T1,T2).to_numpy()}")
eps = TensorEngine.levi_civita(3).to_numpy()
print(f"ε₀₁₂={eps[0,1,2]:.0f}  ε₁₀₂={eps[1,0,2]:.0f}")
eta = TensorEngine.metric_flat(4,"minkowski")
print(f"η_μν = {np.diag(eta.to_numpy())}")
f   = eng.define_function("sin(x)*cos(y)",["x","y"])
F   = eng.define_vector_field(["-y","x","0"],["x","y","z"])
print(f"∇f  = {eng.gradient(f)}")
print(f"∇·F = {DifferentialOperators.divergence(F)}")
print(f"∇×F = {DifferentialOperators.curl(F)}")
print("[OK] Example 04")

"""Example 05 — Group Theory & Abelian Functions"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.algebra.groups  import Group, LieGroup
from mnn.algebra.abelian import AbelianGroup, AbelianFunction
Z6  = Group.cyclic(6)
print(f"{Z6} | abelian={Z6.is_abelian()} | order(2)={Z6.element_order(2)}")
S3  = Group.symmetric(3)
print(f"{S3} | abelian={S3.is_abelian()} | simple={S3.is_simple()}")
so3 = LieGroup.SO(3)
print(f"so(3) basis dim={len(so3.lie_algebra_basis())} | semisimple={so3.is_semisimple()}")
Z8  = AbelianGroup.cyclic(8)
f   = {k:float(np.sin(k)) for k in range(8)}
fh  = Z8.fourier_transform(f)
fr  = Z8.inverse_fourier_transform(fh)
err = max(abs(fr[k]-f[k]) for k in range(8))
print(f"Z₈ Fourier roundtrip error: {err:.2e}")
print(f"θ₁(0.3|i) = {AbelianFunction.theta1(0.3,1j):.6f}")
print(f"θ₃(0.0|i) = {AbelianFunction.theta3(0.0,1j):.6f}")
af  = AbelianFunction([[1.2j]])
print(f"Θ(0.1|1.2i) = {af.riemann_theta([0.1+0.1j]):.6f}")
print("[OK] Example 05")

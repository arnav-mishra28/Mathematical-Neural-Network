"""mnn.core.operators — Differential operators for MNN."""
from __future__ import annotations
import numpy as np
import sympy as sp
from mnn.core.math_engine import ScalarField, VectorField, TensorObject

class DifferentialOperators:
    @staticmethod
    def gradient(f: ScalarField) -> VectorField: return f.gradient()
    @staticmethod
    def laplacian(f: ScalarField) -> ScalarField: return f.laplacian()
    @staticmethod
    def divergence(F: VectorField) -> ScalarField: return F.divergence()
    @staticmethod
    def curl(F: VectorField) -> VectorField: return F.curl()
    @staticmethod
    def biharmonic(f: ScalarField) -> ScalarField: return f.laplacian().laplacian()
    @staticmethod
    def d_alembertian(f: ScalarField, time_var: str, c: float=1.0) -> ScalarField:
        t=sp.Symbol(time_var); d2t=sp.diff(f.expr,t,2)
        lap=f.laplacian().expr
        return ScalarField(sp.simplify((1/c**2)*d2t - lap), f.var_names)
    @staticmethod
    def directional_derivative(f: ScalarField, direction: list) -> ScalarField:
        v=np.array(direction,dtype=float); v/=np.linalg.norm(v)
        grad=f.gradient()
        result=sum(float(v[i])*grad.components[i] for i in range(len(grad.components)))
        return ScalarField(sp.simplify(result), f.var_names)
    @staticmethod
    def vector_laplacian(F: VectorField) -> VectorField:
        return VectorField([ScalarField(c,F.var_names).laplacian().expr for c in F.components], F.var_names)
    @staticmethod
    def jacobian(F: VectorField) -> sp.Matrix: return F.jacobian()
    @staticmethod
    def exterior_derivative(f: ScalarField) -> VectorField: return f.gradient()
    @staticmethod
    def hodge_star_3d(F: VectorField) -> VectorField:
        if F.dim!=3: raise ValueError("3D only")
        Fx,Fy,Fz=F.components
        return VectorField([Fy,Fz,Fx], F.var_names)
    @staticmethod
    def numerical_gradient(f: np.ndarray, dx: float=1.0): return list(np.gradient(f,dx))
    @staticmethod
    def numerical_laplacian(f: np.ndarray, dx: float=1.0) -> np.ndarray:
        res=np.zeros_like(f)
        for i in range(f.ndim): res+=np.gradient(np.gradient(f,dx,axis=i),dx,axis=i)
        return res
    @staticmethod
    def numerical_curl_2d(Fx,Fy,dx=1.0): return np.gradient(Fy,dx,axis=1)-np.gradient(Fx,dx,axis=0)
    def __repr__(self): return "DifferentialOperators()"

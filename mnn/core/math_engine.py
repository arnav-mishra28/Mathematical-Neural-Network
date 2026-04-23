"""mnn.core.math_engine — Central mathematical engine for MNN."""
from __future__ import annotations
import numpy as np
import sympy as sp
from sympy import diff, integrate, simplify
from typing import Union, List, Dict, Optional, Tuple

class ScalarField:
    """Scalar-valued function f: Rⁿ → R."""
    def __init__(self, expr, variables: List[str]):
        self.expr      = expr
        self.variables = [sp.Symbol(v) for v in variables]
        self.var_names = variables
    def evaluate(self, point: Dict[str, float]) -> float:
        return float(self.expr.subs({sp.Symbol(k): v for k, v in point.items()}).evalf())
    def evaluate_grid(self, grids: Dict[str, np.ndarray]) -> np.ndarray:
        f = sp.lambdify(self.variables, self.expr, modules=["numpy"])
        return f(*[grids[v] for v in self.var_names])
    def gradient(self) -> "VectorField":
        return VectorField([diff(self.expr, v) for v in self.variables], self.var_names)
    def laplacian(self) -> "ScalarField":
        return ScalarField(simplify(sum(diff(self.expr, v, 2) for v in self.variables)), self.var_names)
    def to_latex(self) -> str: return sp.latex(self.expr)
    def __repr__(self): return f"ScalarField({self.expr}, vars={self.var_names})"

class VectorField:
    """Vector-valued function F: Rⁿ → Rᵐ."""
    def __init__(self, components: List, variables: List[str]):
        self.components = components
        self.variables  = [sp.Symbol(v) for v in variables]
        self.var_names  = variables
        self.dim        = len(components)
    def evaluate(self, point: Dict[str, float]) -> np.ndarray:
        s = {sp.Symbol(k): v for k, v in point.items()}
        return np.array([float(c.subs(s).evalf()) for c in self.components])
    def divergence(self) -> ScalarField:
        return ScalarField(simplify(sum(diff(c, v) for c, v in zip(self.components, self.variables))), self.var_names)
    def curl(self) -> "VectorField":
        if self.dim != 3: raise ValueError("Curl requires 3D")
        Fx,Fy,Fz = self.components; x,y,z = self.variables
        return VectorField([simplify(diff(Fz,y)-diff(Fy,z)),
                            simplify(diff(Fx,z)-diff(Fz,x)),
                            simplify(diff(Fy,x)-diff(Fx,y))], self.var_names)
    def jacobian(self) -> sp.Matrix:
        return sp.Matrix([[diff(c,v) for v in self.variables] for c in self.components])
    def to_latex(self) -> str:
        return r"\begin{pmatrix}" + r"\\".join(sp.latex(c) for c in self.components) + r"\end{pmatrix}"
    def __repr__(self): return f"VectorField(dim={self.dim}, vars={self.var_names})"

class TensorObject:
    """General tensor of rank (r,s)."""
    def __init__(self, data, rank=(2,0), variables=None, name="T"):
        self.name=name; self.rank=rank; self.variables=variables or []
        if isinstance(data, np.ndarray): self.data=data; self.symbolic=False
        elif isinstance(data, (list,tuple)):
            arr=np.array(data)
            if arr.dtype==object: self.data=sp.Array(data); self.symbolic=True
            else: self.data=arr; self.symbolic=False
        elif isinstance(data, sp.Array): self.data=data; self.symbolic=True
        else: self.data=data; self.symbolic=False
    @property
    def shape(self): return self.data.shape
    def to_numpy(self) -> np.ndarray:
        return np.array(self.data.tolist(), dtype=object) if self.symbolic else np.array(self.data)
    def symmetrize(self) -> "TensorObject":
        from itertools import permutations
        a=self.to_numpy().astype(float); axes=list(range(a.ndim))
        res=np.zeros_like(a,dtype=float); perms=list(permutations(axes))
        for p in perms: res+=np.transpose(a,p)
        return TensorObject(res/len(perms))
    def __repr__(self): return f"TensorObject(name={self.name}, rank={self.rank}, shape={self.shape})"

class MathEngine:
    """Central math engine — symbolic + numeric unified interface."""
    def __init__(self):
        self._syms: Dict[str, sp.Symbol] = {}
    def symbol(self, name: str, **kw) -> sp.Symbol:
        if name not in self._syms: self._syms[name]=sp.Symbol(name,**kw)
        return self._syms[name]
    def symbols(self, names: str):
        syms=sp.symbols(names)
        if isinstance(syms,sp.Symbol): syms=(syms,)
        for s in syms: self._syms[str(s)]=s
        return syms
    def _local(self, variables):
        d={v: sp.Symbol(v) for v in variables}
        d.update({"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,
                  "sqrt":sp.sqrt,"pi":sp.pi,"E":sp.E,"I":sp.I,
                  "sinh":sp.sinh,"cosh":sp.cosh,"tanh":sp.tanh,
                  "asin":sp.asin,"acos":sp.acos,"atan":sp.atan,"Abs":sp.Abs})
        return d
    def define_function(self, expr_str: str, variables: List[str]) -> ScalarField:
        return ScalarField(sp.sympify(expr_str, locals=self._local(variables)), variables)
    def define_vector_field(self, components: List[str], variables: List[str]) -> VectorField:
        loc=self._local(variables)
        return VectorField([sp.sympify(c,locals=loc) for c in components], variables)
    def define_tensor(self, data, rank=(2,0), variables=None, name="T") -> TensorObject:
        return TensorObject(data, rank=rank, variables=variables, name=name)
    def differentiate(self, field, variable: str, order: int=1) -> sp.Expr:
        expr=field.expr if isinstance(field,ScalarField) else field
        return diff(expr, sp.Symbol(variable), order)
    def partial(self, field: ScalarField, variable: str, order: int=1) -> ScalarField:
        return ScalarField(self.differentiate(field,variable,order), field.var_names)
    def gradient(self, field: ScalarField) -> VectorField: return field.gradient()
    def hessian(self, field: ScalarField) -> sp.Matrix: return sp.hessian(field.expr, field.variables)
    def integrate_symbolic(self, field, variable: str, limits=None) -> sp.Expr:
        expr=field.expr if isinstance(field,ScalarField) else field
        var=sp.Symbol(variable)
        return integrate(expr,(var,limits[0],limits[1])) if limits else integrate(expr,var)
    def integrate_numeric(self, field: ScalarField, variable: str, limits: Tuple, subs=None) -> float:
        from scipy.integrate import quad
        subs=subs or {}
        expr=field.expr.subs({sp.Symbol(k):v for k,v in subs.items()})
        f=sp.lambdify(sp.Symbol(variable), expr, modules=["numpy"])
        return quad(f, limits[0], limits[1])[0]
    def simplify(self, expr): return simplify(expr)
    def to_latex(self, expr) -> str:
        if isinstance(expr,(ScalarField,VectorField)): return expr.to_latex()
        return sp.latex(expr)
    def to_numpy_function(self, field: ScalarField):
        return sp.lambdify(field.variables, field.expr, modules=["numpy"])
    def taylor_series(self, field: ScalarField, variable: str, point=0, order=5) -> sp.Expr:
        return sp.series(field.expr, sp.Symbol(variable), point, order).removeO()
    def fourier_coefficients(self, field: ScalarField, variable: str, n_terms=5, period=None) -> Dict:
        period=period or 2*np.pi
        xs=np.linspace(0,period,1000)
        ys=sp.lambdify(sp.Symbol(variable), field.expr, modules=["numpy"])(xs)
        a0=(2/period)*np.trapz(ys,xs)
        a,b=[],[]
        for n in range(1,n_terms+1):
            a.append(float((2/period)*np.trapz(ys*np.cos(2*np.pi*n*xs/period),xs)))
            b.append(float((2/period)*np.trapz(ys*np.sin(2*np.pi*n*xs/period),xs)))
        return {"a0":float(a0),"a":a,"b":b}
    def summary(self) -> str: return f"MathEngine(symbols={list(self._syms.keys())})"
    def __repr__(self): return self.summary()

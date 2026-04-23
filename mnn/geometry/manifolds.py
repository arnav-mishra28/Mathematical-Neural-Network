"""mnn.geometry.manifolds — Riemannian manifolds for MNN."""
from __future__ import annotations
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
from typing import List, Dict, Optional, Tuple, Callable

class Manifold:
    def __init__(self, dim: int, name: str="M"): self.dim=dim; self.name=name
    def tangent_space_basis(self, point): return [np.eye(self.dim)[i] for i in range(self.dim)]
    def __repr__(self): return f"{self.__class__.__name__}(dim={self.dim}, name={self.name})"

class RiemannianManifold(Manifold):
    """Riemannian manifold with metric tensor, Christoffel symbols, Ricci curvature, geodesics."""
    def __init__(self, dim, metric_fn=None, coord_vars=None, name="M"):
        super().__init__(dim, name)
        self.coord_vars = coord_vars or [f"x{i}" for i in range(dim)]
        # Use plain Symbol (no real=True) so diff works consistently
        self._sym_vars  = [sp.Symbol(v) for v in self.coord_vars]
        self._metric_fn = metric_fn
        self._g_sym     = None

    @classmethod
    def sphere_S2(cls):
        """S² with round metric g = diag(1, sin²θ)."""
        theta = sp.Symbol("theta")
        phi   = sp.Symbol("phi")
        m = cls(2, coord_vars=["theta","phi"], name="S²")
        m._g_sym = sp.Matrix([[1, 0],[0, theta.subs(theta,theta)**0 * sp.sin(theta)**2]])
        # Build metric using the SAME symbol objects as _sym_vars
        t = m._sym_vars[0]   # theta
        m._g_sym = sp.Matrix([[sp.Integer(1), sp.Integer(0)],
                               [sp.Integer(0), sp.sin(t)**2]])
        return m

    @classmethod
    def hyperbolic_H2(cls):
        m = cls(2, coord_vars=["x","y"], name="H²")
        x, y = m._sym_vars
        d = (1 - x**2 - y**2)**2
        m._g_sym = sp.Matrix([[sp.Integer(4)/d, sp.Integer(0)],
                               [sp.Integer(0),   sp.Integer(4)/d]])
        return m

    @classmethod
    def torus_T2(cls, R=2.0, r=1.0):
        m = cls(2, coord_vars=["theta","phi"], name="T²")
        t = m._sym_vars[0]
        m._g_sym = sp.Matrix([[sp.Integer(r)**2 if isinstance(r,int) else sp.Float(r)**2,
                                sp.Integer(0)],
                               [sp.Integer(0),
                                (R + r*sp.cos(t))**2]])
        return m

    def metric_tensor(self, point=None):
        g = self._g_sym if self._g_sym is not None else sp.eye(self.dim)
        if point:
            return g.subs({sp.Symbol(k): v for k, v in point.items()})
        return g

    def inverse_metric(self): return self.metric_tensor().inv()

    def christoffel_symbols(self):
        g = self.metric_tensor(); g_inv = self.inverse_metric()
        n = self.dim; vs = self._sym_vars; G = {}
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    val = sp.Integer(0)
                    for l in range(n):
                        val += sp.Rational(1,2) * g_inv[k,l] * (
                            sp.diff(g[j,l], vs[i]) +
                            sp.diff(g[i,l], vs[j]) -
                            sp.diff(g[i,j], vs[l]))
                    G[(i,j,k)] = sp.simplify(val)
        return G

    def ricci_scalar(self):
        g = self.metric_tensor(); g_inv = self.inverse_metric()
        G = self.christoffel_symbols(); n = self.dim; vs = self._sym_vars
        # Full Riemann tensor including derivative terms
        RT = {}
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        t = (sp.diff(G.get((nu,sigma,rho), sp.Integer(0)), vs[mu])
                           - sp.diff(G.get((mu,sigma,rho), sp.Integer(0)), vs[nu]))
                        for lam in range(n):
                            t += (G.get((mu,lam,rho), sp.Integer(0)) * G.get((nu,sigma,lam), sp.Integer(0))
                                - G.get((nu,lam,rho), sp.Integer(0)) * G.get((mu,sigma,lam), sp.Integer(0)))
                        RT[(rho,sigma,mu,nu)] = t
        # Ricci tensor R_{mu,nu} = R^rho_{mu,rho,nu}
        Ric = sp.zeros(n, n)
        for mu in range(n):
            for nu in range(n):
                Ric[mu,nu] = sum(RT.get((r,mu,r,nu), sp.Integer(0)) for r in range(n))
        # Ricci scalar R = g^{mu,nu} R_{mu,nu}
        R = sum(g_inv[mu,nu] * Ric[mu,nu] for mu in range(n) for nu in range(n))
        return sp.simplify(R)

    def volume_element(self): return sp.sqrt(sp.Abs(self.metric_tensor().det()))

    def geodesic(self, start, velocity, t_span=(0,1), n_points=200):
        state0 = np.concatenate([start, velocity])
        G_flat = np.zeros((self.dim,self.dim,self.dim))
        def ode(t, s):
            x=s[:self.dim]; v=s[self.dim:]; dv=np.zeros(self.dim)
            for k in range(self.dim):
                for i in range(self.dim):
                    for j in range(self.dim): dv[k]-=G_flat[i,j,k]*v[i]*v[j]
            return np.concatenate([v, dv])
        sol = solve_ivp(ode, t_span, state0,
                        t_eval=np.linspace(*t_span,n_points), method="RK45", rtol=1e-8)
        return sol.y[:self.dim].T

    def __repr__(self): return f"RiemannianManifold(dim={self.dim},name={self.name})"

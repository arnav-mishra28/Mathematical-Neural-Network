"""mnn.algebra.abelian — Abelian group theory and Abelian function theory."""
from __future__ import annotations
import numpy as np, sympy as sp
from typing import List, Dict, Callable
from mnn.algebra.groups import Group

class AbelianGroup(Group):
    """Commutative group: a*b = b*a for all a, b."""
    def __init__(self, elements, operation, name="A"):
        super().__init__(elements, operation, name)
        if not self.is_abelian(): raise ValueError(f"{name} is not abelian.")
    @classmethod
    def cyclic(cls, n): return cls(list(range(n)), lambda a,b:(a+b)%n, f"Z_{n}")
    @classmethod
    def direct_product_cyclic(cls, orders):
        from itertools import product as iprod
        elems=list(iprod(*[range(n) for n in orders]))
        def op(a,b): return tuple((a[i]+b[i])%orders[i] for i in range(len(orders)))
        return cls(elems, op, "×".join(f"Z_{n}" for n in orders))
    def characters(self):
        if not all(isinstance(e,int) for e in self.elements): return []
        n=self.order
        return [lambda m,k=k: np.exp(2j*np.pi*k*m/n) for k in range(n)]
    def character_table(self):
        chis=self.characters(); t=np.zeros((len(chis),self.order),dtype=complex)
        for i,chi in enumerate(chis):
            for j,e in enumerate(self.elements): t[i,j]=chi(e)
        return t
    def dual_group(self): return AbelianGroup.cyclic(self.order)
    def fourier_transform(self, f: Dict):
        chis=self.characters()
        return {k: sum(f.get(e,0)*chi(e) for e in self.elements) for k,chi in enumerate(chis)}
    def inverse_fourier_transform(self, fhat: Dict):
        chis=self.characters()
        return {e: sum(fhat.get(k,0)*np.conj(chi(e)) for k,chi in enumerate(chis))/self.order
                for e in self.elements}
    def __repr__(self): return f"AbelianGroup(name={self.name},order={self.order})"

class AbelianFunction:
    """
    Abelian functions — multi-periodic meromorphic functions on Cⁿ.
    Jacobi theta functions θ₁–θ₄, Riemann theta, Weierstrass ℘.
    """
    def __init__(self, period_matrix=None, name="Θ"):
        self.Omega=np.array(period_matrix if period_matrix is not None else [[1j]],dtype=complex)
        self.n=self.Omega.shape[0]; self.name=name
    @staticmethod
    def theta1(z, tau, N=50):
        q=np.exp(1j*np.pi*tau)
        return 2*sum((-1)**n*q**((n+0.5)**2)*np.sin((2*n+1)*np.pi*z) for n in range(N))
    @staticmethod
    def theta2(z, tau, N=50):
        q=np.exp(1j*np.pi*tau)
        return 2*sum(q**((n+0.5)**2)*np.cos((2*n+1)*np.pi*z) for n in range(N))
    @staticmethod
    def theta3(z, tau, N=50):
        q=np.exp(1j*np.pi*tau)
        return 1+2*sum(q**(n**2)*np.cos(2*n*np.pi*z) for n in range(1,N+1))
    @staticmethod
    def theta4(z, tau, N=50):
        q=np.exp(1j*np.pi*tau)
        return 1+2*sum((-1)**n*q**(n**2)*np.cos(2*n*np.pi*z) for n in range(1,N+1))
    def riemann_theta(self, z, N=5):
        from itertools import product as iprod
        z=np.array(z,dtype=complex); val=0
        for nv in iprod(range(-N,N+1),repeat=self.n):
            n=np.array(nv,dtype=complex)
            val+=np.exp(1j*np.pi*(n@self.Omega@n)+2j*np.pi*(n@z))
        return val
    @staticmethod
    def weierstrass_p(z, N=20):
        tau=1j; t3=AbelianFunction.theta3(0,tau); t1z=AbelianFunction.theta1(z/np.pi,tau)
        t3z=AbelianFunction.theta3(z/np.pi,tau)
        if abs(t1z)<1e-15: return complex(np.inf)
        return (np.pi*t3)**2*(t3z/t1z)**2/np.pi**2-(np.pi*t3)**4/9.0
    def period_lattice_points(self, n=3):
        omega2=complex(self.Omega[0,0])
        return np.array([m+k*omega2 for m in range(-n,n+1) for k in range(-n,n+1)])
    def __repr__(self): return f"AbelianFunction(genus={self.n},name={self.name})"

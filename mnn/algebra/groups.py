"""mnn.algebra.groups — Group theory engine for MNN."""
from __future__ import annotations
import numpy as np, sympy as sp
from itertools import product as iprod, permutations
from typing import List, Dict, Callable, Set, Tuple

class Group:
    """Finite group (G, *). Supports subgroups, center, conjugacy classes, products."""
    def __init__(self, elements, operation: Callable, name="G"):
        self.elements=list(elements); self.operation=operation; self.name=name; self.order=len(elements); self._table=None
    @classmethod
    def cyclic(cls, n): return cls(list(range(n)), lambda a,b:(a+b)%n, f"Z_{n}")
    @classmethod
    def symmetric(cls, n):
        elems=list(permutations(range(n)))
        return cls(elems, lambda p,q:tuple(p[q[i]] for i in range(n)), f"S_{n}")
    @classmethod
    def dihedral(cls, n):
        elems=[(r,s) for s in range(2) for r in range(n)]
        def op(a,b):
            r1,s1=a; r2,s2=b
            return ((r1+r2)%n,s2) if s1==0 else ((r1-r2)%n,(s1+s2)%2)
        return cls(elems, op, f"D_{n}")
    @classmethod
    def alternating(cls, n):
        def sgn(p): return 1 if sum(1 for i in range(len(p)) for j in range(i+1,len(p)) if p[i]>p[j])%2==0 else -1
        elems=[p for p in permutations(range(n)) if sgn(p)==1]
        return cls(elems, lambda p,q:tuple(p[q[i]] for i in range(n)), f"A_{n}")
    def cayley_table(self):
        if self._table is not None: return self._table
        idx={e:i for i,e in enumerate(self.elements)}
        self._table=np.array([[idx[self.operation(a,b)] for b in self.elements] for a in self.elements]); return self._table
    def identity(self):
        for e in self.elements:
            if all(self.operation(e,a)==a for a in self.elements): return e
        raise ValueError("No identity")
    def inverse(self, a):
        e=self.identity()
        for b in self.elements:
            if self.operation(a,b)==e: return b
        raise ValueError(f"No inverse for {a}")
    def element_order(self, a):
        e=self.identity(); cur=a
        for n in range(1,self.order+1):
            if cur==e: return n
            cur=self.operation(cur,a)
        raise ValueError(f"Infinite order for {a}")
    def is_abelian(self): return all(self.operation(a,b)==self.operation(b,a) for a in self.elements for b in self.elements)
    def center(self): return [z for z in self.elements if all(self.operation(z,a)==self.operation(a,z) for a in self.elements)]
    def conjugacy_classes(self):
        idx={e:i for i,e in enumerate(self.elements)}; rem=set(range(self.order)); classes=[]
        while rem:
            i=next(iter(rem)); a=self.elements[i]; cls_=set()
            for g in self.elements:
                conj=self.operation(self.operation(g,a),self.inverse(g)); cls_.add(idx[conj])
            classes.append({self.elements[j] for j in cls_}); rem-=cls_
        return classes
    def _is_subgroup(self, subset):
        e=self.identity()
        if e not in subset: return False
        return all(self.inverse(a) in subset and self.operation(a,b) in subset for a in subset for b in subset)
    def subgroups(self):
        from itertools import combinations
        e=self.identity(); subs=[]
        for r in range(1,self.order+1):
            if self.order%r!=0: continue
            rest=[x for x in self.elements if x!=e]
            for combo in combinations(rest,r-1):
                sub=set(combo)|{e}
                if self._is_subgroup(sub): subs.append(Group(list(sub),self.operation,f"Sub_{r}"))
        return subs
    def normal_subgroups(self):
        result=[]
        for sub in self.subgroups():
            N=set(sub.elements)
            if all({self.operation(self.operation(g,n),self.inverse(g)) for n in N}==N for g in self.elements):
                result.append(sub)
        return result
    def is_simple(self): return len(self.normal_subgroups())==2
    def direct_product(self, other):
        elems=list(iprod(self.elements,other.elements))
        return Group(elems, lambda a,b:(self.operation(a[0],b[0]),other.operation(a[1],b[1])), f"{self.name}×{other.name}")
    def summary(self):
        return f"Group: {self.name}\n  Order={self.order}\n  Abelian={self.is_abelian()}\n  Center={self.center()}"
    def __repr__(self): return f"Group(name={self.name},order={self.order},abelian={self.is_abelian()})"

class LieGroup:
    """Continuous Lie group with associated Lie algebra."""
    def __init__(self, name, n): self.name=name; self.n=n
    @classmethod
    def SO(cls,n): return cls(f"SO({n})",n)
    @classmethod
    def SU(cls,n): return cls(f"SU({n})",n)
    @classmethod
    def GL(cls,n): return cls(f"GL({n})",n)
    @classmethod
    def SL(cls,n): return cls(f"SL({n})",n)
    def lie_algebra_basis(self):
        n=self.n
        if "SO" in self.name:
            basis=[]
            for i in range(n):
                for j in range(i+1,n):
                    E=np.zeros((n,n)); E[i,j]=1; E[j,i]=-1; basis.append(E)
            return basis
        elif "SU" in self.name and n==2:
            return [np.array([[0,1j],[1j,0]]),np.array([[0,1],[-1,0]]),np.array([[1j,0],[0,-1j]])]
        else:
            return [np.eye(n,k=0)*0+np.eye(n,M=n)[i]*np.eye(n,M=n)[j].reshape(-1,1) for i in range(n) for j in range(n)]
    def lie_bracket(self, X, Y): return X@Y-Y@X
    def structure_constants(self):
        basis=self.lie_algebra_basis(); d=len(basis); f=np.zeros((d,d,d))
        for i,ei in enumerate(basis):
            for j,ej in enumerate(basis):
                br=self.lie_bracket(ei,ej)
                for k,ek in enumerate(basis):
                    denom=np.real(np.trace(ek@ek.conj().T))+1e-15
                    f[i,j,k]=np.real(np.trace(br@ek.conj().T))/denom
        return f
    def exponential_map(self, X):
        from scipy.linalg import expm; return expm(X)
    def logarithmic_map(self, g):
        from scipy.linalg import logm; return logm(g)
    def adjoint_representation(self, X):
        basis=self.lie_algebra_basis(); d=len(basis); ad=np.zeros((d,d))
        for j,ej in enumerate(basis):
            br=self.lie_bracket(X,ej)
            for i,ei in enumerate(basis):
                ad[i,j]=np.real(np.trace(br@ei.conj().T))/(np.real(np.trace(ei@ei.conj().T))+1e-15)
        return ad
    def killing_form(self):
        basis=self.lie_algebra_basis(); d=len(basis); K=np.zeros((d,d))
        for i,ei in enumerate(basis):
            adi=self.adjoint_representation(ei)
            for j,ej in enumerate(basis):
                K[i,j]=np.real(np.trace(adi@self.adjoint_representation(ej)))
        return K
    def is_semisimple(self): return abs(np.linalg.det(self.killing_form()))>1e-10
    def geodesic(self, g0, X, t_max=1.0, n=100):
        from scipy.linalg import expm
        return [g0@expm(t*X) for t in np.linspace(0,t_max,n)]
    def __repr__(self): return f"LieGroup({self.name},dim_alg={len(self.lie_algebra_basis())})"

class SymmetryGroup:
    @staticmethod
    def rotation_matrix_2d(theta):
        return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    @staticmethod
    def rotation_matrix_3d(axis, theta):
        axis=axis/np.linalg.norm(axis); ux,uy,uz=axis; c,s=np.cos(theta),np.sin(theta)
        return np.array([[c+ux**2*(1-c),ux*uy*(1-c)-uz*s,ux*uz*(1-c)+uy*s],
                         [uy*ux*(1-c)+uz*s,c+uy**2*(1-c),uy*uz*(1-c)-ux*s],
                         [uz*ux*(1-c)-uy*s,uz*uy*(1-c)+ux*s,c+uz**2*(1-c)]])
    @staticmethod
    def reflection_matrix(normal):
        n=normal/np.linalg.norm(normal); return np.eye(len(n))-2*np.outer(n,n)

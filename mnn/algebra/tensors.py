"""mnn.algebra.tensors — Full tensor calculus engine for MNN."""
from __future__ import annotations
import numpy as np, sympy as sp
from itertools import permutations
from typing import Tuple
from mnn.core.math_engine import TensorObject

class TensorEngine:
    """Tensor calculus: einsum, Riemann/Ricci/Einstein/Weyl tensors, Hodge dual, differential forms."""
    @staticmethod
    def zeros(shape, symbolic=False):
        return TensorObject(sp.zeros(*shape) if (symbolic and len(shape)==2) else np.zeros(shape))
    @staticmethod
    def eye(n): return TensorObject(np.eye(n))
    @staticmethod
    def levi_civita(n):
        eps=np.zeros([n]*n)
        for perm in permutations(range(n)):
            inv=sum(1 for i in range(n) for j in range(i+1,n) if perm[i]>perm[j])
            eps[perm]=(-1)**inv
        return TensorObject(eps,name="ε")
    @staticmethod
    def metric_flat(n, signature="euclidean"):
        if signature=="euclidean": return TensorObject(np.eye(n),rank=(0,2),name="g")
        if signature=="minkowski": return TensorObject(np.diag([-1]+[1]*(n-1)),rank=(0,2),name="η")
        raise ValueError(f"Unknown signature: {signature}")
    @staticmethod
    def einsum(subscripts, *tensors):
        return TensorObject(np.einsum(subscripts,*[t.to_numpy().astype(float) for t in tensors]))
    @staticmethod
    def outer_product(A, B):
        return TensorObject(np.tensordot(A.to_numpy().astype(float),B.to_numpy().astype(float),axes=0),name=f"{A.name}⊗{B.name}")
    @staticmethod
    def contract(A, B, axes):
        return TensorObject(np.tensordot(A.to_numpy().astype(float),B.to_numpy().astype(float),axes=axes))
    @staticmethod
    def trace(A, axis1=0, axis2=1):
        return TensorObject(np.trace(A.to_numpy().astype(float),axis1=axis1,axis2=axis2))
    @staticmethod
    def symmetrize(A): return A.symmetrize()
    @staticmethod
    def antisymmetrize(A):
        a=A.to_numpy().astype(float); ndim=a.ndim; axes=list(range(ndim))
        res=np.zeros_like(a,dtype=float); perms=list(permutations(axes))
        for perm in perms:
            inv=sum(1 for i in range(ndim) for j in range(i+1,ndim) if perm[i]>perm[j])
            res+=(-1)**inv*np.transpose(a,perm)
        return TensorObject(res/len(perms))
    @staticmethod
    def raise_index(T, metric_inv, index=0):
        g=metric_inv.to_numpy().astype(float); t=T.to_numpy().astype(float)
        r=np.tensordot(g,t,axes=[[1],[index]]); axes=list(range(r.ndim)); axes.insert(index,axes.pop(0))
        return TensorObject(np.transpose(r,axes))
    @staticmethod
    def lower_index(T, metric, index=0):
        g=metric.to_numpy().astype(float); t=T.to_numpy().astype(float)
        r=np.tensordot(g,t,axes=[[1],[index]]); axes=list(range(r.ndim)); axes.insert(index,axes.pop(0))
        return TensorObject(np.transpose(r,axes))
    @staticmethod
    def christoffel_numeric(metric_fn, coords, h=1e-5):
        n=len(coords); g=metric_fn(coords); g_inv=np.linalg.inv(g); G=np.zeros((n,n,n))
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    s=0
                    for l in range(n):
                        def dg(a,b,mu):
                            cp=coords.copy(); cp[mu]+=h; cm=coords.copy(); cm[mu]-=h
                            return (metric_fn(cp)[a,b]-metric_fn(cm)[a,b])/(2*h)
                        s+=g_inv[k,l]*(dg(j,l,i)+dg(i,l,j)-dg(i,j,l))
                    G[i,j,k]=0.5*s
        return G
    @staticmethod
    def riemann_tensor(G):
        n=G.shape[0]; R=np.zeros((n,n,n,n))
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        r=sum(G[mu,l,rho]*G[nu,sigma,l]-G[nu,l,rho]*G[mu,sigma,l] for l in range(n))
                        R[rho,sigma,mu,nu]=r
        return R
    @staticmethod
    def ricci_tensor(riemann):
        n=riemann.shape[0]
        return np.array([[sum(riemann[r,mu,r,nu] for r in range(n)) for nu in range(n)] for mu in range(n)])
    @staticmethod
    def ricci_scalar(ricci, g_inv): return float(np.einsum('ij,ij->',g_inv,ricci))
    @staticmethod
    def einstein_tensor(ricci, R_scalar, metric): return ricci-0.5*R_scalar*metric
    @staticmethod
    def weyl_tensor(riemann, ricci, R_scalar, metric):
        n=metric.shape[0]
        if n<3: raise ValueError("Weyl requires n≥3")
        C=riemann.copy()
        for r in range(n):
            for s in range(n):
                for m in range(n):
                    for v in range(n):
                        C[r,s,m,v]-=(2/(n-2))*(metric[r,m]*ricci[s,v]-metric[r,v]*ricci[s,m]-metric[s,m]*ricci[r,v]+metric[s,v]*ricci[r,m])
                        C[r,s,m,v]+=(R_scalar/((n-1)*(n-2)))*(metric[r,m]*metric[s,v]-metric[r,v]*metric[s,m])
        return C
    @staticmethod
    def wedge_product(omega, eta):
        if omega.ndim==1 and eta.ndim==1: return np.outer(omega,eta)-np.outer(eta,omega)
        raise NotImplementedError("Use antisymmetrize for higher-rank forms.")
    @staticmethod
    def hodge_dual(form, metric):
        n=metric.shape[0]; sqrt_g=np.sqrt(abs(np.linalg.det(metric)))
        eps=np.zeros([n]*n)
        for perm in permutations(range(n)):
            inv=sum(1 for i in range(n) for j in range(i+1,n) if perm[i]>perm[j])
            eps[perm]=(-1)**inv*sqrt_g
        if form.ndim==1 and n==3:
            g_inv=np.linalg.inv(metric); w=g_inv@form; dual=np.zeros((n,n))
            for i in range(n):
                for j in range(n): dual[i,j]=sum(eps[i,j,k]*w[k] for k in range(n))
            return dual
        return eps
    def __repr__(self): return "TensorEngine()"

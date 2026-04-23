"""mnn.geometry.hyperspheres — Hypersphere S^(n-1) in Rⁿ."""
from __future__ import annotations
import numpy as np
from typing import Optional

class Hypersphere:
    """S^{n-1} = {x ∈ Rⁿ : ‖x − c‖ = r}"""
    def __init__(self, dim, center=None, radius=1.0):
        self.dim=dim; self.sphere_dim=dim-1
        self.center=np.array(center,dtype=float) if center is not None else np.zeros(dim)
        self.radius=float(radius)
    def surface_area(self):
        from scipy.special import gamma
        return 2*np.pi**(self.dim/2)*self.radius**(self.dim-1)/gamma(self.dim/2)
    def volume(self):
        from scipy.special import gamma
        return np.pi**(self.dim/2)*self.radius**self.dim/gamma(self.dim/2+1)
    def contains(self, pt, tol=1e-9): return abs(np.linalg.norm(pt-self.center)-self.radius)<tol
    def distance_to_surface(self, pt): return abs(np.linalg.norm(pt-self.center)-self.radius)
    def project_to_surface(self, pt):
        v=np.array(pt,dtype=float)-self.center; n=np.linalg.norm(v)
        if n<1e-15: v=np.eye(self.dim)[0]; n=1.0
        return self.center+self.radius*v/n
    def sample_surface(self, n=100):
        x=np.random.randn(n,self.dim); return self.center+self.radius*x/np.linalg.norm(x,axis=1,keepdims=True)
    def sample_interior(self, n=100):
        s=self.sample_surface(n); r=np.random.uniform(0,1,(n,1))**(1/self.dim)
        return self.center+r*(s-self.center)
    def geodesic_distance(self, p1, p2):
        a=self.project_to_surface(p1)-self.center; b=self.project_to_surface(p2)-self.center
        return float(self.radius*np.arccos(np.clip(np.dot(a,b)/self.radius**2,-1,1)))
    def geodesic_path(self, p1, p2, n=100):
        a=self.project_to_surface(p1)-self.center; b=self.project_to_surface(p2)-self.center
        dot=np.clip(np.dot(a,b)/self.radius**2,-1+1e-12,1-1e-12); omega=np.arccos(dot)
        t=np.linspace(0,1,n)
        if abs(omega)<1e-10: return np.tile(a+self.center,(n,1))
        return (np.sin((1-t[:,None])*omega)*a+np.sin(t[:,None]*omega)*b)/np.sin(omega)+self.center
    def tangent_space_basis(self, pt):
        p=self.project_to_surface(pt); n=(p-self.center)/self.radius
        _,_,Vt=np.linalg.svd(n.reshape(1,-1)); return Vt[1:]
    def exponential_map(self, base, tangent):
        p=self.project_to_surface(base)-self.center; v=np.array(tangent,dtype=float); nv=np.linalg.norm(v)
        if nv<1e-15: return base
        return p*np.cos(nv/self.radius)+v/nv*np.sin(nv/self.radius)*self.radius+self.center
    def logarithmic_map(self, base, target):
        p=self.project_to_surface(base)-self.center; q=self.project_to_surface(target)-self.center
        dot=np.clip(np.dot(p,q)/self.radius**2,-1+1e-12,1-1e-12)
        d=self.radius*np.arccos(dot); v=q/self.radius-dot*p/self.radius; nv=np.linalg.norm(v)
        return np.zeros_like(p) if nv<1e-15 else d*v/nv
    def __repr__(self): return f"Hypersphere(dim={self.dim},r={self.radius},c={self.center})"

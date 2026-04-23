"""mnn.geometry.hyperplanes — Hyperplane geometry in Rⁿ."""
from __future__ import annotations
import numpy as np
from typing import Optional

class Hyperplane:
    """n·x = d  in R^n."""
    def __init__(self, normal, offset=0.0):
        n=np.array(normal,dtype=float); self.normal=n/np.linalg.norm(n); self.offset=float(offset); self.dim=len(normal)
    @classmethod
    def from_points(cls, points):
        pts=np.array(points,dtype=float); c=pts.mean(axis=0)
        _,_,Vt=np.linalg.svd(pts-c); return cls(Vt[-1],float(Vt[-1]@c))
    @classmethod
    def from_equation(cls, coeffs, rhs=0.0): return cls(np.array(coeffs),rhs)
    def signed_distance(self, pt): return float(self.normal@np.array(pt)-self.offset)
    def distance(self, pt): return abs(self.signed_distance(pt))
    def project(self, pt): p=np.array(pt,dtype=float); return p-self.signed_distance(p)*self.normal
    def reflect(self, pt): p=np.array(pt,dtype=float); return p-2*self.signed_distance(p)*self.normal
    def contains(self, pt, tol=1e-9): return abs(self.signed_distance(pt))<tol
    def intersect_line(self, lp, ld):
        p=np.array(lp,dtype=float); d=np.array(ld,dtype=float); denom=self.normal@d
        if abs(denom)<1e-12: return None
        return p+(self.offset-self.normal@p)/denom*d
    def positive_half_space(self, pt): return self.signed_distance(pt)>0
    def sample_on_hyperplane(self, n=100, center=None, radius=1.0):
        c=self.project(center if center is not None else np.zeros(self.dim))
        _,_,Vt=np.linalg.svd(self.normal.reshape(1,-1))
        return c+np.random.randn(n,self.dim-1)*radius@Vt[1:]
    def __repr__(self): return f"Hyperplane(dim={self.dim},offset={self.offset:.4f})"

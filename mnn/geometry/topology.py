"""mnn.geometry.topology — Topological spaces and simplicial complexes."""
from __future__ import annotations
import numpy as np, networkx as nx
from itertools import combinations
from typing import List, Dict, Set, Tuple

class TopologicalSpace:
    def __init__(self, points, name="X"):
        self.points=np.array(points); self.name=name; self.n=len(points)
        self.dim_ambient=points.shape[1] if len(np.array(points).shape)>1 else 1
    def epsilon_neighborhood(self, i, eps):
        return list(np.where(np.linalg.norm(self.points-self.points[i],axis=1)<eps)[0])
    def connected_components(self, eps):
        G=nx.Graph(); G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in self.epsilon_neighborhood(i,eps):
                if i!=j: G.add_edge(i,j)
        return [list(c) for c in nx.connected_components(G)]
    def betti_numbers_approx(self, eps):
        G=nx.Graph(); G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                if np.linalg.norm(self.points[i]-self.points[j])<eps: G.add_edge(i,j)
        b0=nx.number_connected_components(G)
        b1=max(0,G.number_of_edges()-G.number_of_nodes()+b0)
        return {"beta_0":b0,"beta_1":b1}
    def hausdorff_dimension_estimate(self, n_scales=10):
        mn=self.points.min(axis=0); mx=self.points.max(axis=0)
        scales=np.logspace(-2,0,n_scales)*(mx-mn).max()
        counts=[]
        for eps in scales:
            boxes=set()
            for p in self.points: boxes.add(tuple(int((p[d]-mn[d])/eps) for d in range(self.dim_ambient)))
            counts.append(len(boxes))
        s=scales[np.array(counts)>0]; c=np.array(counts)[np.array(counts)>0]
        if len(s)<2: return float(self.dim_ambient)
        slope,_=np.polyfit(np.log(1/s),np.log(c),1); return float(slope)
    def __repr__(self): return f"TopologicalSpace(n={self.n},dim={self.dim_ambient},name={self.name})"

class SimplicialComplex:
    def __init__(self): self.simplices: Dict[int,List[Tuple]]={}
    def add_simplex(self, simplex):
        s=tuple(sorted(simplex)); d=len(s)-1
        self.simplices.setdefault(d,[])
        if s not in self.simplices[d]: self.simplices[d].append(s)
        if d>0:
            for face in combinations(s,d): self.add_simplex(face)
    def add_simplices(self, simplices):
        for s in simplices: self.add_simplex(s)
    def boundary_matrix(self, dim):
        if dim not in self.simplices or (dim-1) not in self.simplices:
            return np.zeros((0,0),dtype=int)
        chains=self.simplices[dim]; bds=self.simplices[dim-1]
        mat=np.zeros((len(bds),len(chains)),dtype=int)
        for j,s in enumerate(chains):
            for i_f in range(len(s)):
                face=tuple(s[k] for k in range(len(s)) if k!=i_f)
                if face in bds: mat[bds.index(face),j]=1
        return mat%2
    def betti_number(self, dim):
        def rank_mod2(M):
            if M.size==0: return 0
            M=M.copy()%2; r=0
            for col in range(M.shape[1]):
                piv=next((row for row in range(r,M.shape[0]) if M[row,col]==1),None)
                if piv is None: continue
                M[[r,piv]]=M[[piv,r]]
                for row in range(M.shape[0]):
                    if row!=r and M[row,col]==1: M[row]=(M[row]+M[r])%2
                r+=1
            return r
        nc=len(self.simplices.get(dim,[])); ker=nc-rank_mod2(self.boundary_matrix(dim))
        im=rank_mod2(self.boundary_matrix(dim+1)) if (dim+1) in self.simplices else 0
        return max(0,ker-im)
    def euler_characteristic(self):
        return sum((-1)**d*len(s) for d,s in self.simplices.items())
    def all_betti_numbers(self):
        return {d:self.betti_number(d) for d in sorted(self.simplices.keys())}
    def __repr__(self):
        return f"SimplicialComplex({{{', '.join(f'd{d}:{len(s)}' for d,s in self.simplices.items())}}})"

"""Example 03 — Geometry & Topology"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.geometry.manifolds    import RiemannianManifold
from mnn.geometry.topology     import TopologicalSpace, SimplicialComplex
from mnn.geometry.hyperplanes  import Hyperplane
from mnn.geometry.hyperspheres import Hypersphere
S2  = RiemannianManifold.sphere_S2()
print(f"S² Ricci scalar = {S2.ricci_scalar()}")
S5  = Hypersphere(6,radius=3.)
pts = S5.sample_surface(200)
print(f"S⁵ surface area = {S5.surface_area():.4f}")
print(f"Sample norms mean = {np.linalg.norm(pts,axis=1).mean():.6f}")
H   = Hyperplane([1,2,-1,0.5],3.)
pt  = np.array([4.,2.,1.,0.])
print(f"Hyperplane dist = {H.distance(pt):.4f}")
sc  = SimplicialComplex()
sc.add_simplices([(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
print(f"Tetrahedron χ={sc.euler_characteristic()}, β={sc.all_betti_numbers()}")
th  = np.linspace(0,2*np.pi,200)
pts2= np.column_stack([np.cos(th),np.sin(th)])+np.random.randn(200,2)*0.05
b   = TopologicalSpace(pts2).betti_numbers_approx(0.3)
print(f"S¹ β₀={b['beta_0']}, β₁={b['beta_1']}")
print("[OK] Example 03")

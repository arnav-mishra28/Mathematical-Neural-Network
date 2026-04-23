"""Tests for mnn — test_geometry"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.geometry.manifolds    import RiemannianManifold
from mnn.geometry.topology     import TopologicalSpace, SimplicialComplex
from mnn.geometry.hyperplanes  import Hyperplane
from mnn.geometry.hyperspheres import Hypersphere
def test_s2_metric():
    S2=RiemannianManifold.sphere_S2(); assert S2.metric_tensor().shape==(2,2)
def test_s2_ricci():
    import sympy as sp
    S2=RiemannianManifold.sphere_S2(); R=S2.ricci_scalar(); assert sp.simplify(R-2)==0
def test_hypersphere_norm():
    S=Hypersphere(4,radius=2.); pts=S.sample_surface(200)
    assert np.allclose(np.linalg.norm(pts,axis=1),2.,atol=1e-9)
def test_geodesic_dist_antipodal():
    S=Hypersphere(3,radius=1.)
    p1=np.array([1.,0.,0.]); p2=np.array([-1.,0.,0.])
    assert abs(S.geodesic_distance(p1,p2)-np.pi)<1e-6
def test_hyperplane_dist():
    H=Hyperplane([0,0,1],0.); assert abs(H.distance(np.array([0.,0.,3.]))-3.)<1e-9
def test_hyperplane_proj():
    H=Hyperplane([0,0,1],0.); p=H.project(np.array([1.,2.,5.])); assert abs(p[2])<1e-9
def test_hyperplane_from_pts():
    pts=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float); H=Hyperplane.from_points(pts)
    for p in pts: assert H.contains(p,1e-8)
def test_euler_char():
    sc=SimplicialComplex(); sc.add_simplices([(0,1),(1,2),(2,0)]); assert sc.euler_characteristic()==0
def test_betti_circle():
    th=np.linspace(0,2*np.pi,100); pts=np.column_stack([np.cos(th),np.sin(th)])
    b=TopologicalSpace(pts).betti_numbers_approx(0.3); assert b["beta_0"]==1
if __name__=="__main__":
    for fn in [test_s2_metric,test_s2_ricci,test_hypersphere_norm,test_geodesic_dist_antipodal,
               test_hyperplane_dist,test_hyperplane_proj,test_hyperplane_from_pts,
               test_euler_char,test_betti_circle]:
        fn(); print(f"  ✓ {fn.__name__}")
    print("[ALL GEOMETRY TESTS PASSED]")

"""Tests for mnn — test_chaos"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.chaos.attractors import LorenzAttractor, RosslerAttractor
from mnn.chaos.dynamics   import NonlinearSystem, BifurcationAnalyzer
from mnn.chaos.analysis   import ChaosAnalyzer
def test_lorenz_shape():
    t=LorenzAttractor().integrate(t_span=(0,5),dt=0.01); assert t.shape[1]==3
def test_lorenz_fps():
    fps=LorenzAttractor().fixed_points(); assert len(fps)==3
def test_lorenz_chaotic():
    t=LorenzAttractor().integrate(t_span=(0,20),dt=0.01); assert np.std(t[:,0])>5.
def test_rossler():    assert RosslerAttractor().integrate(t_span=(0,5),dt=0.01).shape[1]==3
def test_bifurcation():
    r,x=BifurcationAnalyzer.logistic_map_bifurcation(n_r=100,n_iter=100)
    assert len(r)>0 and np.all(x>=0) and np.all(x<=1)
def test_poincare():
    lor=LorenzAttractor(); s=NonlinearSystem(lor.ode,3)
    sec=s.poincare_section(lor._default_initial(),(0,20),2,27.,0.005)
    assert isinstance(sec,np.ndarray)
def test_corr_dim():
    t=LorenzAttractor().integrate(t_span=(0,15),dt=0.01)
    d,_,_=ChaosAnalyzer.correlation_dimension(t[::10][:300]); assert d>1.
def test_perm_entropy():
    t=LorenzAttractor().integrate(t_span=(0,40),dt=0.01)
    pe=ChaosAnalyzer.permutation_entropy(t[:,0][:2000],order=4); assert pe>0.5
def test_zero_one():
    t=LorenzAttractor().integrate(t_span=(0,20),dt=0.01)
    K=ChaosAnalyzer.zero_one_test(t[:,0][:400],n_samples=10); assert K>0.5
def test_kaplan_yorke():
    s=np.array([0.9,0.,-14.5]); d=ChaosAnalyzer.kaplan_yorke_dimension(s); assert 2.<d<3.
if __name__=="__main__":
    for fn in [test_lorenz_shape,test_lorenz_fps,test_lorenz_chaotic,test_rossler,
               test_bifurcation,test_poincare,test_corr_dim,test_perm_entropy,
               test_zero_one,test_kaplan_yorke]:
        fn(); print(f"  ✓ {fn.__name__}")
    print("[ALL CHAOS TESTS PASSED]")

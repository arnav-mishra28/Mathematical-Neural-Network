"""Example 06 — Chaos Theory"""
import sys; sys.path.insert(0,"..")
import numpy as np
from mnn.chaos.attractors import LorenzAttractor, RosslerAttractor, ChenAttractor
from mnn.chaos.dynamics   import NonlinearSystem, BifurcationAnalyzer
from mnn.chaos.analysis   import ChaosAnalyzer
lor  = LorenzAttractor()
traj = lor.integrate(t_span=(0,50),dt=0.01)
print(f"Lorenz traj: {traj.shape} | fps: {len(lor.fixed_points())}")
spec = ChaosAnalyzer.lyapunov_spectrum(lor.ode,lor._default_initial(),t_end=15,dt=0.2,renorm_steps=5)
print(f"Lyapunov spectrum: {spec.round(3)}")
print(f"Kaplan-Yorke dim: {ChaosAnalyzer.kaplan_yorke_dimension(spec):.4f}")
dim,_,_ = ChaosAnalyzer.correlation_dimension(traj[::10][:400])
print(f"Correlation dim: {dim:.3f}")
K = ChaosAnalyzer.zero_one_test(traj[:,0][:500],n_samples=15)
print(f"0-1 test K = {K:.4f}  (→1 = chaotic)")
pe = ChaosAnalyzer.permutation_entropy(traj[:,0][:2000],order=4)
print(f"Permutation entropy = {pe:.4f}")
r_arr,x_arr = BifurcationAnalyzer.logistic_map_bifurcation(n_r=200,n_iter=200)
print(f"Logistic bifurcation: {len(r_arr)} points")
sys_l = NonlinearSystem(lor.ode,3)
sec   = sys_l.poincare_section(lor._default_initial(),(0,30),2,27.,0.005)
print(f"Poincaré section: {len(sec)} points")
print("[OK] Example 06")

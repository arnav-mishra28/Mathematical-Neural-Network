"""Example 07 — Full MNN Pipeline: Heat Equation → PDE-constrained MNN → Geometry → Chaos"""
import sys; sys.path.insert(0,"..")
import numpy as np
import sympy as sp
from mnn.core.math_engine        import MathEngine
from mnn.neural.base_network     import MNNNetwork
from mnn.neural.training         import MNNTrainer
from mnn.neural.constraint_layer import PDEConstraint
from mnn.geometry.topology       import TopologicalSpace
from mnn.algebra.tensors         import TensorEngine
from mnn.chaos.attractors        import LorenzAttractor
from mnn.chaos.analysis          import ChaosAnalyzer

print("=== STEP 1: Symbolic setup ===")
eng = MathEngine()
u   = eng.define_function("sin(pi*x)*exp(-pi**2*t)",["x","t"])
ut  = eng.differentiate(u,"t")
uxx = eng.differentiate(u,"x",2)
print(f"Heat eq residual (symbolic): {sp.simplify(ut-uxx)}")

print("\n=== STEP 2: PDE-constrained MNN ===")
net    = MNNNetwork(2,1,64,4)
pde_fn = PDEConstraint.heat_1d(net,1.0)
xt_col = np.random.uniform(0,1,(600,2)).astype(np.float32)
xs_ic  = np.column_stack([np.linspace(0,1,100),np.zeros(100)]).astype(np.float32)
ys_ic  = np.sin(np.pi*xs_ic[:,0:1]).astype(np.float32)
tr = MNNTrainer(net,1e-3)
tk = tr.train_constrained(xt_col,pde_fn,x_data=xs_ic,y_data=ys_ic,w_pde=1.,w_data=5.,n_epochs=500,verbose=False)
print(f"Losses: {tk.latest()}")
xt_test = np.column_stack([np.linspace(0,1,50),np.full(50,0.1)]).astype(np.float32)
exact   = np.sin(np.pi*xt_test[:,0])*np.exp(-np.pi**2*0.1)
mse     = float(np.mean((tr.evaluate(xt_test).flatten()-exact)**2))
print(f"MSE vs exact at t=0.1: {mse:.6f}")

print("\n=== STEP 3: Topological analysis ===")
xt_g  = np.array([[x,t] for x in np.linspace(0,1,15) for t in np.linspace(0,1,15)],dtype=np.float32)
u_g   = tr.evaluate(xt_g).flatten()
T_sp  = TopologicalSpace(np.column_stack([xt_g,u_g.reshape(-1,1)])[:100])
print(f"Betti: {T_sp.betti_numbers_approx(0.3)}")

print("\n=== STEP 4: Tensor + Chaos ===")
eps = TensorEngine.levi_civita(3).to_numpy()
print(f"ε₀₁₂={eps[0,1,2]:.0f}")
lor = LorenzAttractor()
traj = lor.integrate(t_span=(0,20),dt=0.01)
K = ChaosAnalyzer.zero_one_test(traj[:,0][:400],n_samples=10)
print(f"Lorenz 0-1 test K={K:.4f}")
print("\n✓ Symbolic ✓ Neural ✓ Geometry ✓ Tensors ✓ Chaos")
print("[OK] Example 07")

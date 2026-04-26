"""
Example 13 — Gradient, Laplacian & Curl Operators
===================================================
Train scalar and vector field networks, then compute
all differential operators via autograd and compare
to exact symbolic results.

Fields covered
--------------
  f(x,y,z) = sin(πx)cos(πy)exp(-z)  → Laplacian, gradient
  F(x,y,z) = (yz, xz, xy)           → divergence=0, curl, Jacobian
  Harmonic field: ∇²f = 0           → Laplace equation solver
"""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch
import sympy as sp

from mnn.advanced.vector_calculus import (
    ScalarFieldNet, VectorFieldNet,
    FieldOperators, FieldConstraints, FieldTrainer, SymbolicValidator
)

print("=" * 60)
print("  MNN Example 13 — Operators: ∇, ∇², ∇×, ∇·")
print("=" * 60)

# ── Symbolic operator computation ─────────────────────────────
print("\n[1] Symbolic operator computation via SymPy")
from mnn.advanced.vector_calculus.symbolic_validation import SymbolicField

x,y,z = sp.symbols("x y z")

# Scalar field
f_sym  = SymbolicField(sp.sin(sp.pi*x)*sp.cos(sp.pi*y)*sp.exp(-z), ["x","y","z"], "f")
grad_f = f_sym.gradient()
lap_f  = f_sym.laplacian()
print(f"  f     = {f_sym.to_latex()}")
print(f"  ∇f    = {grad_f.to_latex()}")
print(f"  ∇²f   = {lap_f.to_latex()}")

# Vector field  F = (yz, xz, xy)
F_sym  = SymbolicField([y*z, x*z, x*y], ["x","y","z"], "F")
div_F  = F_sym.divergence()
curl_F = F_sym.curl()
jac_F  = F_sym.jacobian()
print(f"\n  F     = {F_sym.to_latex()}")
print(f"  ∇·F   = {div_F.to_latex()}")
print(f"  ∇×F   = {curl_F.to_latex()}")
print(f"  J_F   =\n{jac_F}")

# ── Train a scalar field network ──────────────────────────────
print("\n[2] Train scalar field network for f(x,y) = sin(x)cos(y)")
# Target: f(x,y) = sin(x)cos(y), ∇²f = -2sin(x)cos(y) = -2f
n_pts = 800
xy    = np.random.uniform(-np.pi, np.pi, (n_pts, 2)).astype(np.float32)
f_exact = (np.sin(xy[:,0]) * np.cos(xy[:,1])).reshape(-1,1).astype(np.float32)

scalar_net = ScalarFieldNet(space_dim=2, width=64, depth=4)
trainer_s  = FieldTrainer(scalar_net, lr=1e-3)
trainer_s.add_data(xy, f_exact, weight=1.0)
trainer_s.train(xy, n_epochs=1500, verbose=True, print_every=500)

# Evaluate gradient via autograd
x_eval  = torch.tensor(np.array([[0., 0.],[np.pi/2, 0.],[0., np.pi/2]]),
                         dtype=torch.float32)
grad_net = FieldOperators.gradient(scalar_net, x_eval.clone(), create_graph=False)
print(f"\n  ∇f at (0,0)     : network={grad_net[0].detach().numpy()}")
print(f"  ∇f exact (0,0)  : [{np.cos(0)*np.cos(0):.4f}, {-np.sin(0)*np.sin(0):.4f}]")
print(f"  ∇f at (π/2,0)   : network={grad_net[1].detach().numpy()}")
print(f"  ∇f exact (π/2,0): [{np.cos(np.pi/2)*np.cos(0):.4f}, {-np.sin(np.pi/2)*np.sin(0):.4f}]")

# Laplacian
lap_net = FieldOperators.laplacian(scalar_net, x_eval.clone(), create_graph=False)
lap_exact_0 = -2*np.sin(0)*np.cos(0)  # = 0
lap_exact_pi2 = -2*np.sin(np.pi/2)*np.cos(0)  # = -2
print(f"\n  ∇²f at (0,0)    : {lap_net[0].item():.4f}  (exact: {lap_exact_0:.4f})")
print(f"  ∇²f at (π/2,0)  : {lap_net[1].item():.4f}  (exact: {lap_exact_pi2:.4f})")

# ── Train harmonic field: ∇²f = 0 ─────────────────────────────
print("\n[3] Harmonic field: train ∇²f = 0 (Laplace equation)")
harm_net = ScalarFieldNet(space_dim=2, width=64, depth=4)
trainer_h = FieldTrainer(harm_net, lr=5e-4)

# Laplace constraint in interior
lap_constraint = FieldConstraints.harmonic(harm_net)
trainer_h.add_constraint("laplacian=0", lap_constraint, weight=1.0)

# Boundary: f = sin(πx) on y=0 (Dirichlet)
x_bc   = np.column_stack([np.linspace(0,1,100), np.zeros(100)]).astype(np.float32)
y_bc   = np.sin(np.pi * x_bc[:,0:1]).astype(np.float32)
trainer_h.add_data(x_bc, y_bc, weight=5.0)

pts_2d = np.random.uniform(0, 1, (500, 2)).astype(np.float32)
res_h  = trainer_h.train(pts_2d, n_epochs=2000, verbose=True, print_every=500)

violations_h = trainer_h.compute_constraint_violation(pts_2d[:200])
print(f"\n  Laplacian RMS violation: {violations_h['laplacian=0_rms']:.6f}")

# ── 3D Vector field: train then compute all operators ─────────
print("\n[4] 3D Vector field: curl, divergence, Jacobian via autograd")
# Target: F(x,y,z) = (yz, xz, xy)  →  ∇·F = 0
n3     = 600
xyz3   = np.random.uniform(-1, 1, (n3, 3)).astype(np.float32)
Fxyz   = np.column_stack([
    xyz3[:,1]*xyz3[:,2],   # yz
    xyz3[:,0]*xyz3[:,2],   # xz
    xyz3[:,0]*xyz3[:,1],   # xy
]).astype(np.float32)

vec_net  = VectorFieldNet(space_dim=3, width=64, depth=4)
trainer_v = FieldTrainer(vec_net, lr=1e-3)
trainer_v.add_data(xyz3, Fxyz, weight=1.0)
trainer_v.train(xyz3, n_epochs=2000, verbose=True, print_every=500)

# Operators at test point
x_t  = torch.tensor([[1.0, 0.5, -0.3]], dtype=torch.float32)
F_t  = vec_net(x_t)
div_t= FieldOperators.divergence(vec_net, x_t.clone(), create_graph=False)
curl_t=FieldOperators.curl(vec_net, x_t.clone(), create_graph=False)
J_t  = FieldOperators.jacobian(vec_net, x_t.clone(), create_graph=False)

print(f"\n  At (1.0, 0.5, -0.3):")
print(f"  F_net    = {F_t.detach().numpy().flatten().round(4)}")
print(f"  F_exact  = {np.array([0.5*(-0.3), 1.0*(-0.3), 1.0*0.5]).round(4)}")
print(f"  ∇·F_net  = {div_t.item():.4f}  (exact: 0.0000)")
print(f"  ∇×F_net  = {curl_t.detach().numpy().flatten().round(4)}")
print(f"  exact curl = {np.array([0.,0.,0.]).round(4)}")  # curl(yz,xz,xy) = 0
print(f"  Jacobian shape: {J_t.shape}")

print("\n[OK] Example 13 complete.")

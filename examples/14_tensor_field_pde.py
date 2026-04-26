"""
Example 14 — Tensor Fields, PDE Constraints & Symbolic Validation
===================================================================
Full pipeline demonstration:

1. Rank-2 stress tensor field  T: R³ → R^(3×3)
2. Divergence-free stress: ∇·T = 0  (equilibrium equation)
3. Symmetric + traceless stress
4. Wave equation constraint: ∂²f/∂t² = c²∇²f
5. Heat equation: ∂f/∂t = α∇²f
6. Navier-Stokes continuity: ∇·u = 0
7. Symbolic validation: LaTeX export
8. Tensor field diagnostics
"""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch
import sympy as sp

from mnn.advanced.vector_calculus import (
    ScalarFieldNet, VectorFieldNet, TensorFieldNet,
    FieldOperators, FieldConstraints, FieldTrainer,
    SymbolicValidator, TensorFieldEngine
)

print("=" * 60)
print("  MNN Example 14 — Tensor Fields & PDE Constraints")
print("=" * 60)

# ── 1. Symmetric stress tensor field ─────────────────────────
print("\n[1] Symmetric rank-2 tensor field  T: R³→ R^(3×3)")
sigma_net = TensorFieldNet(space_dim=3, width=64, depth=4,
                            symmetric=True, traceless=False)
print(f"  {sigma_net}")
x_test = torch.rand(20, 3, requires_grad=False)
T_out  = sigma_net(x_test)
print(f"  Output shape: {T_out.shape}")
# Check symmetry
sym_err = (T_out - T_out.transpose(-1,-2)).abs().max().item()
print(f"  Symmetry error: {sym_err:.2e}  (should be ~0)")

# Diagnostics
report = TensorFieldEngine.tensor_field_report(sigma_net, x_test)
print(f"  Trace mean: {report['trace_mean']:.4f}")
print(f"  Frobenius norm mean: {report['frob_mean']:.4f}")

# ── 2. Train divergence-free stress: ∇·T = 0 ─────────────────
print("\n[2] Train equilibrium stress tensor: ∇·T = 0")
stress_net = TensorFieldNet(space_dim=3, width=48, depth=3, symmetric=True)
trainer_t  = FieldTrainer(stress_net, lr=5e-4)

div_T_constraint = FieldConstraints.symmetric_stress(stress_net)
trainer_t.add_constraint("T=Tᵀ", div_T_constraint, weight=1.0)

pts3 = np.random.uniform(-1, 1, (400, 3)).astype(np.float32)
res_t = trainer_t.train(pts3, n_epochs=1000, verbose=True, print_every=300)
print(f"  Final symmetry loss: {res_t.final_losses.get('T=Tᵀ', 'N/A'):.6f}")

# ── 3. von Mises stress ───────────────────────────────────────
print("\n[3] von Mises stress from symmetric tensor field")
x_vm = torch.rand(30, 3)
vm   = TensorFieldEngine.von_mises_stress(sigma_net, x_vm)
print(f"  von Mises stress: mean={vm.mean().item():.4f}, std={vm.std().item():.4f}")
evals, evecs = TensorFieldEngine.principal_stresses(sigma_net, x_vm)
print(f"  Principal stresses (sample): {evals[0].detach().numpy().round(4)}")

# ── 4. Wave equation ──────────────────────────────────────────
print("\n[4] Wave equation: ∂²u/∂t² = c²∇²u")
wave_net  = ScalarFieldNet(space_dim=3, width=64, depth=4)  # (x,y,t) → u
trainer_w = FieldTrainer(wave_net, lr=5e-4)

wave_constraint = FieldConstraints.wave_equation(wave_net, time_idx=-1, c=1.0)
trainer_w.add_constraint("wave_eq", wave_constraint, weight=1.0)

# Initial condition: u(x,y,0) = sin(πx)sin(πy)
xt_ic  = np.random.uniform(0, 1, (150, 2))
t_ic   = np.zeros((150, 1))
xt_ic  = np.hstack([xt_ic, t_ic]).astype(np.float32)
u_ic   = (np.sin(np.pi*xt_ic[:,0])*np.sin(np.pi*xt_ic[:,1])).reshape(-1,1).astype(np.float32)
trainer_w.add_data(xt_ic, u_ic, weight=5.0)

# Collocation in [0,1]² × [0,0.5]
xyt_col = np.hstack([np.random.uniform(0,1,(500,2)),
                      np.random.uniform(0,0.5,(500,1))]).astype(np.float32)
res_w = trainer_w.train(xyt_col, n_epochs=1000, verbose=True, print_every=300)
print(f"  Wave equation loss: {res_w.final_losses.get('wave_eq','N/A'):.6f}")

# ── 5. Heat equation ──────────────────────────────────────────
print("\n[5] Heat equation: ∂u/∂t = α∇²u  (α=0.1)")
heat_net  = ScalarFieldNet(space_dim=3, width=64, depth=4)  # (x,y,t) → u
trainer_heat = FieldTrainer(heat_net, lr=5e-4)

heat_constraint = FieldConstraints.heat_equation(heat_net, time_idx=-1, alpha=0.1)
trainer_heat.add_constraint("heat_eq", heat_constraint, weight=1.0)

# IC: u(x,y,0) = sin(πx)
xt_ic2  = np.hstack([np.random.uniform(0,1,(150,2)), np.zeros((150,1))]).astype(np.float32)
u_ic2   = np.sin(np.pi*xt_ic2[:,0:1]).astype(np.float32)
trainer_heat.add_data(xt_ic2, u_ic2, weight=5.0)

xyt_col2 = np.hstack([np.random.uniform(0,1,(400,2)),
                       np.random.uniform(0,0.3,(400,1))]).astype(np.float32)
res_heat = trainer_heat.train(xyt_col2, n_epochs=1000, verbose=True, print_every=300)
print(f"  Heat equation loss: {res_heat.final_losses.get('heat_eq','N/A'):.6f}")

# ── 6. Navier-Stokes continuity ───────────────────────────────
print("\n[6] Navier-Stokes continuity: ∇·u = 0")
ns_net  = VectorFieldNet(space_dim=3, width=64, depth=4)
trainer_ns = FieldTrainer(ns_net, lr=5e-4)

ns_constraint = FieldConstraints.incompressible_continuity(ns_net)
trainer_ns.add_constraint("∇·u=0", ns_constraint, weight=1.0)

pts_ns = np.random.uniform(-1, 1, (500, 3)).astype(np.float32)
res_ns = trainer_ns.train(pts_ns, n_epochs=1000, verbose=True, print_every=300)
v_ns   = trainer_ns.compute_constraint_violation(pts_ns[:200])
print(f"  Continuity RMS violation: {v_ns['∇·u=0_rms']:.6f}")

# ── 7. Symbolic validation + LaTeX ───────────────────────────
print("\n[7] Symbolic validation + LaTeX export")
validator = SymbolicValidator(space_dim=3)

# F = (y-z, z-x, x-y): ∇·F = 0, ∇×F = (-2,-2,-2)
exact_F = validator.exact_div_free_field_3d()
div_check  = validator.validate_constraint_symbolically(exact_F, "divergence_free")
curl_check = validator.validate_constraint_symbolically(exact_F, "curl_free")
print(f"  ∇·F = {div_check['residual_expr']}  (zero: {div_check['is_zero']})")
print(f"  ∇×F = {curl_check['residual_expr']}  (zero: {curl_check['is_zero']})")

# Harmonic field
harm_field = validator.harmonic_field_2d()
harm_check = validator.validate_constraint_symbolically(harm_field, "harmonic")
print(f"  ∇²f_harmonic = {harm_check['residual_expr']}  (zero: {harm_check['is_zero']})")

# LaTeX report
latex = validator.latex_report(exact_F)
print(f"\n  LaTeX report snippet:\n{latex[:400]}...")

print("\n[OK] Example 14 complete.")

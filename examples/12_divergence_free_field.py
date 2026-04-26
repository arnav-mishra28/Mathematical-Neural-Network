"""
Example 12 — Divergence-Free Vector Field
==========================================
THE CORE DEMO: Train a neural vector field F: R³ → R³
such that ∇·F = 0 everywhere.

This is the mathematical law of incompressible fluid flow.
No training labels — only the divergence constraint.

Methods demonstrated
--------------------
1. Soft constraint (penalise divergence in loss)
2. Hard constraint (DivergenceFreeFieldNet via stream function / vector potential)
3. Symbolic validation via SymPy
"""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch

from mnn.advanced.vector_calculus import (
    VectorFieldNet, DivergenceFreeFieldNet,
    FieldOperators, FieldConstraints, FieldTrainer, SymbolicValidator
)

print("=" * 60)
print("  MNN Example 12 — Divergence-Free Vector Field")
print("=" * 60)

# ── Method 1: Soft constraint via training ────────────────────
print("\n[1] Soft constraint: train F with ∇·F = 0 loss")

net     = VectorFieldNet(space_dim=3, width=64, depth=4)
trainer = FieldTrainer(net, lr=1e-3)

# Only one constraint: ∇·F = 0
div_constraint = FieldConstraints.divergence_free(net)
trainer.add_constraint("divergence_free", div_constraint, weight=1.0)

# Collocation points in [-1,1]³
pts = np.random.uniform(-1, 1, (600, 3)).astype(np.float32)
result = trainer.train(pts, n_epochs=2000, verbose=True, print_every=400)

print("\n  Final losses:", result.final_losses)
violations = trainer.compute_constraint_violation(pts[:100])
print(f"  Constraint violation RMS: {violations['divergence_free_rms']:.6f}")

# ── Method 2: Hard constraint (exact by construction) ────────
print("\n[2] Hard constraint: DivergenceFreeFieldNet (∇·F = 0 exactly)")
div_free_net = DivergenceFreeFieldNet(space_dim=3, width=48, depth=3)

x_test = torch.tensor(np.random.uniform(-1,1,(50,3)).astype(np.float32), requires_grad=True)
F_out  = div_free_net(x_test)

# Verify divergence is zero
div_exact = FieldOperators.divergence(div_free_net, x_test.clone(), create_graph=False)
print(f"  Field output shape: {F_out.shape}")
print(f"  Divergence max  : {div_exact.abs().max().item():.2e}  (should be ~0)")
print(f"  Divergence mean : {div_exact.abs().mean().item():.2e}  (should be ~0)")

# ── Method 3: Symbolic validation ────────────────────────────
print("\n[3] Symbolic validation via SymPy")
validator = SymbolicValidator(space_dim=3)

# Exact divergence-free field: F = (y-z, z-x, x-y)
exact = validator.exact_div_free_field_3d()
print(f"  Exact field: {exact}")
check = validator.validate_constraint_symbolically(exact, "divergence_free")
print(f"  Symbolic ∇·F = {check['residual_expr']}  (is_zero={check['is_zero']})")
print(f"  LaTeX: {check['latex']}")

# Evaluate exact field on test points
test_pts = np.random.uniform(-1,1,(100,3)).astype(np.float32)
exact_vals = exact.evaluate_numpy(test_pts)
print(f"  Exact field sample: F(0,0,0) = {exact.evaluate_numpy(np.array([[0.,0.,0.]]))[0]}")

# Taylor-Green vortex
tg = validator.taylor_green_vortex()
tg_check = validator.validate_constraint_symbolically(tg, "divergence_free")
print(f"\n  Taylor-Green ∇·F = {tg_check['residual_expr']}  (is_zero={tg_check['is_zero']})")

print("\n[OK] Example 12 complete.")

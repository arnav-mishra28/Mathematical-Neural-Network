"""
Example 11 — MNN Prototype: Derivative-Constrained Network
=============================================================
THE FIRST MEANINGFUL MNN PROTOTYPE.

Task
----
  Train a neural network f: R → R such that:
      df/dx = 2x   for all x

  The expected (exact) solution is:  f(x) = x² + C

  We also anchor f(0) = 0, so the network should learn f(x) = x².

How the MNN constraint works
-----------------------------
  Unlike standard regression (data → output),
  the MNN encodes a MATHEMATICAL IDENTITY as the loss:

      L = mean( [f'(xᵢ) − 2xᵢ]² )  +  λ·(f(0) − 0)²

  f'(xᵢ) is computed by torch.autograd — automatic differentiation.
  No labels f(xᵢ) are ever provided. The network learns the function
  purely from the DERIVATIVE CONSTRAINT.

Expected outcome
----------------
  - Constraint loss → near zero
  - f(x) ≈ x² (after removing the free constant)
  - f'(x) ≈ 2x everywhere
"""
import sys; sys.path.insert(0, "..")
import numpy as np
from mnn.advanced.prototype import run_prototype, DerivativeConstrainedNet, DerivativeTrainer

print("=" * 56)
print("  MNN PROTOTYPE — df/dx = 2x  →  f(x) = x²")
print("=" * 56)

# ── Run the prototype ─────────────────────────────────────────
result = run_prototype(
    n_epochs   = 3000,
    width      = 64,
    depth      = 5,
    lr         = 1e-3,
    n_colloc   = 500,
    x_range    = (-3., 3.),
    w_deriv    = 1.0,
    w_anchor   = 10.0,
    anchor_x   = 0.0,
    anchor_val = 0.0,
    verbose    = True,
    seed       = 42,
)

# ── Inspect results ───────────────────────────────────────────
print("\n--- Point-wise evaluation ---")
x_show = [-3., -2., -1., 0., 1., 2., 3.]
for xi in x_show:
    idx  = np.argmin(np.abs(result.x_eval - xi))
    f_p  = result.f_pred[idx]
    f_e  = result.f_exact[idx]
    df_p = result.df_pred[idx]
    df_e = result.df_exact[idx]
    print(f"  x={xi:+.1f}:  f_pred={f_p:+7.4f}  f_exact={f_e:+7.4f}  "
          f"df_pred={df_p:+7.4f}  df_exact={df_e:+7.4f}")

# ── Training convergence ──────────────────────────────────────
print("\n--- Training convergence (every 300 epochs) ---")
every = max(1, len(result.total_losses) // 10)
for i in range(0, len(result.total_losses), every):
    print(f"  Epoch {i+1:>4}: "
          f"L_deriv={result.constraint_losses[i]:.6f}  "
          f"L_anchor={result.anchor_losses[i]:.6f}  "
          f"L_total={result.total_losses[i]:.6f}")

# ── Variant: custom derivative constraint ────────────────────
print("\n" + "=" * 56)
print("  VARIANT — df/dx = cos(x)  →  f(x) = sin(x) + C")
print("=" * 56)

model2  = DerivativeConstrainedNet(width=64, depth=5)
trainer2 = DerivativeTrainer(
    model           = model2,
    target_deriv_fn = np.cos,       # df/dx = cos(x)
    anchor_x        = 0.0,
    anchor_target   = 0.0,          # f(0) = sin(0) = 0
    w_deriv         = 1.0,
    w_anchor        = 10.0,
    lr              = 1e-3,
)
x_col2 = np.linspace(-np.pi, np.pi, 400).astype(np.float32)
trainer2.train(x_col2, n_epochs=2000, verbose=True, print_every=400)

x_eval2 = np.linspace(-np.pi, np.pi, 200).astype(np.float32)
res2     = trainer2.evaluate(x_eval2)

# Align to sin(x)
offset2  = float(np.mean(res2.f_pred - np.sin(res2.x_eval)))
f2_align = res2.f_pred - offset2
mse_f2   = float(np.mean((f2_align - np.sin(res2.x_eval))**2))
mse_df2  = float(np.mean((res2.df_pred - np.cos(res2.x_eval))**2))
print(f"\n  MSE(f vs sin(x)) = {mse_f2:.8f}")
print(f"  MSE(f' vs cos(x)) = {mse_df2:.8f}")

print("\n--- Point-wise (variant) ---")
for xi in [-3.14, -1.57, 0., 1.57, 3.14]:
    idx  = np.argmin(np.abs(res2.x_eval - xi))
    print(f"  x={xi:+5.2f}: f_pred={res2.f_pred[idx]:+6.4f}  "
          f"sin(x)={np.sin(xi):+6.4f}  "
          f"df_pred={res2.df_pred[idx]:+6.4f}  "
          f"cos(x)={np.cos(xi):+6.4f}")

print("\n[OK] Example 11 complete.")

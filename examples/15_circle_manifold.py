"""
Example 15 — Part 1 & 2: Circle Manifold + Autoencoder
=========================================================
PART 1: Build S¹ dataset — a circle manifold in R²
PART 2: Train autoencoder to learn the circle's structure
PART 3: Enforce x² + y² = 1 topological constraint
"""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch
from mnn.advanced.manifold_learning.datasets  import ManifoldGenerator
from mnn.advanced.manifold_learning.autoencoder import ManifoldAutoencoder, ManifoldAETrainer
from mnn.advanced.manifold_learning.constraints import ManifoldConstraints
from mnn.advanced.manifold_learning.analysis    import ManifoldAnalyzer

print("=" * 58)
print("  MNN Example 15 — Circle Manifold Learning")
print("=" * 58)

# ── PART 1: Build circle dataset ─────────────────────────────
print("\n[PART 1] Build S¹ dataset")
ds = ManifoldGenerator.circle(n=2000, radius=1.0, noise=0.05)
print(f"  {ds}")
print(f"  Points shape:    {ds.points.shape}")
print(f"  Params shape:    {ds.params.shape}  (theta values)")
print(f"  Ambient dim:     {ds.ambient_dim}")
print(f"  Intrinsic dim:   {ds.intrinsic_dim}")
print(f"  Radius:          {ds.radius}")
print(f"  Sample point:    {ds.points[0]}  (should have ‖·‖ ≈ 1)")
print(f"  Sample norm:     {float(np.linalg.norm(ds.points[0])):.4f}")

# Dataset statistics
norms = np.linalg.norm(ds.points, axis=1)
print(f"  Norm stats:      mean={norms.mean():.4f}, std={norms.std():.4f}")

# ── PART 2: Train autoencoder (ambient=2, latent=1) ──────────
print("\n[PART 2] Train Manifold Autoencoder — 2D→1D→2D")
ae     = ManifoldAutoencoder(ambient_dim=2, latent_dim=1,
                              encoder_widths=[32, 16],
                              decoder_widths=[16, 32])
trainer = ManifoldAETrainer(ae, lr=1e-3)
print(f"  Model: {ae}")

result = trainer.train(ds.points, n_epochs=2000, batch_size=256,
                        verbose=True, print_every=500,
                        manifold_name="S¹")
print(f"\n  Training result:\n{result.summary()}")

# Evaluate reconstruction
recon_err = trainer.reconstruction_error(ds.points)
print(f"\n  Reconstruction MSE: {recon_err:.6f}")

# Encode a few points
z_sample = ae.encode_numpy(ds.points[:5])
print(f"\n  Latent codes (5 samples):\n  {z_sample.flatten()}")

# ── PART 3: Enforce x² + y² = 1 topological constraint ───────
print("\n[PART 3] Train with topological constraint: x² + y² = 1")
ae2     = ManifoldAutoencoder(ambient_dim=2, latent_dim=1,
                               encoder_widths=[32, 16],
                               decoder_widths=[16, 32])
trainer2 = ManifoldAETrainer(ae2, lr=1e-3)

# Add constraint: reconstructed points must lie on the circle
circle_constraint = ManifoldConstraints.on_circle(radius=1.0)
trainer2.add_constraint("on_circle", circle_constraint, weight=2.0)

result2 = trainer2.train(ds.points, n_epochs=2000, batch_size=256,
                          verbose=True, print_every=500,
                          manifold_name="S¹+Constraint")
print(f"\n  With constraint:\n{result2.summary()}")

# Check how well the constraint is satisfied
recon2 = trainer2.reconstruct(ds.points)
norms2 = np.linalg.norm(recon2, axis=1)
print(f"\n  Reconstructed norm: mean={norms2.mean():.4f}  std={norms2.std():.4f}")
print(f"  (Should be ≈ 1.0000)")

# Topological analysis
print("\n[PART 3b] Topological analysis")
betti = ManifoldAnalyzer.betti_numbers_vietoris_rips(ds.points[:300], epsilon=0.3)
print(f"  β₀ = {betti['beta_0']}  (expected 1 = one component)")
print(f"  β₁ = {betti['beta_1']}  (expected 1 = one loop = circle!)")

dim_corr = ManifoldAnalyzer.intrinsic_dim_correlation(ds.points[:500])
dim_mle  = ManifoldAnalyzer.intrinsic_dim_mle(ds.points[:500])
print(f"  Intrinsic dim (correlation): {dim_corr:.2f}  (expected ≈ 1)")
print(f"  Intrinsic dim (MLE):         {dim_mle:.2f}   (expected ≈ 1)")

print("\n[OK] Example 15 complete.")

"""
Example 16 — Part 4: Higher Manifolds
=======================================
PART 4: Generalise to S², T², hyperspheres, SO(3), trefoil knot.
Demonstrate manifold-aware learning across topologies.
"""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch
from mnn.advanced.manifold_learning.datasets   import ManifoldGenerator
from mnn.advanced.manifold_learning.autoencoder import (
    ManifoldAutoencoder, ManifoldVAE, GeometricAutoencoder, ManifoldAETrainer
)
from mnn.advanced.manifold_learning.constraints import ManifoldConstraints
from mnn.advanced.manifold_learning.analysis    import ManifoldAnalyzer

print("=" * 58)
print("  MNN Example 16 — Higher Manifolds")
print("=" * 58)

# ── S²: 2-sphere in R³ ───────────────────────────────────────
print("\n[1] S² — 2-Sphere in R³ (intrinsic dim=2, ambient=3)")
ds_s2 = ManifoldGenerator.sphere(n=2000, radius=1.0, noise=0.02)
print(f"  {ds_s2}")

ae_s2    = ManifoldAutoencoder(ambient_dim=3, latent_dim=2,
                                encoder_widths=[64, 32],
                                decoder_widths=[32, 64])
trainer_s2 = ManifoldAETrainer(ae_s2, lr=1e-3)
trainer_s2.add_constraint("on_sphere", ManifoldConstraints.on_sphere(1.0), weight=2.0)

res_s2 = trainer_s2.train(ds_s2.points, n_epochs=1500, batch_size=256,
                            verbose=True, print_every=500, manifold_name="S²")
recon_s2 = trainer_s2.reconstruct(ds_s2.points)
norms_s2 = np.linalg.norm(recon_s2, axis=1)
print(f"  Recon MSE:     {res_s2.final_losses.get('recon',0):.6f}")
print(f"  Recon norm:    mean={norms_s2.mean():.4f}  (should ≈ 1.0)")

betti_s2 = ManifoldAnalyzer.betti_numbers_vietoris_rips(ds_s2.points[:300])
print(f"  β₀={betti_s2['beta_0']}, β₁={betti_s2['beta_1']}  (S²: expect β₁=0, β₂=1)")
dim_s2 = ManifoldAnalyzer.intrinsic_dim_mle(ds_s2.points[:400])
print(f"  Intrinsic dim (MLE): {dim_s2:.2f}  (expected ≈ 2)")

# ── T²: Torus in R³ ──────────────────────────────────────────
print("\n[2] T² — Torus in R³ (intrinsic dim=2)")
ds_t2 = ManifoldGenerator.torus(n=2000, R=2.0, r=0.8, noise=0.02)
print(f"  {ds_t2}")

ae_t2    = ManifoldAutoencoder(ambient_dim=3, latent_dim=2,
                                encoder_widths=[64, 32],
                                decoder_widths=[32, 64])
trainer_t2 = ManifoldAETrainer(ae_t2, lr=1e-3)
torus_c  = ManifoldConstraints.on_torus(R=2.0, r=0.8)
trainer_t2.add_constraint("on_torus", torus_c, weight=1.0)

res_t2 = trainer_t2.train(ds_t2.points, n_epochs=1500, batch_size=256,
                            verbose=True, print_every=500, manifold_name="T²")
print(f"  Recon MSE: {res_t2.final_losses.get('recon',0):.6f}")
dim_t2 = ManifoldAnalyzer.intrinsic_dim_mle(ds_t2.points[:400])
print(f"  Intrinsic dim (MLE): {dim_t2:.2f}  (expected ≈ 2)")

# ── Trefoil Knot ─────────────────────────────────────────────
print("\n[3] Trefoil Knot in R³ (intrinsic dim=1, non-trivial topology)")
ds_tref = ManifoldGenerator.trefoil_knot(n=1500, noise=0.02)
print(f"  {ds_tref}")

ae_tref    = ManifoldAutoencoder(ambient_dim=3, latent_dim=1,
                                  encoder_widths=[64, 32],
                                  decoder_widths=[32, 64])
trainer_tref = ManifoldAETrainer(ae_tref, lr=1e-3)
res_tref = trainer_tref.train(ds_tref.points, n_epochs=1200, batch_size=128,
                               verbose=True, print_every=400, manifold_name="TrefoilKnot")
print(f"  Recon MSE: {res_tref.final_losses.get('recon',0):.6f}")
dim_tref = ManifoldAnalyzer.intrinsic_dim_mle(ds_tref.points[:300])
print(f"  Intrinsic dim (MLE): {dim_tref:.2f}  (expected ≈ 1)")

# ── Hypersphere S⁴ ⊂ R⁵ ─────────────────────────────────────
print("\n[4] S⁴ ⊂ R⁵ — 4-sphere in 5D ambient space")
ds_s4 = ManifoldGenerator.hypersphere(n=2000, ambient_dim=5, radius=1.0)
print(f"  {ds_s4}")

ae_s4    = ManifoldAutoencoder(ambient_dim=5, latent_dim=4,
                                encoder_widths=[128, 64],
                                decoder_widths=[64, 128])
trainer_s4 = ManifoldAETrainer(ae_s4, lr=1e-3)
trainer_s4.add_constraint("on_sphere5", ManifoldConstraints.on_sphere(1.0), weight=2.0)

res_s4 = trainer_s4.train(ds_s4.points, n_epochs=1200, batch_size=256,
                            verbose=True, print_every=400, manifold_name="S⁴⊂R⁵")
recon_s4 = trainer_s4.reconstruct(ds_s4.points)
norms_s4 = np.linalg.norm(recon_s4, axis=1)
print(f"  Recon norm: mean={norms_s4.mean():.4f}  (should ≈ 1.0)")
dim_s4 = ManifoldAnalyzer.intrinsic_dim_pca(ds_s4.points)
print(f"  Intrinsic dim (PCA): {dim_s4}  (expected ≈ 4)")

# ── VAE on Swiss Roll ─────────────────────────────────────────
print("\n[5] VAE — Swiss Roll (intrinsic dim=2 hidden in R³)")
ds_sr = ManifoldGenerator.swiss_roll(n=2000, noise=0.1)
print(f"  {ds_sr}")

vae_sr   = ManifoldVAE(ambient_dim=3, latent_dim=2,
                        hidden_widths=[64, 32], beta=1.0)
trainer_vae = ManifoldAETrainer(vae_sr, lr=1e-3)
res_vae = trainer_vae.train(ds_sr.points, n_epochs=1500, batch_size=256,
                              verbose=True, print_every=500, manifold_name="SwissRoll-VAE")
print(f"  VAE losses: {res_vae.final_losses}")

z_sr = trainer_vae.encode(ds_sr.points)
print(f"  Latent coverage: {ManifoldAnalyzer.latent_coverage(z_sr):.4f}")
dim_sr = ManifoldAnalyzer.intrinsic_dim_mle(ds_sr.points[:400])
print(f"  Intrinsic dim (MLE): {dim_sr:.2f}  (expected ≈ 2)")

# ── Geometric AE on Sphere ────────────────────────────────────
print("\n[6] Geometric AE — S² with isometry constraint")
ds_s2b = ManifoldGenerator.sphere(n=1500, radius=1.0, noise=0.01)

geo_ae   = GeometricAutoencoder(ambient_dim=3, latent_dim=2,
                                  hidden_widths=[64,32], lambda_dist=0.5)
trainer_g = ManifoldAETrainer(geo_ae, lr=1e-3)
res_g = trainer_g.train(ds_s2b.points, n_epochs=1200, batch_size=256,
                         verbose=True, print_every=400, manifold_name="S²-Geo")
smooth = ManifoldAnalyzer.manifold_smoothness(geo_ae.encoder, ds_s2b.points[:200])
print(f"  Encoder smoothness (Lipschitz ratio): {smooth:.4f}")

# ── Full analysis report ──────────────────────────────────────
print("\n[7] Full analysis report for S²")
recon_full = trainer_s2.reconstruct(ds_s2.points[:500])
z_full     = ae_s2.encode_numpy(ds_s2.points[:500])
report     = ManifoldAnalyzer.full_report(
    ds_s2.points[:500], recon_full, z_full, true_intrinsic_dim=2
)
for k,v in report.items():
    print(f"  {k:<30}: {v:.4f}" if isinstance(v,float) else f"  {k:<30}: {v}")

print("\n[OK] Example 16 complete.")

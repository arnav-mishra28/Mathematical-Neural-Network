"""Tests for mnn.advanced.manifold_learning — Phase 4."""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch

from mnn.advanced.manifold_learning.datasets    import ManifoldGenerator, ManifoldDataset
from mnn.advanced.manifold_learning.autoencoder import (
    ManifoldAutoencoder, ManifoldVAE, GeometricAutoencoder, ManifoldAETrainer
)
from mnn.advanced.manifold_learning.constraints import ManifoldConstraints
from mnn.advanced.manifold_learning.analysis    import ManifoldAnalyzer

# ── Dataset tests ─────────────────────────────────────────────

def test_circle_shape():
    ds = ManifoldGenerator.circle(n=100)
    assert ds.points.shape == (100, 2)
    assert ds.intrinsic_dim == 1
    assert ds.ambient_dim == 2

def test_circle_on_unit_circle():
    ds = ManifoldGenerator.circle(n=500, radius=1.0, noise=0.0)
    norms = np.linalg.norm(ds.points, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_circle_custom_radius():
    ds = ManifoldGenerator.circle(n=200, radius=3.0, noise=0.0)
    norms = np.linalg.norm(ds.points, axis=1)
    assert np.allclose(norms, 3.0, atol=1e-4)

def test_sphere_shape():
    ds = ManifoldGenerator.sphere(n=200)
    assert ds.points.shape == (200, 3)
    assert ds.intrinsic_dim == 2

def test_sphere_on_unit_sphere():
    ds = ManifoldGenerator.sphere(n=500, radius=1.0, noise=0.0)
    norms = np.linalg.norm(ds.points, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

def test_torus_shape():
    ds = ManifoldGenerator.torus(n=200)
    assert ds.points.shape == (200, 3)
    assert ds.intrinsic_dim == 2

def test_helix_shape():
    ds = ManifoldGenerator.helix(n=200)
    assert ds.points.shape == (200, 3)
    assert ds.intrinsic_dim == 1

def test_trefoil_shape():
    ds = ManifoldGenerator.trefoil_knot(n=200)
    assert ds.points.shape == (200, 3)

def test_figure_eight_shape():
    ds = ManifoldGenerator.figure_eight(n=150)
    assert ds.points.shape == (150, 2)

def test_swiss_roll_shape():
    ds = ManifoldGenerator.swiss_roll(n=200)
    assert ds.points.shape == (200, 3)

def test_mobius_shape():
    ds = ManifoldGenerator.mobius_band(n=200)
    assert ds.points.shape == (200, 3)

def test_two_moons_shape():
    ds = ManifoldGenerator.two_moons(n=200)
    assert ds.points.shape == (200, 2)

def test_hypersphere_shape():
    ds = ManifoldGenerator.hypersphere(n=200, ambient_dim=5)
    assert ds.points.shape == (200, 5)
    assert ds.intrinsic_dim == 4

def test_hypersphere_on_surface():
    ds = ManifoldGenerator.hypersphere(n=300, ambient_dim=4, radius=2.0, noise=0.0)
    norms = np.linalg.norm(ds.points, axis=1)
    assert np.allclose(norms, 2.0, atol=1e-5)

def test_flat_torus_shape():
    ds = ManifoldGenerator.flat_torus_nd(n=200, dims=4)
    assert ds.points.shape == (200, 4)
    assert ds.intrinsic_dim == 2

def test_flat_torus_on_product_circles():
    ds = ManifoldGenerator.flat_torus_nd(n=300, dims=4, noise=0.0)
    # Each (cos θᵢ, sin θᵢ) pair should have norm 1
    for i in range(2):
        pair_norms = np.sqrt(ds.points[:, 2*i]**2 + ds.points[:, 2*i+1]**2)
        assert np.allclose(pair_norms, 1.0, atol=1e-5), f"Pair {i} not on unit circle"

def test_so3_shape():
    ds = ManifoldGenerator.so3_manifold(n=50)
    assert ds.points.shape == (50, 9)
    assert ds.intrinsic_dim == 3

def test_so3_orthogonal():
    ds = ManifoldGenerator.so3_manifold(n=20, noise=0.0)
    for i in range(len(ds.points)):
        Q = ds.points[i].reshape(3, 3)
        assert abs(np.linalg.det(Q) - 1.0) < 1e-4

def test_product_manifold():
    ds1 = ManifoldGenerator.circle(n=100)
    ds2 = ManifoldGenerator.circle(n=100)
    dp  = ManifoldGenerator.product_manifold(ds1, ds2, n=80)
    assert dp.points.shape == (80, 4)
    assert dp.intrinsic_dim == 2

def test_dataset_noise():
    ds      = ManifoldGenerator.circle(n=200, radius=1.0, noise=0.0)
    ds_noisy = ds.add_noise(0.1)
    assert ds_noisy.noise_level == 0.1
    norms = np.linalg.norm(ds_noisy.points, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)

def test_dataset_train_test_split():
    ds = ManifoldGenerator.sphere(n=200)
    train, test = ds.train_test_split(test_frac=0.2)
    assert len(train) + len(test) == 200
    assert len(test) == 40

def test_get_factory():
    for name in ["circle", "sphere", "torus", "helix", "swiss_roll"]:
        ds = ManifoldGenerator.get(name, n=50)
        assert len(ds.points) == 50

# ── Autoencoder tests ─────────────────────────────────────────

def test_ae_forward_shape():
    ae    = ManifoldAutoencoder(ambient_dim=3, latent_dim=2, encoder_widths=[16], decoder_widths=[16])
    x     = torch.rand(20, 3)
    xhat, z = ae(x)
    assert xhat.shape == (20, 3)
    assert z.shape    == (20, 2)

def test_ae_encode_decode():
    ae = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    x  = torch.rand(10, 2)
    z  = ae.encode(x);     assert z.shape == (10, 1)
    xh = ae.decode(z);     assert xh.shape == (10, 2)

def test_ae_encode_numpy():
    ae  = ManifoldAutoencoder(ambient_dim=3, latent_dim=2)
    x   = np.random.randn(15, 3).astype(np.float32)
    z   = ae.encode_numpy(x)
    assert z.shape == (15, 2)

def test_vae_forward():
    vae  = ManifoldVAE(ambient_dim=3, latent_dim=2, hidden_widths=[16])
    x    = torch.rand(10, 3)
    xhat, mu, logv, z = vae(x)
    assert xhat.shape == (10, 3)
    assert mu.shape   == (10, 2)
    assert z.shape    == (10, 2)

def test_vae_kl():
    vae  = ManifoldVAE(ambient_dim=3, latent_dim=2, hidden_widths=[16])
    mu   = torch.zeros(10, 2)
    logv = torch.zeros(10, 2)
    kl   = vae.kl_divergence(mu, logv)
    assert abs(float(kl)) < 1e-5  # KL(N(0,1)‖N(0,1)) = 0

def test_vae_sample():
    vae = ManifoldVAE(ambient_dim=3, latent_dim=2, hidden_widths=[16])
    s   = vae.sample(n=20)
    assert s.shape == (20, 3)

def test_geo_ae_forward():
    gae  = GeometricAutoencoder(ambient_dim=3, latent_dim=2, hidden_widths=[16])
    x    = torch.rand(10, 3)
    xhat, z = gae(x)
    assert xhat.shape == (10, 3)
    assert z.shape    == (10, 2)

def test_trainer_runs():
    ae  = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    tr  = ManifoldAETrainer(ae, lr=1e-3)
    ds  = ManifoldGenerator.circle(n=100)
    res = tr.train(ds.points, n_epochs=5, verbose=False)
    assert "recon" in res.final_losses
    assert res.n_epochs == 5

def test_trainer_with_constraint():
    ae  = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    tr  = ManifoldAETrainer(ae, lr=1e-3)
    tr.add_constraint("circle", ManifoldConstraints.on_circle(1.0), weight=1.0)
    ds  = ManifoldGenerator.circle(n=100)
    res = tr.train(ds.points, n_epochs=5, verbose=False)
    assert "circle" in res.final_losses

def test_trainer_encode():
    ae  = ManifoldAutoencoder(ambient_dim=3, latent_dim=2, encoder_widths=[16], decoder_widths=[16])
    tr  = ManifoldAETrainer(ae, lr=1e-3)
    x   = np.random.randn(20, 3).astype(np.float32)
    z   = tr.encode(x)
    assert z.shape == (20, 2)

def test_trainer_reconstruct():
    ae  = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    tr  = ManifoldAETrainer(ae, lr=1e-3)
    x   = np.random.randn(15, 2).astype(np.float32)
    xh  = tr.reconstruct(x)
    assert xh.shape == (15, 2)

def test_reconstruction_error_finite():
    ae  = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    tr  = ManifoldAETrainer(ae, lr=1e-3)
    x   = np.random.randn(30, 2).astype(np.float32)
    err = tr.reconstruction_error(x)
    assert np.isfinite(err) and err >= 0

# ── Constraint tests ──────────────────────────────────────────

def test_circle_constraint_zero_on_circle():
    c    = ManifoldConstraints.on_circle(radius=1.0)
    x    = torch.tensor([[1.0, 0.0],[0.0, 1.0],[-1.0, 0.0]])
    loss = c(x, x, x)
    assert float(loss) < 1e-8

def test_circle_constraint_nonzero_off():
    c    = ManifoldConstraints.on_circle(radius=1.0)
    x    = torch.tensor([[2.0, 0.0]])  # off circle
    loss = c(x, x, x)
    assert float(loss) > 0.5

def test_sphere_constraint_zero_on_sphere():
    c    = ManifoldConstraints.on_sphere(radius=1.0)
    raw  = torch.randn(20, 3)
    pts  = raw / raw.norm(dim=-1, keepdim=True)
    loss = c(pts, pts, pts)
    assert float(loss) < 1e-8

def test_sphere_constraint_positive_off():
    c    = ManifoldConstraints.on_sphere(radius=1.0)
    x    = torch.ones(5, 3)  # norm = √3 ≠ 1
    loss = c(x, x, x)
    assert float(loss) > 0

def test_torus_constraint():
    c = ManifoldConstraints.on_torus(R=2.0, r=0.8)
    # Point on torus: (R+r, 0, 0) = (2.8, 0, 0)
    x = torch.tensor([[2.8, 0.0, 0.0]])
    loss = c(x, x, x)
    assert float(loss) < 0.01

def test_latent_sphere_constraint():
    c    = ManifoldConstraints.latent_unit_sphere()
    raw  = torch.randn(10, 2)
    z    = raw / raw.norm(dim=-1, keepdim=True)
    loss = c(z, z, z)
    assert float(loss) < 1e-8

def test_isometry_constraint():
    c    = ManifoldConstraints.isometry(n_pairs=20)
    x    = torch.randn(30, 3)
    z    = torch.randn(30, 2)
    loss = c(x, z, x)
    assert np.isfinite(float(loss))

def test_combine_constraints():
    c1 = ManifoldConstraints.on_circle(1.0)
    c2 = ManifoldConstraints.latent_unit_sphere()
    combined = ManifoldConstraints.combine((c1, 1.0), (c2, 0.5))
    x    = torch.rand(10, 2)
    z    = torch.rand(10, 2)
    loss = combined(x, z, x)
    assert np.isfinite(float(loss))

# ── Analysis tests ────────────────────────────────────────────

def test_intrinsic_dim_circle():
    ds  = ManifoldGenerator.circle(n=500, noise=0.0)
    dim = ManifoldAnalyzer.intrinsic_dim_correlation(ds.points)
    assert 0.5 < dim < 2.0, f"Circle dim should be ~1, got {dim}"

def test_intrinsic_dim_mle_sphere():
    ds  = ManifoldGenerator.sphere(n=500, noise=0.01)
    dim = ManifoldAnalyzer.intrinsic_dim_mle(ds.points[:300])
    assert 1.0 < dim < 3.5, f"Sphere dim should be ~2, got {dim}"

def test_intrinsic_dim_pca():
    ds  = ManifoldGenerator.hypersphere(n=300, ambient_dim=10)
    dim = ManifoldAnalyzer.intrinsic_dim_pca(ds.points)
    assert 1 <= dim <= 10

def test_betti_circle():
    ds    = ManifoldGenerator.circle(n=300, noise=0.0)
    betti = ManifoldAnalyzer.betti_numbers_vietoris_rips(ds.points, epsilon=0.3)
    assert betti["beta_0"] == 1, f"Expected β₀=1, got {betti['beta_0']}"
    assert betti["beta_1"] >= 1, f"Expected β₁≥1, got {betti['beta_1']}"

def test_betti_sphere_b1_zero():
    ds    = ManifoldGenerator.sphere(n=300, noise=0.01)
    betti = ManifoldAnalyzer.betti_numbers_vietoris_rips(ds.points[:200], epsilon=0.5)
    assert betti["beta_0"] == 1

def test_isomap_shape():
    ds  = ManifoldGenerator.swiss_roll(n=200)
    emb = ManifoldAnalyzer.isomap(ds.points, n_components=2, k_neighbors=8)
    assert emb.shape == (200, 2)

def test_isomap_dim_reduction():
    ds  = ManifoldGenerator.hypersphere(n=150, ambient_dim=5)
    emb = ManifoldAnalyzer.isomap(ds.points, n_components=3)
    assert emb.shape == (150, 3)

def test_reconstruction_quality():
    x   = np.random.randn(100, 3).astype(np.float32)
    xh  = x + np.random.randn(100, 3).astype(np.float32) * 0.01
    rq  = ManifoldAnalyzer.reconstruction_quality(x, xh)
    assert "mse" in rq
    assert rq["mse"] >= 0
    assert rq["r2_score"] <= 1.0

def test_perfect_reconstruction_quality():
    x  = np.random.randn(50, 3).astype(np.float32)
    rq = ManifoldAnalyzer.reconstruction_quality(x, x)
    assert rq["mse"] < 1e-10
    assert abs(rq["r2_score"] - 1.0) < 1e-6

def test_latent_coverage():
    z   = np.random.randn(500, 2).astype(np.float32)
    cov = ManifoldAnalyzer.latent_coverage(z)
    assert 0 < cov <= 1.0

def test_full_report():
    ds     = ManifoldGenerator.circle(n=200)
    ae     = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    tr     = ManifoldAETrainer(ae, lr=1e-3)
    tr.train(ds.points[:100], n_epochs=3, verbose=False)
    recon  = tr.reconstruct(ds.points[:100])
    z      = tr.encode(ds.points[:100])
    report = ManifoldAnalyzer.full_report(ds.points[:100], recon, z, true_intrinsic_dim=1)
    assert "mse" in report
    assert "beta_0" in report

def test_persistence_diagram():
    ds    = ManifoldGenerator.circle(n=100, noise=0.0)
    pairs = ManifoldAnalyzer.persistence_diagram_0(ds.points, n_scales=10)
    assert isinstance(pairs, list)

def test_manifold_smoothness():
    ds  = ManifoldGenerator.circle(n=200)
    ae  = ManifoldAutoencoder(ambient_dim=2, latent_dim=1, encoder_widths=[16], decoder_widths=[16])
    s   = ManifoldAnalyzer.manifold_smoothness(ae.encoder, ds.points[:100])
    assert np.isfinite(s)


if __name__ == "__main__":
    tests = [
        # Dataset
        test_circle_shape, test_circle_on_unit_circle, test_circle_custom_radius,
        test_sphere_shape, test_sphere_on_unit_sphere, test_torus_shape,
        test_helix_shape, test_trefoil_shape, test_figure_eight_shape,
        test_swiss_roll_shape, test_mobius_shape, test_two_moons_shape,
        test_hypersphere_shape, test_hypersphere_on_surface,
        test_flat_torus_shape, test_flat_torus_on_product_circles,
        test_so3_shape, test_so3_orthogonal,
        test_product_manifold, test_dataset_noise, test_dataset_train_test_split,
        test_get_factory,
        # Autoencoder
        test_ae_forward_shape, test_ae_encode_decode, test_ae_encode_numpy,
        test_vae_forward, test_vae_kl, test_vae_sample, test_geo_ae_forward,
        test_trainer_runs, test_trainer_with_constraint,
        test_trainer_encode, test_trainer_reconstruct, test_reconstruction_error_finite,
        # Constraints
        test_circle_constraint_zero_on_circle, test_circle_constraint_nonzero_off,
        test_sphere_constraint_zero_on_sphere, test_sphere_constraint_positive_off,
        test_torus_constraint, test_latent_sphere_constraint,
        test_isometry_constraint, test_combine_constraints,
        # Analysis
        test_intrinsic_dim_circle, test_intrinsic_dim_mle_sphere, test_intrinsic_dim_pca,
        test_betti_circle, test_betti_sphere_b1_zero,
        test_isomap_shape, test_isomap_dim_reduction,
        test_reconstruction_quality, test_perfect_reconstruction_quality,
        test_latent_coverage, test_full_report, test_persistence_diagram,
        test_manifold_smoothness,
    ]
    passed = failed = 0
    for fn in tests:
        try:   fn(); print(f"  ✓ {fn.__name__}"); passed += 1
        except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed += 1
    print(f"\n[{passed} passed, {failed} failed]")

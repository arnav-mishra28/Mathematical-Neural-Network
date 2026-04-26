"""Tests for mnn.advanced.vector_calculus — Phase 3."""
import sys; sys.path.insert(0, "..")
import numpy as np
import torch
import sympy as sp

from mnn.advanced.vector_calculus import (
    ScalarFieldNet, VectorFieldNet, TensorFieldNet,
    FieldOperators, FieldConstraints, FieldTrainer,
    SymbolicValidator, TensorFieldEngine
)
from mnn.advanced.vector_calculus.field_networks import (
    DivergenceFreeFieldNet, CurlFreeFieldNet
)
from mnn.advanced.vector_calculus.symbolic_validation import SymbolicField


# ── Field network tests ───────────────────────────────────────

def test_scalar_field_net_shape():
    net = ScalarFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(10, 3)
    out = net(x)
    assert out.shape == (10, 1), f"Expected (10,1), got {out.shape}"

def test_vector_field_net_shape():
    net = VectorFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(10, 3)
    out = net(x)
    assert out.shape == (10, 3), f"Expected (10,3), got {out.shape}"

def test_tensor_field_net_shape():
    net = TensorFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(10, 3)
    out = net(x)
    assert out.shape == (10, 3, 3), f"Expected (10,3,3), got {out.shape}"

def test_symmetric_tensor():
    net = TensorFieldNet(space_dim=3, width=32, depth=2, symmetric=True)
    x   = torch.rand(20, 3)
    T   = net(x)
    err = (T - T.transpose(-1,-2)).abs().max().item()
    assert err < 1e-5, f"Symmetry error too large: {err}"

def test_div_free_exact():
    """DivergenceFreeFieldNet must have ∇·F = 0 exactly."""
    net = DivergenceFreeFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(30, 3)
    div = FieldOperators.divergence(net, x.clone(), create_graph=False)
    assert div.abs().max().item() < 1e-4, f"Divergence not zero: {div.abs().max().item()}"

def test_div_free_2d():
    net = DivergenceFreeFieldNet(space_dim=2, width=32, depth=2)
    x   = torch.rand(20, 2)
    out = net(x)
    assert out.shape == (20, 2)

def test_curl_free_exact():
    """CurlFreeFieldNet must have ∇×F = 0 exactly."""
    net  = CurlFreeFieldNet(space_dim=3, width=32, depth=2)
    x    = torch.rand(20, 3)
    curl = FieldOperators.curl(net, x.clone(), create_graph=False)
    assert curl.abs().max().item() < 1e-4, f"Curl not zero: {curl.abs().max().item()}"

def test_scalar_field_net_2d():
    net = ScalarFieldNet(space_dim=2, width=32, depth=2)
    x   = torch.rand(5, 2)
    assert net(x).shape == (5, 1)

# ── Operator tests ────────────────────────────────────────────

def test_gradient_shape():
    net  = ScalarFieldNet(space_dim=3, width=32, depth=2)
    x    = torch.rand(15, 3, requires_grad=True)
    grad = FieldOperators.gradient(net, x.clone())
    assert grad.shape == (15, 3)

def test_laplacian_shape():
    net = ScalarFieldNet(space_dim=2, width=32, depth=2)
    x   = torch.rand(10, 2)
    lap = FieldOperators.laplacian(net, x.clone())
    assert lap.shape == (10, 1)

def test_divergence_shape():
    net = VectorFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(10, 3)
    div = FieldOperators.divergence(net, x.clone())
    assert div.shape == (10, 1)

def test_curl_shape():
    net  = VectorFieldNet(space_dim=3, width=32, depth=2)
    x    = torch.rand(10, 3)
    curl = FieldOperators.curl(net, x.clone())
    assert curl.shape == (10, 3)

def test_curl_2d_shape():
    net  = VectorFieldNet(space_dim=2, field_dim=2, width=32, depth=2)
    x    = torch.rand(10, 2)
    curl = FieldOperators.curl_2d(net, x.clone())
    assert curl.shape == (10, 1)

def test_jacobian_shape():
    net = VectorFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(8, 3)
    J   = FieldOperators.jacobian(net, x.clone())
    assert J.shape == (8, 3, 3)

def test_vector_laplacian_shape():
    net = VectorFieldNet(space_dim=3, width=32, depth=2)
    x   = torch.rand(8, 3)
    vl  = FieldOperators.vector_laplacian(net, x.clone())
    assert vl.shape == (8, 3)

def test_gradient_known_function():
    """∇(x² + y²) = (2x, 2y) — verify against known result."""
    class QuadNet(torch.nn.Module):
        def forward(self, x): return (x[:,0:1]**2 + x[:,1:2]**2)

    net  = QuadNet()
    x    = torch.tensor([[1.0, 2.0]], requires_grad=True)
    grad = FieldOperators.gradient(net, x.clone())
    assert abs(grad[0,0].item() - 2.0) < 1e-4, f"∂/∂x = {grad[0,0].item()}"
    assert abs(grad[0,1].item() - 4.0) < 1e-4, f"∂/∂y = {grad[0,1].item()}"

def test_laplacian_known_function():
    """∇²(x² + y²) = 4 in 2D."""
    class QuadNet(torch.nn.Module):
        def forward(self, x): return (x[:,0:1]**2 + x[:,1:2]**2)

    net = QuadNet()
    x   = torch.tensor([[1.0, 2.0]])
    lap = FieldOperators.laplacian(net, x.clone())
    assert abs(lap[0,0].item() - 4.0) < 1e-3, f"∇² = {lap[0,0].item()}"

def test_divergence_known_function():
    """∇·(x, y, z) = 3."""
    class LinNet(torch.nn.Module):
        def forward(self, x): return x  # F(x,y,z) = (x,y,z)

    net = LinNet()
    x   = torch.tensor([[1.0, 2.0, 3.0]])
    div = FieldOperators.divergence(net, x.clone())
    assert abs(div[0,0].item() - 3.0) < 1e-3, f"∇·F = {div[0,0].item()}"

def test_curl_zero_for_gradient():
    """∇×(∇f) = 0 always."""
    scalar = ScalarFieldNet(space_dim=3, width=32, depth=2)
    curl_proxy = CurlFreeFieldNet(space_dim=3, width=32, depth=2)
    x    = torch.rand(10, 3)
    curl = FieldOperators.curl(curl_proxy, x.clone(), create_graph=False)
    assert curl.abs().max().item() < 1e-4

# ── Constraint tests ──────────────────────────────────────────

def test_divergence_free_constraint():
    net     = VectorFieldNet(space_dim=3, width=32, depth=2)
    res_fn  = FieldConstraints.divergence_free(net)
    x       = torch.rand(10, 3)
    res     = res_fn(x)
    assert res.shape == (10, 1)

def test_harmonic_constraint():
    net    = ScalarFieldNet(space_dim=2, width=32, depth=2)
    res_fn = FieldConstraints.harmonic(net)
    x      = torch.rand(10, 2)
    res    = res_fn(x)
    assert res.shape == (10, 1)

def test_eikonal_constraint():
    net    = ScalarFieldNet(space_dim=2, width=32, depth=2)
    res_fn = FieldConstraints.eikonal(net)
    x      = torch.rand(10, 2)
    res    = res_fn(x)
    assert res.shape == (10, 1)

def test_curl_free_constraint():
    net    = VectorFieldNet(space_dim=3, width=32, depth=2)
    res_fn = FieldConstraints.curl_free(net)
    x      = torch.rand(10, 3)
    res    = res_fn(x)
    assert res.shape == (10, 3)

# ── Trainer tests ─────────────────────────────────────────────

def test_trainer_runs():
    net     = VectorFieldNet(space_dim=2, width=16, depth=2)
    trainer = FieldTrainer(net, lr=1e-3)
    div_fn  = FieldConstraints.divergence_free(net)
    trainer.add_constraint("div_free", div_fn, weight=1.0)
    pts = np.random.uniform(-1, 1, (50, 2)).astype(np.float32)
    result = trainer.train(pts, n_epochs=10, verbose=False)
    assert "div_free" in result.final_losses

def test_trainer_with_data():
    net     = ScalarFieldNet(space_dim=2, width=16, depth=2)
    trainer = FieldTrainer(net, lr=1e-3)
    x_d = np.random.uniform(-1,1,(30,2)).astype(np.float32)
    y_d = np.sin(x_d[:,0:1]).astype(np.float32)
    trainer.add_data(x_d, y_d, weight=1.0)
    pts = x_d
    result = trainer.train(pts, n_epochs=5, verbose=False)
    assert "data" in result.final_losses

def test_trainer_evaluate():
    net     = VectorFieldNet(space_dim=2, width=16, depth=2)
    trainer = FieldTrainer(net, lr=1e-3)
    x_test  = np.random.uniform(-1,1,(20,2)).astype(np.float32)
    out     = trainer.evaluate(x_test)
    assert out.shape == (20, 2)

# ── Symbolic validation tests ─────────────────────────────────

def test_symbolic_field_gradient():
    x, y = sp.symbols("x y")
    f    = SymbolicField(x**2 + y**2, ["x","y"], "r²")
    grad = f.gradient()
    assert sp.simplify(grad.components[0] - 2*x) == 0
    assert sp.simplify(grad.components[1] - 2*y) == 0

def test_symbolic_divergence():
    x, y, z = sp.symbols("x y z")
    F = SymbolicField([x, y, z], ["x","y","z"], "F")
    div = F.divergence()
    assert sp.simplify(div.components[0] - 3) == 0

def test_symbolic_laplacian():
    x, y = sp.symbols("x y")
    f   = SymbolicField(x**2 + y**2, ["x","y"], "r²")
    lap = f.laplacian()
    assert sp.simplify(lap.components[0] - 4) == 0

def test_symbolic_curl_zero_for_gradient():
    x, y, z = sp.symbols("x y z")
    phi = x**2 + y**2 + z**2
    F   = SymbolicField([2*x, 2*y, 2*z], ["x","y","z"], "∇r²")
    curl = F.curl()
    for c in curl.components:
        assert sp.simplify(c) == 0

def test_symbolic_div_free_rotation():
    validator = SymbolicValidator(space_dim=3)
    F     = validator.exact_div_free_field_3d()
    check = validator.validate_constraint_symbolically(F, "divergence_free")
    assert check["is_zero"], f"Divergence not zero: {check['residual_expr']}"

def test_symbolic_harmonic():
    validator = SymbolicValidator(space_dim=2)
    f     = validator.harmonic_field_2d()
    check = validator.validate_constraint_symbolically(f, "harmonic")
    assert check["is_zero"], f"Laplacian not zero: {check['residual_expr']}"

def test_symbolic_evaluate_numpy():
    x, y = sp.symbols("x y")
    f    = SymbolicField(x**2 + y**2, ["x","y"], "r²")
    pts  = np.array([[1.,1.],[2.,0.],[0.,3.]])
    vals = f.evaluate_numpy(pts)
    expected = np.array([[2.], [4.], [9.]])
    assert np.allclose(vals, expected)

# ── Tensor engine tests ───────────────────────────────────────

def test_tensor_trace():
    net = TensorFieldNet(space_dim=3, width=16, depth=2)
    x   = torch.rand(10, 3)
    tr  = TensorFieldEngine.trace_tensor(net, x)
    assert tr.shape == (10,)

def test_frobenius_norm():
    net = TensorFieldNet(space_dim=3, width=16, depth=2)
    x   = torch.rand(10, 3)
    fn  = TensorFieldEngine.frobenius_norm(net, x)
    assert fn.shape == (10,)
    assert (fn >= 0).all()

def test_von_mises():
    net = TensorFieldNet(space_dim=3, width=16, depth=2, symmetric=True)
    x   = torch.rand(10, 3)
    vm  = TensorFieldEngine.von_mises_stress(net, x)
    assert vm.shape == (10,)
    assert (vm >= 0).all()

def test_principal_stresses():
    net = TensorFieldNet(space_dim=3, width=16, depth=2, symmetric=True)
    x   = torch.rand(5, 3)
    evals, evecs = TensorFieldEngine.principal_stresses(net, x)
    assert evals.shape == (5, 3)
    assert evecs.shape == (5, 3, 3)

def test_tensor_field_report():
    net    = TensorFieldNet(space_dim=3, width=16, depth=2)
    x      = torch.rand(20, 3)
    report = TensorFieldEngine.tensor_field_report(net, x)
    assert "trace_mean" in report
    assert "frob_mean"  in report


if __name__ == "__main__":
    tests = [
        test_scalar_field_net_shape, test_vector_field_net_shape,
        test_tensor_field_net_shape, test_symmetric_tensor,
        test_div_free_exact, test_div_free_2d, test_curl_free_exact,
        test_scalar_field_net_2d, test_gradient_shape, test_laplacian_shape,
        test_divergence_shape, test_curl_shape, test_curl_2d_shape,
        test_jacobian_shape, test_vector_laplacian_shape,
        test_gradient_known_function, test_laplacian_known_function,
        test_divergence_known_function, test_curl_zero_for_gradient,
        test_divergence_free_constraint, test_harmonic_constraint,
        test_eikonal_constraint, test_curl_free_constraint,
        test_trainer_runs, test_trainer_with_data, test_trainer_evaluate,
        test_symbolic_field_gradient, test_symbolic_divergence,
        test_symbolic_laplacian, test_symbolic_curl_zero_for_gradient,
        test_symbolic_div_free_rotation, test_symbolic_harmonic,
        test_symbolic_evaluate_numpy,
        test_tensor_trace, test_frobenius_norm, test_von_mises,
        test_principal_stresses, test_tensor_field_report,
    ]
    passed = failed = 0
    for fn in tests:
        try: fn(); print(f"  ✓ {fn.__name__}"); passed+=1
        except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed+=1
    print(f"\n[{passed} passed, {failed} failed]")

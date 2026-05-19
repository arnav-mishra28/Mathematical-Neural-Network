"""mnn.intelligence.neural_pde — Generalized PINN for arbitrary PDEs.

Pillar 3: Neural PDE Solvers.
Solves PDEs using MNN by enforcing the PDE residual, boundary conditions,
and initial conditions as differentiable losses. Extends beyond physics-specific
PINNs into a general mathematical PDE solver.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Callable, Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm

from mnn.neural.base_network import MNNNetwork, FourierMNNNetwork


@dataclass
class PDEProblem:
    """Specification of a PDE problem.

    Attributes:
        name: Human-readable name.
        spatial_dim: Number of spatial dimensions.
        has_time: Whether the PDE is time-dependent.
        domain: Dictionary mapping variable names to (min, max) tuples.
        pde_fn: Callable(net, points) -> residual tensor. The PDE residual.
        boundary_conditions: List of (boundary_points, boundary_values) pairs.
        initial_condition: Optional (ic_points, ic_values) for time-dependent PDEs.
        exact_solution: Optional callable for validation.
    """
    name: str = "PDE"
    spatial_dim: int = 1
    has_time: bool = False
    domain: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pde_fn: Optional[Callable] = None
    boundary_conditions: List[Tuple] = field(default_factory=list)
    initial_condition: Optional[Tuple] = None
    exact_solution: Optional[Callable] = None


def _auto_diff(net: nn.Module, x: torch.Tensor, orders: List[Tuple[int, ...]]):
    """Compute arbitrary-order partial derivatives of net(x) via autograd.

    Args:
        net: Neural network.
        x: Input tensor (batch, n_vars), requires_grad will be set.
        orders: List of derivative specs, e.g. [(0,), (1,), (0,0), (0,1)] for
                du/dx0, du/dx1, d²u/dx0², d²u/dx0dx1.

    Returns:
        Dictionary mapping order tuples to derivative tensors.
    """
    x = x.requires_grad_(True)
    u = net(x)
    result = {(): u}

    first_grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    for i in range(x.shape[1]):
        result[(i,)] = first_grads[:, i:i+1]

    for order in orders:
        if len(order) <= 1:
            continue
        key = order
        if key in result:
            continue
        prev = order[:-1]
        var = order[-1]
        if prev not in result:
            continue
        g = torch.autograd.grad(result[prev].sum(), x, create_graph=True)[0]
        result[key] = g[:, var:var+1]

    return result


class GeneralizedPINN(nn.Module):
    """Generalized Physics-Informed Neural Network for arbitrary PDEs.

    Supports:
      - Arbitrary spatial dimensions
      - Time-dependent and steady-state problems
      - Dirichlet, Neumann, and periodic boundary conditions
      - Automatic differentiation for PDE residual computation
      - Fourier feature embedding for high-frequency solutions
    """
    def __init__(self, input_dim: int, output_dim: int = 1,
                 width: int = 128, depth: int = 5,
                 use_fourier: bool = True, n_fourier: int = 64,
                 fourier_scale: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if use_fourier:
            self.net = FourierMNNNetwork(input_dim, output_dim, n_fourier=n_fourier,
                                         width=width, depth=depth, scale=fourier_scale)
        else:
            self.net = MNNNetwork(input_dim, output_dim, width=width, depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_derivatives(self, x: torch.Tensor, max_order: int = 2) -> Dict:
        x = x.requires_grad_(True)
        u = self.forward(x)
        derivs = {(): u}

        # First-order
        g1 = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        for i in range(x.shape[1]):
            derivs[(i,)] = g1[:, i:i+1]

        # Second-order
        if max_order >= 2:
            for i in range(x.shape[1]):
                g2 = torch.autograd.grad(g1[:, i].sum(), x, create_graph=True)[0]
                for j in range(i, x.shape[1]):
                    derivs[(i, j)] = g2[:, j:j+1]
                    if i != j:
                        derivs[(j, i)] = g2[:, j:j+1]

        # Third-order (if needed)
        if max_order >= 3:
            for i in range(x.shape[1]):
                g2_i = torch.autograd.grad(g1[:, i].sum(), x, create_graph=True)[0]
                for j in range(x.shape[1]):
                    g3 = torch.autograd.grad(g2_i[:, j].sum(), x, create_graph=True)[0]
                    for k in range(x.shape[1]):
                        key = tuple(sorted([i, j, k]))
                        if key not in derivs:
                            derivs[key] = g3[:, k:k+1]

        return derivs


class NeuralPDESolver:
    """Solve arbitrary PDEs using the MNN framework.

    Combines:
      - PDE residual minimization (collocation)
      - Boundary condition enforcement
      - Initial condition enforcement
      - Adaptive weighting of loss components
    """
    def __init__(self, problem: PDEProblem, width: int = 128, depth: int = 5,
                 use_fourier: bool = True, lr: float = 1e-3,
                 device: str = "cpu"):
        self.problem = problem
        self.device = device
        input_dim = problem.spatial_dim + (1 if problem.has_time else 0)
        self.pinn = GeneralizedPINN(input_dim, 1, width=width, depth=depth,
                                     use_fourier=use_fourier).to(device)
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, min_lr=1e-6)
        self.history: Dict[str, List[float]] = {}

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(np.array(x), dtype=torch.float32).to(self.device)

    def sample_collocation(self, n_points: int) -> torch.Tensor:
        domain = self.problem.domain
        keys = sorted(domain.keys())
        samples = []
        for k in keys:
            lo, hi = domain[k]
            samples.append(torch.rand(n_points, 1) * (hi - lo) + lo)
        return torch.cat(samples, dim=1).to(self.device)

    def train(self, n_epochs: int = 5000, n_collocation: int = 2000,
              w_pde: float = 1.0, w_bc: float = 10.0, w_ic: float = 10.0,
              adaptive_weights: bool = True, verbose: bool = True,
              print_every: int = 500) -> Dict:

        # Prepare boundary data
        bc_data = []
        for bp, bv in self.problem.boundary_conditions:
            bc_data.append((self._t(bp), self._t(bv)))

        ic_data = None
        if self.problem.initial_condition is not None:
            ip, iv = self.problem.initial_condition
            ic_data = (self._t(ip), self._t(iv))

        it = tqdm(range(n_epochs), desc=f"Solving {self.problem.name}") if verbose else range(n_epochs)

        for ep in it:
            self.optimizer.zero_grad()
            losses = {}

            # PDE residual
            x_col = self.sample_collocation(n_collocation)
            residual = self.problem.pde_fn(self.pinn, x_col)
            pde_loss = torch.mean(residual ** 2)
            losses["pde"] = pde_loss.item()
            total = w_pde * pde_loss

            # Boundary conditions
            bc_total = torch.tensor(0.0, device=self.device)
            for bp, bv in bc_data:
                pred = self.pinn(bp)
                bc_total = bc_total + nn.functional.mse_loss(pred, bv)
            if bc_data:
                losses["bc"] = bc_total.item()
                total = total + w_bc * bc_total

            # Initial conditions
            if ic_data is not None:
                ip, iv = ic_data
                pred_ic = self.pinn(ip)
                ic_loss = nn.functional.mse_loss(pred_ic, iv)
                losses["ic"] = ic_loss.item()
                total = total + w_ic * ic_loss

            # Adaptive weight adjustment
            if adaptive_weights and ep > 0 and ep % 500 == 0:
                with torch.no_grad():
                    grad_pde = torch.autograd.grad(pde_loss, self.pinn.parameters(),
                                                    retain_graph=True, allow_unused=True)
                    grad_bc = torch.autograd.grad(bc_total, self.pinn.parameters(),
                                                   retain_graph=True, allow_unused=True)
                    max_grad_pde = max(g.abs().max().item() for g in grad_pde if g is not None)
                    max_grad_bc = max(g.abs().max().item() for g in grad_bc if g is not None) or 1e-8
                    w_bc = max(0.1, min(100.0, max_grad_pde / max_grad_bc))

            losses["total"] = total.item()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(total.item())

            for k, v in losses.items():
                self.history.setdefault(k, []).append(v)

            if verbose and (ep + 1) % print_every == 0:
                tqdm.write(f"[{ep+1}] " + " | ".join(f"{k}={v:.6f}" for k, v in losses.items()))

        return self.history

    def predict(self, points: np.ndarray) -> np.ndarray:
        self.pinn.eval()
        with torch.no_grad():
            return self.pinn(self._t(points)).cpu().numpy()

    def error(self, points: np.ndarray) -> float:
        if self.problem.exact_solution is None:
            raise ValueError("No exact solution provided")
        pred = self.predict(points)
        exact = self.problem.exact_solution(points)
        return float(np.mean((pred.flatten() - exact.flatten()) ** 2))

    def __repr__(self):
        return f"NeuralPDESolver(problem={self.problem.name})"


# --- Pre-built PDE problem factories ---

def poisson_2d(source_fn: Optional[Callable] = None) -> PDEProblem:
    """Poisson equation: ∇²u = f(x,y) on [0,1]²."""
    def pde_fn(net, xy):
        xy = xy.requires_grad_(True)
        u = net(xy)
        ux = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        uxx = torch.autograd.grad(ux[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
        uyy = torch.autograd.grad(ux[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
        laplacian = uxx + uyy
        if source_fn is not None:
            f = source_fn(xy)
            return laplacian - f
        return laplacian

    # Zero Dirichlet BC
    n_bc = 100
    bc_pts, bc_vals = [], []
    for side in range(4):
        t = torch.linspace(0, 1, n_bc).unsqueeze(1)
        if side == 0: pts = torch.cat([t, torch.zeros(n_bc, 1)], 1)
        elif side == 1: pts = torch.cat([t, torch.ones(n_bc, 1)], 1)
        elif side == 2: pts = torch.cat([torch.zeros(n_bc, 1), t], 1)
        else: pts = torch.cat([torch.ones(n_bc, 1), t], 1)
        bc_pts.append(pts)
        bc_vals.append(torch.zeros(n_bc, 1))

    return PDEProblem(
        name="Poisson-2D",
        spatial_dim=2, has_time=False,
        domain={"x": (0, 1), "y": (0, 1)},
        pde_fn=pde_fn,
        boundary_conditions=[(torch.cat(bc_pts), torch.cat(bc_vals))],
    )


def heat_1d(alpha: float = 1.0, L: float = 1.0, T: float = 1.0) -> PDEProblem:
    """Heat equation: u_t = α u_xx on [0,L] × [0,T]."""
    def pde_fn(net, xt):
        xt = xt.requires_grad_(True)
        u = net(xt)
        g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        ux, ut = g[:, 0:1], g[:, 1:2]
        uxx = torch.autograd.grad(ux.sum(), xt, create_graph=True)[0][:, 0:1]
        return ut - alpha * uxx

    # BCs: u(0,t) = u(L,t) = 0
    n_bc = 100
    t_bc = torch.linspace(0, T, n_bc).unsqueeze(1)
    bc_left = torch.cat([torch.zeros(n_bc, 1), t_bc], 1)
    bc_right = torch.cat([torch.full((n_bc, 1), L), t_bc], 1)
    bc_pts = torch.cat([bc_left, bc_right])
    bc_vals = torch.zeros(2 * n_bc, 1)

    # IC: u(x,0) = sin(πx/L)
    x_ic = torch.linspace(0, L, 200).unsqueeze(1)
    ic_pts = torch.cat([x_ic, torch.zeros(200, 1)], 1)
    ic_vals = torch.sin(np.pi * x_ic / L)

    def exact(xt):
        x, t = xt[:, 0], xt[:, 1]
        return (np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi / L) ** 2 * t)).reshape(-1, 1)

    return PDEProblem(
        name="Heat-1D",
        spatial_dim=1, has_time=True,
        domain={"x": (0, L), "t": (0, T)},
        pde_fn=pde_fn,
        boundary_conditions=[(bc_pts, bc_vals)],
        initial_condition=(ic_pts, ic_vals),
        exact_solution=exact,
    )


def wave_1d(c: float = 1.0, L: float = 1.0, T: float = 2.0) -> PDEProblem:
    """Wave equation: u_tt = c² u_xx on [0,L] × [0,T]."""
    def pde_fn(net, xt):
        xt = xt.requires_grad_(True)
        u = net(xt)
        g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        ux, ut = g[:, 0:1], g[:, 1:2]
        uxx = torch.autograd.grad(ux.sum(), xt, create_graph=True)[0][:, 0:1]
        utt = torch.autograd.grad(ut.sum(), xt, create_graph=True)[0][:, 1:2]
        return utt - c ** 2 * uxx

    n_bc = 100
    t_bc = torch.linspace(0, T, n_bc).unsqueeze(1)
    bc_pts = torch.cat([
        torch.cat([torch.zeros(n_bc, 1), t_bc], 1),
        torch.cat([torch.full((n_bc, 1), L), t_bc], 1),
    ])
    bc_vals = torch.zeros(2 * n_bc, 1)

    x_ic = torch.linspace(0, L, 200).unsqueeze(1)
    ic_pts = torch.cat([x_ic, torch.zeros(200, 1)], 1)
    ic_vals = torch.sin(np.pi * x_ic / L)

    return PDEProblem(
        name="Wave-1D",
        spatial_dim=1, has_time=True,
        domain={"x": (0, L), "t": (0, T)},
        pde_fn=pde_fn,
        boundary_conditions=[(bc_pts, bc_vals)],
        initial_condition=(ic_pts, ic_vals),
    )


def burgers_1d(nu: float = 0.01, L: float = 2 * np.pi, T: float = 1.0) -> PDEProblem:
    """Burgers equation: u_t + u·u_x = ν u_xx."""
    def pde_fn(net, xt):
        xt = xt.requires_grad_(True)
        u = net(xt)
        g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        ux, ut = g[:, 0:1], g[:, 1:2]
        uxx = torch.autograd.grad(ux.sum(), xt, create_graph=True)[0][:, 0:1]
        return ut + u * ux - nu * uxx

    n_bc = 100
    t_bc = torch.linspace(0, T, n_bc).unsqueeze(1)
    bc_pts = torch.cat([
        torch.cat([torch.zeros(n_bc, 1), t_bc], 1),
        torch.cat([torch.full((n_bc, 1), L), t_bc], 1),
    ])
    bc_vals = torch.zeros(2 * n_bc, 1)

    x_ic = torch.linspace(0, L, 200).unsqueeze(1)
    ic_pts = torch.cat([x_ic, torch.zeros(200, 1)], 1)
    ic_vals = torch.sin(x_ic)

    return PDEProblem(
        name="Burgers-1D",
        spatial_dim=1, has_time=True,
        domain={"x": (0, L), "t": (0, T)},
        pde_fn=pde_fn,
        boundary_conditions=[(bc_pts, bc_vals)],
        initial_condition=(ic_pts, ic_vals),
    )

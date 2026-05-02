"""
mnn.advanced.chaos_simulation.learner
=======================================
Neural dynamics learner for chaotic systems.

Core idea: train a network f_θ(x) ≈ dx/dt.
Given the current state x(t), predict the derivative,
then integrate to forecast x(t + Δt).

Architecture options
--------------------
  DynamicsNet     — standard MNN (tanh, residual)  
  FourierDynNet   — Fourier-feature network (captures oscillatory dynamics)
  PhysicsNet      — physics-informed: penalise energy conservation violations
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from tqdm import tqdm
from .simulator import ChaosTrajectory


# ── Network architectures ─────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, width, act=nn.Tanh()):
        super().__init__()
        self.l = nn.Linear(width, width); self.n = nn.LayerNorm(width); self.act = act
    def forward(self, x): return self.act(self.n(self.l(x)) + x)


class DynamicsNet(nn.Module):
    """
    Neural network f_θ: Rⁿ → Rⁿ that learns dx/dt = f(x).
    Input: state x  |  Output: derivative dx/dt
    """
    def __init__(self, state_dim: int, width: int = 128, depth: int = 5,
                 activation: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        act  = acts.get(activation, nn.Tanh())
        self.state_dim = state_dim; self.width = width; self.depth = depth
        self.embed  = nn.Sequential(nn.Linear(state_dim, width), nn.LayerNorm(width), act.__class__())
        self.blocks = nn.ModuleList([ResBlock(width, act.__class__()) for _ in range(depth)])
        self.head   = nn.Linear(width, state_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight, 0.5); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, state_dim) → dx/dt: (N, state_dim)"""
        h = self.embed(x)
        for b in self.blocks: h = b(h)
        return self.head(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad(): return self.forward(torch.tensor(x, dtype=torch.float32)).numpy()

    def __repr__(self):
        return (f"DynamicsNet(dim={self.state_dim}, w={self.width}, "
                f"d={self.depth}, params={self.count_parameters():,})")


class FourierDynNet(nn.Module):
    """Dynamics network with random Fourier feature embedding — better for oscillatory systems."""
    def __init__(self, state_dim: int, n_fourier: int = 64, width: int = 128,
                 depth: int = 4, scale: float = 1.0):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer("B", torch.randn(state_dim, n_fourier) * scale)
        self.net = DynamicsNet(2 * n_fourier, width, depth)
        self.head = nn.Linear(self.net.state_dim, state_dim)

    def embed(self, x): xB = x @ self.B; return torch.cat([torch.sin(xB), torch.cos(xB)], -1)

    def forward(self, x):
        h = self.net.embed(self.embed(x))
        for b in self.net.blocks: h = b(h)
        return self.head(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, x):
        self.eval()
        with torch.no_grad(): return self.forward(torch.tensor(x, dtype=torch.float32)).numpy()


# ── Result container ──────────────────────────────────────────

@dataclass
class DynamicsResult:
    loss_history:  Dict[str, List[float]] = field(default_factory=dict)
    final_losses:  Dict[str, float]       = field(default_factory=dict)
    n_epochs:      int                    = 0
    n_params:      int                    = 0
    system_name:   str                    = ""
    state_dim:     int                    = 3

    def summary(self) -> str:
        w = 52
        rows = [
            "╔" + "═"*w + "╗",
            f"║  Neural Dynamics: {self.system_name:<{w-19}}║",
            "╠" + "═"*w + "╣",
            f"║  State dim    : {self.state_dim:<{w-17}}║",
            f"║  Parameters   : {self.n_params:<{w-17},}║",
            f"║  Epochs       : {self.n_epochs:<{w-17}}║",
            "╠" + "═"*w + "╣",
        ]
        for k, v in self.final_losses.items():
            rows.append(f"║  {k:<22}: {v:<{w-26}.8f}║")
        rows.append("╚" + "═"*w + "╝")
        return "\n".join(rows)


# ── Trainer ───────────────────────────────────────────────────

class DynamicsTrainer:
    """
    Trains a neural dynamics model f_θ(x) ≈ dx/dt on chaotic trajectory data.

    Loss components
    ---------------
    L_deriv   = ‖f_θ(x) − ẋ‖²          (derivative matching — primary)
    L_rollout = ‖integrate(f_θ, x₀) − x‖²  (trajectory rollout — optional)
    L_physics = physics constraint (optional, system-specific)
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.Adam(model.parameters(), lr=lr)
        self.sched  = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=2000, eta_min=lr*0.01)
        self.result = DynamicsResult(
            n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            state_dim=getattr(model, 'state_dim', 3)
        )
        self._extra: List[Tuple[str, Callable, float]] = []

    def add_physics_constraint(self, name: str, fn: Callable, weight: float = 1.0):
        """fn(model, x) → residual tensor. Added to loss."""
        self._extra.append((name, fn, weight))
        return self

    def _t(self, x): return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)

    def train(self, trajectory: ChaosTrajectory, n_epochs: int = 3000,
              batch_size: int = 512, verbose: bool = True,
              print_every: int = 500, w_deriv: float = 1.0,
              w_rollout: float = 0.0, rollout_steps: int = 5) -> DynamicsResult:
        """
        Train on a ChaosTrajectory.
        w_rollout > 0 adds rollout loss (expensive but more accurate for prediction).
        """
        X = self._t(trajectory.states)       # (N, dim)
        Y = self._t(trajectory.derivatives)  # (N, dim) = dx/dt

        N   = X.shape[0]
        itr = tqdm(range(n_epochs), desc=f"Dynamics [{trajectory.system_name}]") if verbose else range(n_epochs)

        for ep in itr:
            idx   = torch.randperm(N)[:min(batch_size, N)]
            x_b   = X[idx]; y_b = Y[idx]
            self.opt.zero_grad()
            losses = {}

            # Primary: derivative matching
            pred      = self.model(x_b)
            deriv_l   = w_deriv * nn.functional.mse_loss(pred, y_b)
            losses["deriv"] = float(deriv_l.detach())
            total     = deriv_l

            # Rollout loss: integrate forward k steps and compare
            if w_rollout > 0 and rollout_steps > 0:
                roll_l = self._rollout_loss(x_b, y_b, trajectory.dt, rollout_steps)
                total  = total + w_rollout * roll_l
                losses["rollout"] = float(roll_l.detach())

            # Extra physics constraints
            for cname, cfn, cw in self._extra:
                cl    = cw * cfn(self.model, x_b)
                total = total + cl
                losses[cname] = float(cl.detach())

            losses["total"] = float(total.detach())
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step(); self.sched.step()

            for k, v in losses.items():
                self.result.loss_history.setdefault(k, []).append(v)

            if verbose and (ep+1) % print_every == 0:
                ls = "  ".join(f"{k}={v:.5f}" for k,v in losses.items())
                tqdm.write(f"  [{ep+1:>5}]  {ls}")

        self.result.n_epochs    = n_epochs
        self.result.final_losses = {k: v[-1] for k,v in self.result.loss_history.items()}
        self.result.system_name  = trajectory.system_name
        return self.result

    def _rollout_loss(self, x0: torch.Tensor, true_deriv: torch.Tensor,
                      dt: float, steps: int) -> torch.Tensor:
        """Euler rollout and compare to ground truth."""
        x = x0.detach().clone()
        loss = torch.tensor(0.0, device=self.device)
        for _ in range(steps):
            dx = self.model(x)
            x  = x + dt * dx
        return nn.functional.mse_loss(x, x0 + dt * steps * true_deriv)

    def predict_derivative(self, states: np.ndarray) -> np.ndarray:
        return self.model.predict_numpy(states)

    def evaluate_derivative_error(self, trajectory: ChaosTrajectory) -> Dict[str, float]:
        pred = self.predict_derivative(trajectory.states)
        err  = pred - trajectory.derivatives
        return {
            "mse":           float(np.mean(err**2)),
            "rmse":          float(np.sqrt(np.mean(err**2))),
            "relative_err":  float(np.sqrt(np.mean(err**2)) / (np.std(trajectory.derivatives) + 1e-10)),
            "max_err":       float(np.max(np.abs(err))),
        }

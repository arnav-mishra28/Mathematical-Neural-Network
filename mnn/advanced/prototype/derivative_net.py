"""
mnn.advanced.prototype.derivative_net
=======================================
THE FIRST MNN PROTOTYPE — Derivative-Constrained Neural Network.

Goal:
  Learn f(x) such that  df/dx = 2x
  Expected solution:     f(x) = x² + C

This is the simplest example of embedding a mathematical constraint
(a first-order ODE / derivative condition) directly into neural training
via automatic differentiation.

Architecture:
  - MNNNetwork: 1D input → 1D output
  - Constraint loss:   L_constraint = mean( (f'(x) - 2x)² )
  - Anchor loss:       L_anchor    = (f(0) - C)²   [pins the constant]
  - Total loss:        L = w_c · L_constraint + w_a · L_anchor

After training:
  - f(x) ≈ x² + C
  - f'(x) ≈ 2x
  - The network has "discovered" the antiderivative of 2x

This prototype demonstrates the core MNN philosophy:
  Neural networks + Mathematical Constraints = Structured Learning
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm


# ── Network ───────────────────────────────────────────────────────────────────

class DerivativeConstrainedNet(nn.Module):
    """
    Compact MNN for the derivative-constrained prototype.
    Input: x ∈ R  →  Output: f(x) ∈ R

    Uses sin activations (smooth, infinite derivatives — ideal for ODE constraints).
    """
    def __init__(self, width: int = 64, depth: int = 4,
                 activation: str = "tanh"):
        super().__init__()
        acts = {
            "tanh":  torch.tanh,
            "sin":   torch.sin,
            "gelu":  nn.functional.gelu,
            "silu":  nn.functional.silu,
        }
        self.act   = acts.get(activation, torch.tanh)
        self.width = width
        self.depth = depth
        self.activation_name = activation

        # Build layers
        layers = [nn.Linear(1, width), nn.LayerNorm(width)]
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.LayerNorm(width)]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual forward with activations between linear layers
        h = x
        for i, layer in enumerate(self.net):
            h = layer(h)
            # Apply activation after each Linear except the last
            if isinstance(layer, nn.Linear) and i < len(self.net) - 1:
                h = self.act(h)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate on numpy array."""
        self.eval()
        with torch.no_grad():
            xt = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
            return self.forward(xt).numpy().flatten()

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute df/dx via autograd at numpy points."""
        self.eval()
        xt = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        fx = self.forward(xt)
        grad = torch.autograd.grad(fx.sum(), xt)[0]
        return grad.detach().numpy().flatten()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"DerivativeConstrainedNet(width={self.width}, depth={self.depth}, "
                f"act={self.activation_name}, params={self.count_parameters():,})")


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class PrototypeResult:
    """All outputs from the prototype training run."""
    network:           DerivativeConstrainedNet
    loss_history:      Dict[str, List[float]]
    x_train:           np.ndarray
    f_pred:            np.ndarray      # f(x) predicted
    df_pred:           np.ndarray      # f'(x) predicted
    f_exact:           np.ndarray      # x² + C
    df_exact:          np.ndarray      # 2x
    anchor_constant:   float           # the C in f(x) = x² + C
    final_losses:      Dict[str, float]
    mse_function:      float           # MSE between f_pred and x²+C
    mse_derivative:    float           # MSE between f'_pred and 2x
    convergence_epoch: Optional[int]   # epoch where derivative MSE < 1e-4

    def summary(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║   MNN PROTOTYPE — Derivative Constraint      ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Goal:      learn f s.t. df/dx = 2x         ║",
            f"║  Solution:  f(x) = x² + C                   ║",
            "╠══════════════════════════════════════════════╣",
            f"║  MSE  f(x) vs x²+C :  {self.mse_function:.2e}{'':>16}║",
            f"║  MSE f'(x) vs 2x   :  {self.mse_derivative:.2e}{'':>16}║",
            f"║  Anchor constant C :  {self.anchor_constant:.6f}{'':>13}║",
            f"║  Converged at epoch:  {str(self.convergence_epoch):<22}║",
            "╠══════════════════════════════════════════════╣",
            f"║  {self.network}  ║" if len(str(self.network)) < 44 else f"║  {str(self.network)[:42]}  ║",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ── Trainer ───────────────────────────────────────────────────────────────────

class DerivativeTrainer:
    """
    Trains DerivativeConstrainedNet with a combined loss:

      L_constraint(x) = mean( (df/dx - 2x)² )   ← enforce df/dx = 2x
      L_anchor        = (f(0) - C_target)²        ← pin f(0) to fix constant
      L_data(x)       = mean( (f(x) - x²-C)² )   ← optional: direct supervision

      L_total = w_c · L_constraint + w_a · L_anchor + w_d · L_data
    """

    def __init__(self, net: DerivativeConstrainedNet,
                 lr: float = 1e-3,
                 w_constraint: float = 1.0,
                 w_anchor:     float = 1.0,
                 w_data:       float = 0.0,   # 0 = pure constraint (no direct data)
                 anchor_x:     float = 0.0,   # pin f(anchor_x)
                 anchor_val:   float = 0.0,   # f(anchor_x) = anchor_val
                 device: str = "cpu"):
        self.net          = net.to(device)
        self.device       = device
        self.w_c          = w_constraint
        self.w_a          = w_anchor
        self.w_d          = w_data
        self.anchor_x     = torch.tensor([[anchor_x]], dtype=torch.float32)
        self.anchor_val   = torch.tensor([[anchor_val]], dtype=torch.float32)
        self.optimizer    = torch.optim.Adam(net.parameters(), lr=lr)
        self.scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
                                self.optimizer, T_max=2000, eta_min=1e-5)
        self.loss_history: Dict[str, List[float]] = {
            "constraint": [], "anchor": [], "data": [], "total": []
        }
        self.convergence_epoch: Optional[int] = None

    def _derivative_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L_constraint = mean( (df/dx(xᵢ) - 2xᵢ)² )."""
        x     = x.requires_grad_(True)
        fx    = self.net(x)
        # Compute df/dx via autograd
        dfdx  = torch.autograd.grad(
            outputs=fx.sum(),
            inputs=x,
            create_graph=True
        )[0]
        target = 2.0 * x          # the constraint: df/dx = 2x
        return torch.mean((dfdx - target) ** 2)

    def _anchor_loss(self) -> torch.Tensor:
        """Compute L_anchor = (f(anchor_x) - anchor_val)²."""
        f_at_anchor = self.net(self.anchor_x)
        return (f_at_anchor - self.anchor_val) ** 2

    def _data_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((self.net(x) - y) ** 2)

    def train(self,
              x_collocation: np.ndarray,
              n_epochs: int = 3000,
              x_data: Optional[np.ndarray] = None,
              y_data: Optional[np.ndarray] = None,
              verbose: bool = True,
              print_every: int = 200,
              tol_converge: float = 1e-4) -> Dict[str, List[float]]:
        """
        Train the network.

        Parameters
        ----------
        x_collocation : points where df/dx = 2x is enforced
        n_epochs       : number of training epochs
        x_data, y_data : optional direct supervision data
        tol_converge   : convergence threshold for derivative MSE
        """
        X_col = torch.tensor(x_collocation.reshape(-1, 1), dtype=torch.float32)
        has_data = (x_data is not None) and (y_data is not None)
        if has_data:
            X_d = torch.tensor(x_data.reshape(-1, 1), dtype=torch.float32)
            Y_d = torch.tensor(y_data.reshape(-1, 1), dtype=torch.float32)

        iterator = tqdm(range(n_epochs), desc="Prototype Training", ncols=80) if verbose else range(n_epochs)

        for epoch in iterator:
            self.optimizer.zero_grad()

            # ── Constraint loss: enforce df/dx = 2x ──────────────────────────
            L_c = self.w_c * self._derivative_loss(X_col)

            # ── Anchor loss: pin f(0) = 0 (so C = 0) ─────────────────────────
            L_a = self.w_a * self._anchor_loss().squeeze()

            # ── Optional data loss ────────────────────────────────────────────
            L_d = self.w_d * self._data_loss(X_d, Y_d) if has_data else torch.tensor(0.)

            L_total = L_c + L_a + L_d
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # ── Record ────────────────────────────────────────────────────────
            self.loss_history["constraint"].append(float(L_c))
            self.loss_history["anchor"].append(float(L_a))
            self.loss_history["data"].append(float(L_d))
            self.loss_history["total"].append(float(L_total))

            # ── Convergence check ─────────────────────────────────────────────
            if self.convergence_epoch is None and float(L_c) < tol_converge:
                self.convergence_epoch = epoch
                if verbose:
                    tqdm.write(f"  ✓ Derivative constraint converged at epoch {epoch}  (L_c={float(L_c):.2e})")

            if verbose and (epoch + 1) % print_every == 0:
                tqdm.write(
                    f"  [{epoch+1:>5}/{n_epochs}]  "
                    f"constraint={float(L_c):.4e}  "
                    f"anchor={float(L_a):.4e}  "
                    f"total={float(L_total):.4e}"
                )

        return self.loss_history

    def evaluate(self, x_eval: np.ndarray,
                  anchor_constant: float = 0.0) -> PrototypeResult:
        """
        Build a full PrototypeResult from the trained network.
        """
        self.net.eval()
        x   = x_eval.flatten()
        f_p = self.net.predict(x)
        df_p = self.net.derivative(x)

        C         = anchor_constant
        f_exact   = x**2 + C
        df_exact  = 2.0 * x

        mse_f  = float(np.mean((f_p - f_exact)**2))
        mse_df = float(np.mean((df_p - df_exact)**2))

        final = {k: v[-1] for k, v in self.loss_history.items() if v}

        return PrototypeResult(
            network           = self.net,
            loss_history      = dict(self.loss_history),
            x_train           = x,
            f_pred            = f_p,
            df_pred           = df_p,
            f_exact           = f_exact,
            df_exact          = df_exact,
            anchor_constant   = C,
            final_losses      = final,
            mse_function      = mse_f,
            mse_derivative    = mse_df,
            convergence_epoch = self.convergence_epoch,
        )


# ── One-shot runner ───────────────────────────────────────────────────────────

def run_prototype(
    x_range:      Tuple[float, float] = (-3.0, 3.0),
    n_colloc:     int   = 500,
    n_epochs:     int   = 3000,
    width:        int   = 64,
    depth:        int   = 4,
    lr:           float = 1e-3,
    w_constraint: float = 1.0,
    w_anchor:     float = 10.0,
    activation:   str   = "tanh",
    verbose:      bool  = True,
    seed:         int   = 42,
) -> PrototypeResult:
    """
    One-call function to run the full derivative-constrained prototype.

    Trains a network to satisfy df/dx = 2x, pinned at f(0) = 0.
    Expected solution: f(x) = x².

    Returns a PrototypeResult with full diagnostics.

    Example
    -------
    >>> from mnn.advanced.prototype import run_prototype
    >>> result = run_prototype(n_epochs=2000, verbose=True)
    >>> print(result.summary())
    >>> print(f"MSE f vs x²: {result.mse_function:.4e}")
    >>> print(f"MSE f' vs 2x: {result.mse_derivative:.4e}")
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Collocation points (where constraint is enforced) ─────────────────────
    x_col = np.linspace(x_range[0], x_range[1], n_colloc).astype(np.float32)

    # ── Build network & trainer ───────────────────────────────────────────────
    net     = DerivativeConstrainedNet(width=width, depth=depth, activation=activation)
    trainer = DerivativeTrainer(
        net,
        lr=lr,
        w_constraint=w_constraint,
        w_anchor=w_anchor,
        anchor_x=0.0,
        anchor_val=0.0,   # f(0) = 0  →  C = 0  →  f(x) = x²
    )

    if verbose:
        print("══════════════════════════════════════════════")
        print("  MNN Prototype — Derivative Constraint")
        print("  Goal:  learn f(x) s.t. df/dx = 2x")
        print("  Pin:   f(0) = 0  →  solution: f(x) = x²")
        print(f"  Network: {net}")
        print("══════════════════════════════════════════════")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train(x_col, n_epochs=n_epochs, verbose=verbose)

    # ── Evaluate on a fine grid ───────────────────────────────────────────────
    x_eval = np.linspace(x_range[0], x_range[1], 400).astype(np.float32)
    result = trainer.evaluate(x_eval, anchor_constant=0.0)

    if verbose:
        print()
        print(result.summary())

    return result

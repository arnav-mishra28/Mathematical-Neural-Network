"""
mnn.advanced.vector_calculus.trainer
======================================
Multi-constraint field trainer for MNN.

Handles:
  - Multiple simultaneous constraints with individual weights
  - Data supervision terms
  - Boundary conditions
  - Adaptive loss weighting
  - Training history and diagnostics
"""

from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
from tqdm import tqdm


@dataclass
class FieldTrainingResult:
    loss_history: Dict[str, List[float]] = field(default_factory=dict)
    final_losses: Dict[str, float]       = field(default_factory=dict)
    n_epochs:     int                    = 0
    n_params:     int                    = 0

    def summary(self) -> str:
        w = 52
        lines = [
            "╔" + "═"*w + "╗",
            f"║  MNN Field Training Result{'':<{w-27}}║",
            "╠" + "═"*w + "╣",
            f"║  Epochs   : {self.n_epochs:<{w-13}}║",
            f"║  Params   : {self.n_params:<{w-13},}║",
            "╠" + "═"*w + "╣",
        ]
        for k,v in self.final_losses.items():
            lines.append(f"║  {k:<20}: {v:<{w-24}.8f}║")
        lines.append("╚" + "═"*w + "╝")
        return "\n".join(lines)


class FieldTrainer:
    """
    Research-grade trainer for vector/tensor field networks with
    multiple simultaneous mathematical constraints.

    Loss structure
    --------------
    L_total = Σᵢ wᵢ · mean(residualᵢ²)
            + w_data · MSE(prediction, labels)      [optional]
            + w_bc   · boundary_loss()              [optional]
    """

    def __init__(self,
                 network:    nn.Module,
                 lr:         float = 1e-3,
                 optimizer:  str   = "adam",
                 device:     str   = "cpu"):
        self.network = network.to(device)
        self.device  = device
        self.result  = FieldTrainingResult(
            n_params=sum(p.numel() for p in network.parameters() if p.requires_grad)
        )
        opts = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        self.optimizer = opts.get(optimizer, torch.optim.Adam)(
            network.parameters(), lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr*1e-2
        )
        self._constraints: List[Tuple[str, Callable, float]] = []
        self._data_terms:  List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        self._bc_fns:      List[Tuple[Callable, float]] = []

    # ── Configuration ─────────────────────────────────────────────────────────

    def add_constraint(self, name: str,
                       residual_fn: Callable,
                       weight: float = 1.0) -> "FieldTrainer":
        """
        Add a mathematical constraint.

        Parameters
        ----------
        name        : label shown in training output
        residual_fn : callable(x) → residual_tensor (MSE of this is minimised)
        weight      : loss weight
        """
        self._constraints.append((name, residual_fn, weight))
        return self

    def add_data(self, x_data: np.ndarray, y_data: np.ndarray,
                 weight: float = 1.0) -> "FieldTrainer":
        """Add a supervised data term."""
        X = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        Y = torch.tensor(y_data, dtype=torch.float32, device=self.device)
        self._data_terms.append((X, Y, weight))
        return self

    def add_boundary_condition(self, bc_fn: Callable,
                                weight: float = 1.0) -> "FieldTrainer":
        """Add a boundary condition loss callable() → scalar loss."""
        self._bc_fns.append((bc_fn, weight))
        return self

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self,
              collocation_pts: np.ndarray,
              n_epochs:    int   = 3000,
              batch_size:  Optional[int] = None,
              verbose:     bool  = True,
              print_every: int   = 300) -> FieldTrainingResult:
        """
        Train with all registered constraints.

        Parameters
        ----------
        collocation_pts : (N, space_dim) interior collocation points
        n_epochs        : number of training epochs
        batch_size      : mini-batch size (None = full batch)
        """
        X_col = torch.tensor(collocation_pts, dtype=torch.float32, device=self.device)
        itr   = tqdm(range(n_epochs), desc="Field Training") if verbose else range(n_epochs)

        for ep in itr:
            # Sample mini-batch
            if batch_size and batch_size < X_col.shape[0]:
                idx   = torch.randperm(X_col.shape[0])[:batch_size]
                x_b   = X_col[idx]
            else:
                x_b   = X_col

            self.optimizer.zero_grad()
            losses = {}
            total  = torch.tensor(0.0, device=self.device)

            # ── Constraint losses ──────────────────────────────────────────
            for cname, cfn, cw in self._constraints:
                res    = cfn(x_b.detach().clone())
                closs  = cw * torch.mean(res ** 2)
                losses[cname] = float(closs.detach())
                total  = total + closs

            # ── Data losses ────────────────────────────────────────────────
            for Xd, Yd, dw in self._data_terms:
                pred   = self.network(Xd)
                dloss  = dw * nn.functional.mse_loss(pred, Yd)
                losses["data"] = float(dloss.detach())
                total  = total + dloss

            # ── Boundary losses ────────────────────────────────────────────
            for bc_fn, bw in self._bc_fns:
                bloss  = bw * bc_fn()
                losses["boundary"] = float(bloss.detach())
                total  = total + bloss

            losses["total"] = float(total.detach())
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Record
            for k, v in losses.items():
                self.result.loss_history.setdefault(k, []).append(v)

            if verbose and (ep+1) % print_every == 0:
                loss_str = "  ".join(f"{k}={v:.5f}" for k, v in losses.items())
                tqdm.write(f"  [{ep+1:>5}/{n_epochs}]  {loss_str}")

        self.result.n_epochs    = n_epochs
        self.result.final_losses = {k: v[-1] for k, v in self.result.loss_history.items()}
        return self.result

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=self.device)
            return self.network(xt).cpu().numpy()

    def compute_constraint_violation(self, x: np.ndarray) -> Dict[str, float]:
        """Evaluate each constraint residual RMS on test points."""
        xt  = torch.tensor(x, dtype=torch.float32, device=self.device)
        out = {}
        for cname, cfn, _ in self._constraints:
            with torch.enable_grad():
                res = cfn(xt.detach().clone().requires_grad_(True))
            out[cname + "_rms"] = float(torch.sqrt(torch.mean(res**2)).detach())
        return out

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))

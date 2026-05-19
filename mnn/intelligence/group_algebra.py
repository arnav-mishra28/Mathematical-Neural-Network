"""mnn.intelligence.group_algebra — Neural group operations, equivariant networks, invariant learning.

Pillar 2: Abstract Algebra Engine (Group Theory).
The MNN learns to represent algebraic operations as neural networks,
enforcing group axioms (closure, associativity, identity, inverse) and
the abelian commutativity constraint a+b = b+a as differentiable losses.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Callable, Optional, Dict, List, Tuple
from tqdm import tqdm

from mnn.neural.base_network import MNNNetwork


class NeuralGroupOperator(nn.Module):
    """Learn a group operation as a neural network with algebraic constraints.

    Given pairs (a, b), learns f(a, b) such that:
      - Closure: f(a,b) ∈ G
      - Associativity: f(f(a,b), c) = f(a, f(b,c))
      - Identity: f(e, a) = a
      - Inverse: f(a, a⁻¹) = e
      - (Optional) Commutativity: f(a,b) = f(b,a) for abelian groups
    """
    def __init__(self, element_dim: int, width: int = 128, depth: int = 4,
                 abelian: bool = False):
        super().__init__()
        self.element_dim = element_dim
        self.abelian = abelian
        self.op_net = MNNNetwork(2 * element_dim, element_dim, width=width, depth=depth)
        self.inv_net = MNNNetwork(element_dim, element_dim, width=width, depth=depth // 2 or 1)
        self.identity_param = nn.Parameter(torch.zeros(element_dim))

    def operate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.op_net(torch.cat([a, b], dim=-1))

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        return self.inv_net(a)

    def identity(self) -> torch.Tensor:
        return self.identity_param.unsqueeze(0)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.operate(a, b)

    def constraint_losses(self, elements: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = elements.shape[0]
        idx = torch.randperm(n)
        a, b = elements[:n//2], elements[idx[:n//2]]
        c = elements[torch.randperm(n)[:n//2]]

        e = self.identity().expand(a.shape[0], -1)
        losses = {}

        # Identity: f(e, a) ≈ a and f(a, e) ≈ a
        losses["identity"] = (
            nn.functional.mse_loss(self.operate(e, a), a)
            + nn.functional.mse_loss(self.operate(a, e), a)
        )

        # Inverse: f(a, inv(a)) ≈ e
        a_inv = self.inverse(a)
        losses["inverse"] = nn.functional.mse_loss(
            self.operate(a, a_inv), e)

        # Associativity: f(f(a,b), c) ≈ f(a, f(b,c))
        ab = self.operate(a, b)
        lhs = self.operate(ab, c)
        bc = self.operate(b, c)
        rhs = self.operate(a, bc)
        losses["associativity"] = nn.functional.mse_loss(lhs, rhs)

        # Commutativity (abelian only)
        if self.abelian:
            losses["commutativity"] = nn.functional.mse_loss(
                self.operate(a, b), self.operate(b, a))

        return losses


class NeuralGroupTrainer:
    """Train a NeuralGroupOperator on group data with algebraic constraint enforcement."""
    def __init__(self, operator: NeuralGroupOperator, lr: float = 1e-3, device: str = "cpu"):
        self.operator = operator.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
        self.history: Dict[str, List[float]] = {}

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(np.array(x), dtype=torch.float32).to(self.device)

    def train(self, elements: np.ndarray, targets: Optional[np.ndarray] = None,
              pairs: Optional[np.ndarray] = None, results: Optional[np.ndarray] = None,
              n_epochs: int = 2000, w_data: float = 1.0,
              w_identity: float = 1.0, w_inverse: float = 1.0,
              w_assoc: float = 2.0, w_comm: float = 1.0,
              verbose: bool = True, print_every: int = 200) -> Dict:
        E = self._t(elements)
        has_supervision = pairs is not None and results is not None

        if has_supervision:
            P = self._t(pairs)
            R = self._t(results)

        weight_map = {"identity": w_identity, "inverse": w_inverse,
                      "associativity": w_assoc, "commutativity": w_comm}
        it = tqdm(range(n_epochs), desc="Group Training") if verbose else range(n_epochs)

        for ep in it:
            self.optimizer.zero_grad()
            losses = {}

            constraints = self.operator.constraint_losses(E)
            total = torch.tensor(0.0, device=self.device)
            for name, loss in constraints.items():
                w = weight_map.get(name, 1.0)
                losses[name] = loss.item()
                total = total + w * loss

            if has_supervision:
                a_batch = P[:, :self.operator.element_dim]
                b_batch = P[:, self.operator.element_dim:]
                pred = self.operator.operate(a_batch, b_batch)
                data_loss = nn.functional.mse_loss(pred, R)
                losses["data"] = data_loss.item()
                total = total + w_data * data_loss

            losses["total"] = total.item()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.operator.parameters(), 1.0)
            self.optimizer.step()

            for k, v in losses.items():
                self.history.setdefault(k, []).append(v)

            if verbose and (ep + 1) % print_every == 0:
                tqdm.write(f"[{ep+1}] " + " | ".join(f"{k}={v:.5f}" for k, v in losses.items()))

        return self.history


class EquivariantNetwork(nn.Module):
    """Neural network equivariant under a group action.

    f(g · x) = g · f(x)  for all group elements g.

    Uses group averaging: f_eq(x) = (1/|G|) Σ_g g⁻¹ · f_raw(g · x)
    """
    def __init__(self, input_dim: int, output_dim: int, group_actions: List[torch.Tensor],
                 width: int = 128, depth: int = 3):
        super().__init__()
        self.raw_net = MNNNetwork(input_dim, output_dim, width=width, depth=depth)
        self.register_buffer("actions", torch.stack(group_actions))
        self.n_group = len(group_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        result = torch.zeros(batch, self.raw_net.output_dim, device=x.device)
        for i in range(self.n_group):
            g = self.actions[i]
            g_inv = torch.linalg.inv(g)
            gx = (g @ x.unsqueeze(-1)).squeeze(-1)
            f_gx = self.raw_net(gx)
            result = result + (g_inv @ f_gx.unsqueeze(-1)).squeeze(-1)
        return result / self.n_group

    def equivariance_error(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            fx = self.forward(x)
            errors = []
            for i in range(self.n_group):
                g = self.actions[i]
                gx = (g @ x.unsqueeze(-1)).squeeze(-1)
                f_gx = self.forward(gx)
                g_fx = (g @ fx.unsqueeze(-1)).squeeze(-1)
                errors.append(nn.functional.mse_loss(f_gx, g_fx).item())
            return float(np.mean(errors))


class InvariantLearner:
    """Learn invariant quantities I(x) such that I(g · x) = I(x) for all g ∈ G.

    Uses a neural network with an invariance penalty:
      L_inv = Σ_g ||I(g·x) - I(x)||²
    """
    def __init__(self, input_dim: int, invariant_dim: int = 1,
                 group_actions: Optional[List[np.ndarray]] = None,
                 width: int = 64, depth: int = 3, lr: float = 1e-3,
                 device: str = "cpu"):
        self.device = device
        self.net = MNNNetwork(input_dim, invariant_dim, width=width, depth=depth).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        if group_actions is not None:
            self.actions = [torch.tensor(g, dtype=torch.float32).to(device) for g in group_actions]
        else:
            self.actions = []

        self.history: List[float] = []

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(np.array(x), dtype=torch.float32).to(self.device)

    def train(self, data: np.ndarray, targets: Optional[np.ndarray] = None,
              n_epochs: int = 1000, w_invariance: float = 10.0,
              w_data: float = 1.0, verbose: bool = True,
              print_every: int = 200) -> List[float]:
        X = self._t(data)
        has_targets = targets is not None
        if has_targets:
            Y = self._t(targets)

        it = tqdm(range(n_epochs), desc="Invariant Learning") if verbose else range(n_epochs)
        for ep in it:
            self.optimizer.zero_grad()
            pred = self.net(X)
            loss = torch.tensor(0.0, device=self.device)

            if has_targets:
                loss = loss + w_data * nn.functional.mse_loss(pred, Y)

            # Invariance loss
            for g in self.actions:
                gx = (g @ X.unsqueeze(-1)).squeeze(-1)
                pred_gx = self.net(gx)
                loss = loss + w_invariance * nn.functional.mse_loss(pred_gx, pred)

            loss.backward()
            self.optimizer.step()
            self.history.append(loss.item())

            if verbose and (ep + 1) % print_every == 0:
                tqdm.write(f"[{ep+1}] loss={loss.item():.6f}")

        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(self._t(x)).cpu().numpy()

    def __repr__(self):
        return f"InvariantLearner(actions={len(self.actions)})"

"""mnn.intelligence.dynamical — Flow-map learning, stability analysis, bifurcation detection.

Pillar 1: Dynamical + Nonlinear Systems.
Instead of learning dx/dt = f(x), we learn the discrete flow map x(t+Δt) = F(x(t)).
This is more stable, captures long-term dynamics better, and aligns with Neural ODE thinking.
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Dict, List, Tuple
from tqdm import tqdm

from mnn.neural.base_network import MNNNetwork


class FlowMapNetwork(nn.Module):
    """Neural network that learns the discrete-time flow map F: x(t) → x(t+Δt)."""
    def __init__(self, state_dim: int, width: int = 128, depth: int = 4,
                 dt: float = 0.01, residual_form: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.residual_form = residual_form
        self.net = MNNNetwork(state_dim, state_dim, width=width, depth=depth, activation="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_form:
            return x + self.dt * self.net(x)
        return self.net(x)

    def multi_step(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        trajectory = [x]
        for _ in range(n_steps):
            x = self.forward(x)
            trajectory.append(x)
        return torch.stack(trajectory, dim=1)


class FlowMapLearner:
    """Learn the flow map from trajectory data.

    Given trajectory data {x(t_i)}, train a neural network to predict
    x(t + Δt) = F(x(t)) instead of learning the derivative dx/dt.
    """
    def __init__(self, state_dim: int, width: int = 128, depth: int = 4,
                 dt: float = 0.01, residual: bool = True, lr: float = 1e-3,
                 device: str = "cpu"):
        self.device = device
        self.dt = dt
        self.flow_net = FlowMapNetwork(state_dim, width, depth, dt, residual).to(device)
        self.optimizer = torch.optim.Adam(self.flow_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=200, factor=0.5, min_lr=1e-6)
        self.history: Dict[str, List[float]] = {"loss": [], "multi_step_loss": []}

    @staticmethod
    def generate_training_data(ode_fn: Callable, x0: np.ndarray,
                                t_span: Tuple[float, float], dt: float = 0.01
                                ) -> Tuple[np.ndarray, np.ndarray]:
        t_eval = np.arange(t_span[0], t_span[1], dt)
        sol = solve_ivp(ode_fn, t_span, x0, t_eval=t_eval, method="RK45",
                        rtol=1e-10, atol=1e-12)
        traj = sol.y.T
        return traj[:-1], traj[1:]

    def _t(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(np.array(x), dtype=torch.float32).to(self.device)

    def train(self, x_now: np.ndarray, x_next: np.ndarray,
              n_epochs: int = 2000, batch_size: int = 256,
              multi_step_weight: float = 0.1, multi_step_horizon: int = 5,
              verbose: bool = True, print_every: int = 200) -> Dict:
        X = self._t(x_now)
        Y = self._t(x_next)
        loss_fn = nn.MSELoss()
        it = tqdm(range(n_epochs), desc="Flow Map Training") if verbose else range(n_epochs)

        for ep in it:
            self.optimizer.zero_grad()
            idx = torch.randperm(X.shape[0])[:batch_size]
            xb, yb = X[idx], Y[idx]
            pred = self.flow_net(xb)
            loss_1step = loss_fn(pred, yb)
            loss = loss_1step

            if multi_step_weight > 0 and multi_step_horizon > 1:
                ms_idx = torch.randperm(X.shape[0] - multi_step_horizon)[:batch_size // 2]
                ms_loss = torch.tensor(0.0, device=self.device)
                x_cur = X[ms_idx]
                for s in range(1, multi_step_horizon + 1):
                    x_cur = self.flow_net(x_cur)
                    target_idx = ms_idx + s
                    valid = target_idx < X.shape[0]
                    if valid.sum() > 0:
                        ms_loss = ms_loss + loss_fn(x_cur[valid], X[target_idx[valid]])
                loss = loss + multi_step_weight * ms_loss / multi_step_horizon

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_net.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(loss.item())

            self.history["loss"].append(loss_1step.item())
            self.history["multi_step_loss"].append(loss.item())

            if verbose and (ep + 1) % print_every == 0:
                tqdm.write(f"[{ep+1}] 1-step={loss_1step.item():.6f} total={loss.item():.6f}")

        return self.history

    def predict(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        self.flow_net.eval()
        with torch.no_grad():
            x = self._t(x0).unsqueeze(0)
            traj = self.flow_net.multi_step(x, n_steps)
            return traj.squeeze(0).cpu().numpy()

    def __repr__(self):
        return f"FlowMapLearner(dim={self.flow_net.state_dim}, dt={self.dt})"


class StabilityAnalyzer:
    """Analyze stability of learned or analytical dynamical systems.

    Computes Jacobians, eigenvalues, and classifies fixed points.
    Can work with both analytical ODE functions and learned flow maps.
    """
    @staticmethod
    def numerical_jacobian(f: Callable, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
        n = len(x)
        J = np.zeros((n, n))
        f0 = np.array(f(x), dtype=float)
        for j in range(n):
            xp = x.copy(); xp[j] += h
            xm = x.copy(); xm[j] -= h
            J[:, j] = (np.array(f(xp)) - np.array(f(xm))) / (2 * h)
        return J

    @staticmethod
    def classify_fixed_point(eigenvalues: np.ndarray) -> Dict:
        re = np.real(eigenvalues)
        im = np.imag(eigenvalues)
        has_complex = np.any(np.abs(im) > 1e-10)

        if np.all(re < -1e-10):
            kind = "stable focus" if has_complex else "stable node"
        elif np.all(re > 1e-10):
            kind = "unstable focus" if has_complex else "unstable node"
        elif np.any(re < -1e-10) and np.any(re > 1e-10):
            kind = "saddle"
        else:
            kind = "center" if has_complex else "non-hyperbolic"

        return {
            "eigenvalues": eigenvalues,
            "type": kind,
            "stable": np.all(re < 0),
            "hyperbolic": np.all(np.abs(re) > 1e-10),
            "dim_stable_manifold": int(np.sum(re < 0)),
            "dim_unstable_manifold": int(np.sum(re > 0)),
        }

    @staticmethod
    def find_fixed_points(rhs: Callable, dim: int, search_range: float = 5.0,
                          n_starts: int = 50) -> List[np.ndarray]:
        from scipy.optimize import fsolve
        fps = []
        rng = np.random.default_rng(42)
        for _ in range(n_starts):
            x0 = rng.uniform(-search_range, search_range, dim)
            sol = fsolve(lambda x: np.array(rhs(0, x), dtype=float), x0, full_output=True)
            x_star, info, ier, _ = sol
            if ier == 1 and np.linalg.norm(info["fvec"]) < 1e-8:
                is_new = all(np.linalg.norm(x_star - fp) > 1e-4 for fp in fps)
                if is_new:
                    fps.append(x_star)
        return fps

    @staticmethod
    def analyze_system(rhs: Callable, dim: int, **kwargs) -> List[Dict]:
        fps = StabilityAnalyzer.find_fixed_points(rhs, dim, **kwargs)
        results = []
        for fp in fps:
            J = StabilityAnalyzer.numerical_jacobian(lambda x: np.array(rhs(0, x), dtype=float), fp)
            evals = np.linalg.eig(J)[0]
            info = StabilityAnalyzer.classify_fixed_point(evals)
            info["fixed_point"] = fp
            info["jacobian"] = J
            results.append(info)
        return results

    def __repr__(self):
        return "StabilityAnalyzer()"


class BifurcationDetector:
    """Detect bifurcations by sweeping a parameter and tracking fixed-point stability.

    Supports saddle-node, pitchfork, Hopf, and period-doubling bifurcations.
    """
    @staticmethod
    def parameter_sweep(rhs_factory: Callable, param_range: Tuple[float, float],
                        dim: int, n_params: int = 200, x0: Optional[np.ndarray] = None,
                        t_transient: float = 50.0, t_record: float = 20.0,
                        dt: float = 0.01) -> Dict:
        params = np.linspace(*param_range, n_params)
        all_params = []
        all_attractors = []
        stability_changes = []
        prev_stable_count = None

        for p in params:
            rhs = rhs_factory(p)
            init = x0 if x0 is not None else np.random.randn(dim) * 0.1
            try:
                t_eval = np.arange(0, t_transient + t_record, dt)
                sol = solve_ivp(rhs, (0, t_transient + t_record), init,
                                t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
                traj = sol.y.T
                record = traj[int(t_transient / dt):]
                for pt in record[::max(1, len(record) // 50)]:
                    all_params.append(p)
                    all_attractors.append(pt[0] if dim > 0 else pt)
            except Exception:
                continue

            fps = StabilityAnalyzer.find_fixed_points(rhs, dim, n_starts=20)
            n_stable = sum(1 for fp in fps
                           if np.all(np.real(np.linalg.eig(
                               StabilityAnalyzer.numerical_jacobian(
                                   lambda x, r=rhs: np.array(r(0, x), dtype=float), fp))[0]) < 0))

            if prev_stable_count is not None and n_stable != prev_stable_count:
                stability_changes.append({"parameter": float(p), "stable_count_change": (prev_stable_count, n_stable)})
            prev_stable_count = n_stable

        return {
            "params": np.array(all_params),
            "attractors": np.array(all_attractors),
            "bifurcation_candidates": stability_changes,
        }

    @staticmethod
    def detect_hopf(rhs_factory: Callable, param_range: Tuple[float, float],
                    dim: int, fp_guess: np.ndarray, n_params: int = 200) -> List[Dict]:
        from scipy.optimize import fsolve
        params = np.linspace(*param_range, n_params)
        hopf_points = []
        prev_evals = None

        for p in params:
            rhs = rhs_factory(p)
            try:
                sol = fsolve(lambda x: np.array(rhs(0, x), dtype=float), fp_guess, full_output=True)
                fp = sol[0]
                if np.linalg.norm(sol[1]["fvec"]) > 1e-6:
                    continue
                J = StabilityAnalyzer.numerical_jacobian(
                    lambda x, r=rhs: np.array(r(0, x), dtype=float), fp)
                evals = np.linalg.eig(J)[0]
                idx = np.argsort(-np.abs(np.imag(evals)))
                evals = evals[idx]

                if prev_evals is not None:
                    for i in range(min(len(evals), len(prev_evals))):
                        if (np.real(prev_evals[i]) < 0 < np.real(evals[i])
                                and np.abs(np.imag(evals[i])) > 1e-6):
                            hopf_points.append({
                                "parameter": float(p),
                                "fixed_point": fp.copy(),
                                "critical_eigenvalue": evals[i],
                                "frequency": float(np.abs(np.imag(evals[i]))),
                            })
                prev_evals = evals
                fp_guess = fp
            except Exception:
                continue

        return hopf_points

    def __repr__(self):
        return "BifurcationDetector()"

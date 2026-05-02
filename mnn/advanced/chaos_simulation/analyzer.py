"""
mnn.advanced.chaos_simulation.analyzer
=========================================
Chaos diagnostics for neural dynamics models.

Computes:
  - Lyapunov exponent from learned dynamics (via Jacobian of network)
  - Predictability horizon vs. true Lyapunov
  - Phase space reconstruction (Takens embedding)
  - Strange attractor dimension from neural model
  - Butterfly effect demonstration
  - Comparison: neural vs. ground-truth dynamics
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .simulator import ChaosSimulator, ChaosTrajectory
from .predictor import ChaosPredictor


class ChaosNeuralAnalyzer:
    """
    Diagnostic tools for evaluating learned chaos models.
    """

    def __init__(self, model: nn.Module, state_dim: int, dt: float = 0.01):
        self.model     = model
        self.state_dim = state_dim
        self.dt        = dt
        self.predictor = ChaosPredictor(model, state_dim, dt)

    # ── Jacobian of learned dynamics ──────────────────────────

    def jacobian_at(self, x: np.ndarray) -> np.ndarray:
        """
        Compute J_ij = ∂f_i(x)/∂x_j of the learned dynamics via autograd.
        Returns (state_dim, state_dim) Jacobian matrix.
        """
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        J  = torch.zeros(self.state_dim, self.state_dim)
        for i in range(self.state_dim):
            # Fresh leaf tensor for each row to avoid grad-on-non-leaf issues
            xi = xt.detach().clone().requires_grad_(True)
            fi = self.model(xi)[0, i]
            fi.backward()
            if xi.grad is not None:
                J[i] = xi.grad[0].detach()
        return J.numpy()

    # ── Lyapunov exponent from neural model ───────────────────

    def neural_lyapunov_exponent(self, x0: np.ndarray,
                                  n_steps: int = 2000,
                                  renorm_every: int = 50) -> float:
        """
        Estimate largest Lyapunov exponent from the learned dynamics
        using the QR renormalisation method.

        λ ≈ (1/T) Σ log ‖J(x(t))·v‖   after renormalisation.
        """
        x   = x0.copy().astype(np.float64)
        v   = np.random.randn(self.state_dim)
        v   = v / np.linalg.norm(v)
        lam = 0.0
        count = 0

        for step in range(n_steps):
            # Advance state
            f   = self.predictor._f(x)
            x   = x + self.dt * f

            # Advance tangent vector via Jacobian
            J   = self.jacobian_at(x)
            v   = J @ v

            if (step + 1) % renorm_every == 0:
                norm  = np.linalg.norm(v)
                if norm > 1e-15:
                    lam  += np.log(norm)
                    v    /= norm
                    count += 1

        if count == 0: return 0.0
        return float(lam / (count * renorm_every * self.dt))

    # ── Butterfly effect ──────────────────────────────────────

    def butterfly_effect(self, x0: np.ndarray,
                          epsilon: float = 1e-6,
                          n_steps: int = 1000,
                          n_perturbations: int = 5) -> Dict:
        """
        Demonstrate butterfly effect: tiny perturbations → exponential divergence.

        Returns arrays showing how pairs of trajectories diverge.
        """
        ref  = self.predictor.predict_rk4(x0, n_steps)
        divs = []
        for _ in range(n_perturbations):
            direction = np.random.randn(self.state_dim)
            direction /= np.linalg.norm(direction)
            x0_p   = x0 + epsilon * direction
            pert   = self.predictor.predict_rk4(x0_p, n_steps)
            div    = np.linalg.norm(ref - pert, axis=-1)
            divs.append(div)

        divs_arr = np.stack(divs)    # (n_pert, n_steps+1)
        t_arr    = np.arange(n_steps + 1) * self.dt

        # Fit exponential: div ≈ epsilon * exp(λ t)
        mean_div = divs_arr.mean(axis=0)
        lam_fit  = 0.0
        n_fit    = min(n_steps // 4, 100)
        if n_fit > 5:
            t_fit = t_arr[:n_fit]
            d_fit = np.maximum(mean_div[:n_fit], epsilon)
            valid = d_fit > epsilon * 0.01
            if valid.sum() > 3:
                slope, _ = np.polyfit(t_fit[valid], np.log(d_fit[valid]), 1)
                lam_fit  = float(slope)

        return {
            "reference":     ref,
            "divergences":   divs_arr,
            "mean_div":      mean_div,
            "times":         t_arr,
            "epsilon":       epsilon,
            "lambda_fit":    lam_fit,
            "doubling_time": float(np.log(2) / (lam_fit + 1e-15)),
        }

    # ── Phase space reconstruction (Takens embedding) ─────────

    @staticmethod
    def takens_embedding(signal: np.ndarray,
                          dim: int = 3,
                          lag: int = 10) -> np.ndarray:
        """
        Takens delay embedding theorem:
        Reconstruct attractor from a single time series.
        x̃(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(d-1)τ)]

        Parameters
        ----------
        signal : 1D time series
        dim    : embedding dimension
        lag    : delay τ (in steps)

        Returns : (N', dim) embedded time series
        """
        N   = len(signal) - (dim - 1) * lag
        emb = np.zeros((N, dim))
        for i in range(dim):
            emb[:, i] = signal[i*lag : i*lag + N]
        return emb

    # ── Neural vs. ground-truth comparison ────────────────────

    def compare_to_ground_truth(self, simulator: ChaosSimulator,
                                  x0:        np.ndarray,
                                  t_end:     float = 5.0,
                                  dt:        float = None) -> Dict:
        """
        Compare neural prediction to ground-truth ODE integration.

        Returns comprehensive error metrics and trajectory arrays.
        """
        dt_ = dt or self.dt
        # Ground truth
        gt_traj  = simulator.simulate(x0, (0, t_end), dt_)
        gt_states = gt_traj.states

        # Neural prediction (RK4)
        n_steps  = len(gt_states) - 1
        nn_states = self.predictor.predict_rk4(x0, n_steps, dt_)

        # Per-step errors
        errors   = np.linalg.norm(nn_states - gt_states, axis=-1)

        # Find predictability horizon (error > 10% of attractor size)
        attractor_scale = float(np.std(gt_states))
        threshold       = 0.1 * attractor_scale
        above           = np.where(errors > threshold)[0]
        horizon_steps   = int(above[0]) if len(above) > 0 else n_steps
        horizon_time    = horizon_steps * dt_

        # Derivative error
        nn_derivs    = self.model.predict_numpy(gt_states[:50])
        gt_derivs    = gt_traj.derivatives[:50]
        deriv_rmse   = float(np.sqrt(np.mean((nn_derivs - gt_derivs)**2)))

        return {
            "nn_trajectory":    nn_states,
            "gt_trajectory":    gt_states,
            "errors":           errors,
            "horizon_time":     horizon_time,
            "horizon_steps":    horizon_steps,
            "deriv_rmse":       deriv_rmse,
            "attractor_scale":  attractor_scale,
            "max_reliable_time":horizon_time,
            "mean_error_early": float(errors[:horizon_steps//2].mean()) if horizon_steps > 2 else float(errors.mean()),
        }

    # ── Attractor dimension from neural model ─────────────────

    def neural_attractor_dimension(self, x0: np.ndarray,
                                    n_steps: int = 5000,
                                    n_scales: int = 12) -> float:
        """
        Estimate correlation dimension of the neural attractor
        using box-counting on the predicted trajectory.
        """
        traj  = self.predictor.predict_rk4(x0, n_steps)
        pts   = traj[::5]    # subsample
        mins  = pts.min(axis=0); maxs = pts.max(axis=0)
        span  = (maxs - mins).max()
        if span < 1e-10: return 0.0

        scales = np.logspace(-2, 0, n_scales) * span
        counts = []
        for eps in scales:
            boxes = set()
            for p in pts:
                box = tuple(int((p[d]-mins[d])/eps) for d in range(pts.shape[1]))
                boxes.add(box)
            counts.append(len(boxes))

        counts = np.array(counts, dtype=float)
        valid  = counts > 0
        if valid.sum() < 3: return float(pts.shape[1])
        slope, _ = np.polyfit(np.log(1/scales[valid]), np.log(counts[valid]), 1)
        return float(slope)

    # ── Full diagnostics report ───────────────────────────────

    def full_diagnostics(self, x0: np.ndarray,
                          simulator: Optional[ChaosSimulator] = None,
                          n_steps_lyap: int = 1000) -> Dict:
        """Generate a comprehensive chaos diagnostics report."""
        report = {}

        # Lyapunov from neural model
        report["neural_lyapunov"] = self.neural_lyapunov_exponent(
            x0, n_steps=n_steps_lyap, renorm_every=20
        )

        # Butterfly effect
        bf = self.butterfly_effect(x0, epsilon=1e-6, n_steps=500, n_perturbations=3)
        report["lambda_butterfly"] = bf["lambda_fit"]
        report["doubling_time"]    = bf["doubling_time"]

        # Attractor dimension
        report["neural_attractor_dim"] = self.neural_attractor_dimension(x0, n_steps=2000)

        # Ground truth comparison if simulator provided
        if simulator is not None:
            gt = self.compare_to_ground_truth(simulator, x0, t_end=3.0)
            report["predictability_horizon"] = gt["horizon_time"]
            report["deriv_rmse"]             = gt["deriv_rmse"]
            report["attractor_scale"]        = gt["attractor_scale"]

        return report

    def __repr__(self):
        return f"ChaosNeuralAnalyzer(dim={self.state_dim}, dt={self.dt})"

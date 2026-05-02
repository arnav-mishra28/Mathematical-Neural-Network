"""
mnn.advanced.chaos_simulation.predictor
=========================================
Short-term trajectory predictor for chaotic systems.

Key insight: chaos = exponential error growth.
  - Short-term: reliable (Lyapunov time-scale)
  - Long-term:  unreliable (butterfly effect)

The predictor integrates the learned dynamics forward,
tracks error growth, and estimates the predictability horizon.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from typing import Optional, List, Tuple, Dict
from .simulator import ChaosTrajectory


class ChaosPredictor:
    """
    Integrates a learned dynamics model to predict trajectories.

    Methods
    -------
    predict_euler      — simple Euler integration
    predict_rk4        — 4th-order Runge-Kutta integration (recommended)
    predict_adaptive   — adaptive step-size using scipy's solve_ivp
    predictability_horizon — estimate how long predictions stay accurate
    """

    def __init__(self, model: nn.Module, state_dim: int, dt: float = 0.01):
        self.model     = model
        self.state_dim = state_dim
        self.dt        = dt

    def _f(self, x: np.ndarray) -> np.ndarray:
        """Evaluate learned dynamics at state x."""
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32)
            if xt.dim() == 1: xt = xt.unsqueeze(0)
            return self.model(xt).squeeze(0).numpy()

    # ── Integration methods ───────────────────────────────────

    def predict_euler(self, x0: np.ndarray, n_steps: int,
                      dt: Optional[float] = None) -> np.ndarray:
        """
        Euler integration: x_{n+1} = x_n + dt·f(x_n)
        Fast but less accurate. Use for quick diagnostics.
        """
        dt_  = dt or self.dt
        traj = np.zeros((n_steps + 1, self.state_dim), dtype=np.float32)
        traj[0] = x0
        for i in range(n_steps):
            traj[i+1] = traj[i] + dt_ * self._f(traj[i])
        return traj

    def predict_rk4(self, x0: np.ndarray, n_steps: int,
                    dt: Optional[float] = None) -> np.ndarray:
        """
        4th-order Runge-Kutta: more accurate than Euler.
        Recommended for chaotic systems to extend predictability.
        """
        dt_  = dt or self.dt
        traj = np.zeros((n_steps + 1, self.state_dim), dtype=np.float32)
        traj[0] = x0
        for i in range(n_steps):
            x  = traj[i].astype(np.float64)
            k1 = self._f(x)
            k2 = self._f(x + 0.5 * dt_ * k1)
            k3 = self._f(x + 0.5 * dt_ * k2)
            k4 = self._f(x + dt_ * k3)
            traj[i+1] = (x + (dt_/6) * (k1 + 2*k2 + 2*k3 + k4)).astype(np.float32)
        return traj

    def predict_adaptive(self, x0: np.ndarray, t_end: float,
                          dt_out: Optional[float] = None) -> np.ndarray:
        """
        Adaptive step-size integration via scipy's solve_ivp.
        Most accurate but slower.
        """
        dt_out  = dt_out or self.dt
        t_eval  = np.arange(0, t_end, dt_out)

        def ode(t, x):
            return self._f(x).tolist()

        sol = solve_ivp(ode, (0, t_end), x0.tolist(), t_eval=t_eval,
                        method="RK45", rtol=1e-6, atol=1e-9)
        return sol.y.T.astype(np.float32)

    # ── Predictability analysis ───────────────────────────────

    def prediction_error(self, x0: np.ndarray, true_traj: np.ndarray,
                          method: str = "rk4",
                          dt: Optional[float] = None) -> np.ndarray:
        """
        Compute step-by-step prediction error ‖x_pred(t) − x_true(t)‖.
        Returns (n_steps,) error array.
        """
        n_steps = len(true_traj) - 1
        if method == "rk4":
            pred = self.predict_rk4(x0, n_steps, dt)
        else:
            pred = self.predict_euler(x0, n_steps, dt)
        errors = np.linalg.norm(pred - true_traj, axis=-1)
        return errors.astype(np.float32)

    def predictability_horizon(self, x0: np.ndarray,
                                 true_traj: np.ndarray,
                                 error_threshold: float = 1.0,
                                 dt: Optional[float] = None) -> Dict:
        """
        Estimate the predictability horizon:
        the time at which ‖x_pred − x_true‖ > threshold.

        Returns dict with:
          horizon_steps  : step index where error exceeds threshold
          horizon_time   : corresponding time
          errors         : full error array
          lyapunov_fit   : estimated Lyapunov exponent from error growth
        """
        dt_  = dt or self.dt
        errs = self.prediction_error(x0, true_traj, method="rk4", dt=dt_)
        above = np.where(errs > error_threshold)[0]
        horizon_steps = int(above[0]) if len(above) > 0 else len(errs)
        horizon_time  = horizon_steps * dt_

        # Fit exponential to early error growth: err ≈ ε₀ · exp(λt)
        n_fit = min(horizon_steps, len(errs) // 3)
        lam   = 0.0
        if n_fit > 5:
            t_fit  = np.arange(n_fit) * dt_
            e_fit  = np.maximum(errs[:n_fit], 1e-15)
            valid  = e_fit > 1e-12
            if valid.sum() > 3:
                slope, _ = np.polyfit(t_fit[valid], np.log(e_fit[valid]), 1)
                lam = float(slope)

        return {
            "horizon_steps":   horizon_steps,
            "horizon_time":    horizon_time,
            "errors":          errs,
            "lyapunov_fit":    lam,
            "threshold":       error_threshold,
        }

    def multi_step_forecast(self, x0: np.ndarray, n_steps: int,
                             n_restarts: int = 5,
                             noise_std: float = 1e-4) -> Dict:
        """
        Ensemble forecast: run n_restarts trajectories from slightly perturbed IC.
        Quantifies uncertainty from the butterfly effect.
        """
        forecasts = []
        for _ in range(n_restarts):
            ic = x0 + np.random.randn(self.state_dim) * noise_std
            forecasts.append(self.predict_rk4(ic, n_steps))
        forecasts = np.stack(forecasts)   # (n_restarts, n_steps+1, dim)
        return {
            "mean":     forecasts.mean(axis=0),
            "std":      forecasts.std(axis=0),
            "min":      forecasts.min(axis=0),
            "max":      forecasts.max(axis=0),
            "forecasts":forecasts,
        }


class EnsemblePredictor:
    """
    Ensemble of multiple trained dynamics models.
    Averages predictions to reduce overfitting and quantify uncertainty.
    """
    def __init__(self, models: List[nn.Module], state_dim: int, dt: float = 0.01):
        self.predictors = [ChaosPredictor(m, state_dim, dt) for m in models]
        self.state_dim  = state_dim
        self.dt         = dt

    def predict_mean(self, x0: np.ndarray, n_steps: int) -> Dict:
        """Predict with ensemble mean ± std."""
        preds = np.stack([p.predict_rk4(x0, n_steps) for p in self.predictors])
        return {
            "mean": preds.mean(0),
            "std":  preds.std(0),
            "preds":preds,
        }

    def predict_derivative_ensemble(self, x: np.ndarray) -> Dict:
        derivs = np.stack([p._f(x) for p in self.predictors])
        return {
            "mean": derivs.mean(0),
            "std":  derivs.std(0),
        }

    def __repr__(self):
        return f"EnsemblePredictor(n={len(self.predictors)}, dim={self.state_dim})"

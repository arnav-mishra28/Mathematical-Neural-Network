"""
mnn.advanced.chaos_simulation.simulator
=========================================
High-fidelity chaotic system simulators.

Generates training data for neural dynamics learning:
  (state, time) → trajectory + derivatives

Systems
-------
  Lorenz (1963)      — canonical butterfly attractor
  Rössler            — single-scroll attractor
  Chen               — double-scroll variant
  Chua               — electronic circuit chaos
  Duffing            — forced nonlinear oscillator
  Van der Pol        — relaxation oscillator
  Halvorsen          — cyclic attractor
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Dict, List


@dataclass
class ChaosTrajectory:
    """Container for a simulated chaotic trajectory."""
    states:      np.ndarray          # (N, dim) positions
    derivatives: np.ndarray          # (N, dim) dx/dt at each state
    times:       np.ndarray          # (N,) time points
    system_name: str = ""
    params:      Dict = field(default_factory=dict)
    dt:          float = 0.01

    def __len__(self): return len(self.states)

    def train_test_split(self, test_frac: float = 0.2):
        n_test = int(len(self) * test_frac)
        n_train = len(self) - n_test
        tr = ChaosTrajectory(self.states[:n_train], self.derivatives[:n_train],
                              self.times[:n_train], self.system_name, self.params, self.dt)
        te = ChaosTrajectory(self.states[n_train:], self.derivatives[n_train:],
                              self.times[n_train:], self.system_name, self.params, self.dt)
        return tr, te

    def add_noise(self, noise_std: float) -> "ChaosTrajectory":
        noisy_states = self.states + np.random.randn(*self.states.shape) * noise_std
        return ChaosTrajectory(noisy_states.astype(np.float32), self.derivatives,
                                self.times, self.system_name + "_noisy", self.params, self.dt)

    def __repr__(self):
        return (f"ChaosTrajectory({self.system_name}, "
                f"n={len(self)}, dim={self.states.shape[1]}, "
                f"dt={self.dt})")


class ChaosSimulator:
    """
    Base class for chaotic system simulators.
    Integrates ODE and computes exact derivatives at each point.
    """
    def __init__(self, name: str, ode_fn: Callable, dim: int,
                 default_params: Dict, default_ic: np.ndarray):
        self.name           = name
        self.ode_fn         = ode_fn
        self.dim            = dim
        self.default_params = default_params
        self.default_ic     = default_ic

    def simulate(self,
                 initial_state: Optional[np.ndarray] = None,
                 t_span:  Tuple[float, float] = (0, 50),
                 dt:      float = 0.01,
                 method:  str   = "RK45",
                 params:  Optional[Dict] = None) -> ChaosTrajectory:
        """
        Integrate the ODE and return trajectory + derivatives.

        Parameters
        ----------
        initial_state : (dim,) starting point
        t_span        : (t_start, t_end)
        dt            : time step for output
        """
        ic     = initial_state if initial_state is not None else self.default_ic.copy()
        prms   = params or self.default_params
        t_eval = np.arange(t_span[0], t_span[1], dt)

        sol    = solve_ivp(
            lambda t, x: self.ode_fn(t, x, prms),
            t_span, ic, t_eval=t_eval,
            method=method, rtol=1e-9, atol=1e-12, dense_output=False
        )
        states = sol.y.T.astype(np.float32)          # (N, dim)
        derivs = np.array([
            self.ode_fn(0, states[i], prms)
            for i in range(len(states))
        ], dtype=np.float32)                          # (N, dim)

        return ChaosTrajectory(
            states=states, derivatives=derivs,
            times=sol.t.astype(np.float32),
            system_name=self.name, params=prms, dt=dt
        )

    def multi_trajectory(self, n_traj: int = 10,
                          t_span: Tuple = (0, 20),
                          dt: float = 0.01,
                          ic_std: float = 2.0,
                          seed: int = 0) -> ChaosTrajectory:
        """Generate multiple trajectories from perturbed initial conditions."""
        rng = np.random.default_rng(seed)
        all_states, all_derivs, all_times = [], [], []
        for _ in range(n_traj):
            ic   = self.default_ic + rng.standard_normal(self.dim) * ic_std
            traj = self.simulate(ic, t_span, dt)
            all_states.append(traj.states)
            all_derivs.append(traj.derivatives)
            all_times.append(traj.times)
        return ChaosTrajectory(
            states=np.vstack(all_states),
            derivatives=np.vstack(all_derivs),
            times=np.concatenate(all_times),
            system_name=self.name + f"_x{n_traj}", params=self.default_params, dt=dt
        )

    def __repr__(self):
        return f"ChaosSimulator({self.name}, dim={self.dim}, params={self.default_params})"


# ── Lorenz System ─────────────────────────────────────────────

class LorenzSimulator(ChaosSimulator):
    """
    Lorenz system (1963):
      dx/dt = σ(y − x)
      dy/dt = x(ρ − z) − y
      dz/dt = xy − βz

    Classic parameters: σ=10, ρ=28, β=8/3
    Lyapunov exponent: λ₁ ≈ 0.906
    Kaplan-Yorke dimension: D_KY ≈ 2.06
    """
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3):
        def ode(t, s, p):
            x, y, z = s
            return [p["sigma"]*(y-x), x*(p["rho"]-z)-y, x*y-p["beta"]*z]

        super().__init__(
            name="Lorenz",
            ode_fn=ode, dim=3,
            default_params={"sigma": sigma, "rho": rho, "beta": beta},
            default_ic=np.array([1.0, 1.0, 1.0])
        )
        self.sigma = sigma; self.rho = rho; self.beta = beta

    def fixed_points(self) -> List[np.ndarray]:
        """The three fixed points of the Lorenz system."""
        if self.rho <= 1: return [np.zeros(3)]
        c = np.sqrt(self.beta * (self.rho - 1))
        return [np.zeros(3), np.array([c, c, self.rho-1]), np.array([-c,-c,self.rho-1])]

    def jacobian_at(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho-z,  -1,        -x],
            [y,            x,        -self.beta]
        ])

    def theoretical_lyapunov(self) -> float:
        """Approximate largest Lyapunov exponent for σ=10,ρ=28,β=8/3."""
        return 0.9056


# ── Rössler System ────────────────────────────────────────────

class RosslerSimulator(ChaosSimulator):
    """
    Rössler system (1976):
      dx/dt = −y − z
      dy/dt = x + ay
      dz/dt = b + z(x − c)

    Classic: a=0.2, b=0.2, c=5.7
    """
    def __init__(self, a=0.2, b=0.2, c=5.7):
        def ode(t, s, p):
            x, y, z = s
            return [-y-z, x+p["a"]*y, p["b"]+z*(x-p["c"])]
        super().__init__(
            name="Rössler", ode_fn=ode, dim=3,
            default_params={"a":a,"b":b,"c":c},
            default_ic=np.array([1.,0.,0.])
        )


# ── Chen System ───────────────────────────────────────────────

class ChenSimulator(ChaosSimulator):
    """
    Chen system (1999):
      dx/dt = a(y − x)
      dy/dt = (c − a)x − xz + cy
      dz/dt = xy − bz
    """
    def __init__(self, a=35., b=3., c=28.):
        def ode(t, s, p):
            x,y,z = s
            return [p["a"]*(y-x),(p["c"]-p["a"])*x-x*z+p["c"]*y,x*y-p["b"]*z]
        super().__init__(
            name="Chen", ode_fn=ode, dim=3,
            default_params={"a":a,"b":b,"c":c},
            default_ic=np.array([-0.1,0.5,-0.6])
        )


# ── Duffing System ────────────────────────────────────────────

class DuffingSimulator(ChaosSimulator):
    """
    Duffing oscillator (forced, damped):
      dx/dt = y
      dy/dt = γcos(ωt) − δy − αx − βx³

    State = (x, y, t_phase) — augmented with phase as 3rd coordinate.
    """
    def __init__(self, alpha=1., beta=-1., gamma=0.3, omega=1.2, delta=0.15):
        def ode(t, s, p):
            x, y = s[0], s[1]
            dy = p["gamma"]*np.cos(p["omega"]*t) - p["delta"]*y - p["alpha"]*x - p["beta"]*x**3
            return [y, dy]
        super().__init__(
            name="Duffing", ode_fn=ode, dim=2,
            default_params={"alpha":alpha,"beta":beta,"gamma":gamma,"omega":omega,"delta":delta},
            default_ic=np.array([0.,0.])
        )


# ── Van der Pol ───────────────────────────────────────────────

class VanDerPolSimulator(ChaosSimulator):
    """
    Van der Pol oscillator:
      dx/dt = y
      dy/dt = μ(1 − x²)y − x
    """
    def __init__(self, mu=2.0):
        def ode(t, s, p):
            x,y = s
            return [y, p["mu"]*(1-x**2)*y-x]
        super().__init__(
            name="VanDerPol", ode_fn=ode, dim=2,
            default_params={"mu":mu},
            default_ic=np.array([2.,0.])
        )


# ── Halvorsen ─────────────────────────────────────────────────

class HalvorsenSimulator(ChaosSimulator):
    """
    Halvorsen cyclic attractor:
      dx/dt = −ax − 4y − 4z − y²
      (cyclic in x,y,z)
    """
    def __init__(self, a=1.4):
        def ode(t, s, p):
            x,y,z = s; a=p["a"]
            return [-a*x-4*y-4*z-y**2, -a*y-4*z-4*x-z**2, -a*z-4*x-4*y-x**2]
        super().__init__(
            name="Halvorsen", ode_fn=ode, dim=3,
            default_params={"a":a},
            default_ic=np.array([-5.,0.,0.])
        )


# ── Factory ───────────────────────────────────────────────────

class SystemFactory:
    """Get any simulator by name."""
    _registry = {
        "lorenz":    LorenzSimulator,
        "rossler":   RosslerSimulator,
        "chen":      ChenSimulator,
        "duffing":   DuffingSimulator,
        "vanderpol": VanDerPolSimulator,
        "halvorsen": HalvorsenSimulator,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> ChaosSimulator:
        name_l = name.lower()
        if name_l not in cls._registry:
            raise ValueError(f"Unknown system '{name}'. Choose from {list(cls._registry)}")
        return cls._registry[name_l](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())

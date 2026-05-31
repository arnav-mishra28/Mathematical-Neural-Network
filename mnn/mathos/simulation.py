"""mnn.mathos.simulation — Simulation Subsystem (Layer 5) + Discovery Subsystem (Layer 6).

Layer 5: Simulation Subsystem — dynamical systems, PDEs, chaos, geometry,
quantum mathematics integrated from MNN infrastructure.

Layer 6: Discovery Subsystem — continuous autonomous research loop:
Observe → Hypothesize → Experiment → Validate → Publish → New Questions.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============= Layer 5: Simulation Subsystem =============

class SimDomain(Enum):
    DYNAMICS = "dynamics"
    PDE = "pde"
    CHAOS = "chaos"
    GEOMETRY = "geometry"
    QUANTUM = "quantum"
    SPECTRAL = "spectral"
    CUSTOM = "custom"


@dataclass
class SimResult:
    """Result of a simulation."""
    name: str
    domain: SimDomain
    parameters: Dict
    data: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    success: bool = True
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SimulationSubsystem:
    """Layer 5: Integrated simulation engine."""

    def __init__(self):
        self.results: List[SimResult] = []
        self.simulators: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.simulators["heat_1d"] = self._heat_1d
        self.simulators["wave_1d"] = self._wave_1d
        self.simulators["logistic_map"] = self._logistic
        self.simulators["lorenz"] = self._lorenz
        self.simulators["harmonic"] = self._harmonic
        self.simulators["diffusion_2d"] = self._diffusion_2d

    def register(self, name: str, fn: Callable):
        self.simulators[name] = fn

    def run(self, name: str, domain: str = "custom",
            params: Dict = None) -> SimResult:
        sd = SimDomain(domain) if domain in [e.value for e in SimDomain] else SimDomain.CUSTOM
        p = params or {}
        t0 = time.time()
        result = SimResult(name, sd, p)
        sim = self.simulators.get(name)
        if sim:
            try:
                data = sim(**p)
                result.data = data if isinstance(data, dict) else {"output": data}
                result.duration = time.time() - t0
            except Exception as e:
                result.success = False
                result.data = {"error": str(e)}
        else:
            result.success = False
            result.data = {"error": f"Unknown simulator: {name}"}
        self.results.append(result)
        return result

    def _heat_1d(self, n: int = 50, steps: int = 100, alpha: float = 0.1,
                  dt: float = 0.01, **kw) -> Dict:
        dx = 1.0 / n
        u = np.zeros(n); u[n // 2] = 1.0
        for _ in range(steps):
            u_new = u.copy()
            for i in range(1, n - 1):
                u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
            u = u_new
        return {"final": u, "max": float(np.max(u)), "energy": float(np.sum(u**2))}

    def _wave_1d(self, n: int = 50, steps: int = 100, c: float = 1.0,
                  dt: float = 0.005, **kw) -> Dict:
        dx = 1.0 / n
        u, u_prev = np.zeros(n), np.zeros(n)
        u[n // 2] = 1.0
        for _ in range(steps):
            u_new = np.zeros(n)
            for i in range(1, n - 1):
                u_new[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
            u_prev, u = u, u_new
        return {"final": u, "energy": float(np.sum(u**2))}

    def _logistic(self, r: float = 3.9, x0: float = 0.1, steps: int = 200, **kw) -> Dict:
        x = x0
        traj = [x]
        for _ in range(steps):
            x = r * x * (1 - x)
            traj.append(x)
        t = np.array(traj)
        return {"trajectory": t, "mean": float(np.mean(t[50:])),
                "lyapunov": float(np.mean(np.log(np.abs(r - 2*r*t[50:-1]) + 1e-15)))}

    def _lorenz(self, sigma: float = 10, rho: float = 28, beta: float = 8/3,
                 dt: float = 0.01, steps: int = 2000, **kw) -> Dict:
        x, y, z = 1.0, 1.0, 1.0
        traj = [(x, y, z)]
        for _ in range(steps):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz
            traj.append((x, y, z))
        arr = np.array(traj)
        return {"trajectory": arr, "attractor_dim": "~2.06",
                "x_range": [float(arr[:, 0].min()), float(arr[:, 0].max())]}

    def _harmonic(self, omega: float = 1.0, x0: float = 1.0, v0: float = 0.0,
                   dt: float = 0.01, steps: int = 1000, **kw) -> Dict:
        x, v = x0, v0
        traj = [(x, v)]
        for _ in range(steps):
            a = -omega**2 * x
            v += a * dt
            x += v * dt
            traj.append((x, v))
        arr = np.array(traj)
        energy = 0.5 * arr[:, 1]**2 + 0.5 * omega**2 * arr[:, 0]**2
        return {"trajectory": arr, "energy_drift": float(np.std(energy) / (np.mean(energy) + 1e-15))}

    def _diffusion_2d(self, n: int = 20, steps: int = 50, alpha: float = 0.1,
                       dt: float = 0.005, **kw) -> Dict:
        u = np.zeros((n, n)); u[n//2, n//2] = 1.0
        dx = 1.0 / n
        for _ in range(steps):
            u_new = u.copy()
            for i in range(1, n-1):
                for j in range(1, n-1):
                    lap = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx**2
                    u_new[i,j] = u[i,j] + alpha * dt * lap
            u = u_new
        return {"final": u, "max": float(np.max(u)), "total_mass": float(np.sum(u))}

    def summary(self) -> str:
        ok = sum(1 for r in self.results if r.success)
        return f"SimSubsystem: {len(self.results)} runs ({ok} success), {len(self.simulators)} simulators"


# ============= Layer 6: Discovery Subsystem =============

@dataclass
class DiscoveryItem:
    """A discovery in the system."""
    title: str
    category: str  # equation, conjecture, algorithm, structure, connection
    description: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    status: str = "proposed"
    timestamp: float = field(default_factory=time.time)


class DiscoverySubsystem:
    """Layer 6: Autonomous research loop."""

    def __init__(self):
        self.discoveries: List[DiscoveryItem] = []
        self.research_questions: List[str] = []
        self.cycle_count: int = 0

    def observe_patterns(self, data: Dict) -> List[str]:
        """Detect patterns in data."""
        patterns = []
        if "values" in data:
            vals = np.array(data["values"])
            if len(vals) > 2:
                diffs = np.diff(vals)
                if np.std(diffs) < 0.01 * (np.std(vals) + 1e-15):
                    patterns.append("arithmetic_progression")
                ratios = vals[1:] / (vals[:-1] + 1e-15)
                if np.std(ratios) < 0.01 * (np.mean(np.abs(ratios)) + 1e-15):
                    patterns.append("geometric_progression")
        if "symmetry" in str(data):
            patterns.append("symmetry")
        return patterns

    def hypothesize(self, patterns: List[str]) -> List[DiscoveryItem]:
        """Generate hypotheses from patterns."""
        hypotheses = []
        for p in patterns:
            if p == "symmetry":
                hypotheses.append(DiscoveryItem(
                    "Conservation Law", "conjecture",
                    "Symmetry suggests a conservation law (Noether)", 0.65))
            elif p == "arithmetic_progression":
                hypotheses.append(DiscoveryItem(
                    "Linear Structure", "equation",
                    "Data follows linear progression", 0.7))
            elif p == "geometric_progression":
                hypotheses.append(DiscoveryItem(
                    "Exponential Growth/Decay", "equation",
                    "Data follows exponential pattern", 0.7))
            else:
                hypotheses.append(DiscoveryItem(
                    f"Pattern: {p}", "conjecture", f"Detected: {p}", 0.4))
        self.discoveries.extend(hypotheses)
        return hypotheses

    def validate(self, index: int, test_fn: Callable = None,
                  test_inputs: List = None) -> bool:
        if index >= len(self.discoveries):
            return False
        d = self.discoveries[index]
        if test_fn and test_inputs:
            passed = sum(1 for inp in test_inputs if test_fn(inp))
            d.confidence = passed / len(test_inputs)
            d.status = "validated" if d.confidence > 0.7 else "inconclusive"
            return d.confidence > 0.7
        return False

    def generate_questions(self) -> List[str]:
        """Generate next research questions."""
        questions = []
        for d in self.discoveries:
            if d.status == "proposed":
                questions.append(f"Can we prove: {d.title}?")
            elif d.status == "validated":
                questions.append(f"What are the implications of: {d.title}?")
        self.research_questions.extend(questions)
        return questions

    def research_cycle(self, data: Dict = None) -> Dict:
        """One full research cycle."""
        self.cycle_count += 1
        data = data or {}
        patterns = self.observe_patterns(data)
        hypotheses = self.hypothesize(patterns)
        questions = self.generate_questions()
        return {
            "cycle": self.cycle_count,
            "patterns": patterns,
            "hypotheses": len(hypotheses),
            "questions": questions[:5],
        }

    def summary(self) -> str:
        status_counts = {}
        for d in self.discoveries:
            status_counts[d.status] = status_counts.get(d.status, 0) + 1
        return (f"DiscoverySubsystem: {len(self.discoveries)} discoveries, "
                f"cycles={self.cycle_count}, status={status_counts}")

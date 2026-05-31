"""mnn.researcher.simulation — Simulation Engine + Evidence Scoring.

Module 5: Simulation Engine — connect to MNN's PDE solvers, chaos simulators,
topology engines, quantum-inspired models.

Module 6: Evidence Scoring — evaluate confidence from theoretical,
numerical, experimental, and counterexample evidence.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SimulationType(Enum):
    PDE = "pde"
    DYNAMICAL = "dynamical"
    SPECTRAL = "spectral"
    TOPOLOGICAL = "topological"
    ALGEBRAIC = "algebraic"
    QUANTUM = "quantum"
    CUSTOM = "custom"


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    name: str
    sim_type: SimulationType
    parameters: Dict
    data: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    success: bool = True
    error: str = ""
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SimulationEngine:
    """Module 5: Run mathematical simulations using MNN infrastructure."""

    def __init__(self):
        self.results: List[SimulationResult] = []
        self.simulators: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.simulators["heat_1d"] = self._sim_heat_1d
        self.simulators["wave_1d"] = self._sim_wave_1d
        self.simulators["logistic_map"] = self._sim_logistic
        self.simulators["harmonic_oscillator"] = self._sim_harmonic

    def register_simulator(self, name: str, fn: Callable):
        self.simulators[name] = fn

    def run(self, name: str, sim_type: SimulationType = SimulationType.CUSTOM,
            parameters: Dict = None) -> SimulationResult:
        """Run a named simulation."""
        params = parameters or {}
        t0 = time.time()
        result = SimulationResult(name, sim_type, params)

        simulator = self.simulators.get(name)
        if simulator:
            try:
                data = simulator(**params)
                result.data = data if isinstance(data, dict) else {"output": data}
                result.duration = time.time() - t0
            except Exception as e:
                result.success = False
                result.error = str(e)
                result.duration = time.time() - t0
        else:
            result.success = False
            result.error = f"Unknown simulator: {name}"

        self.results.append(result)
        return result

    def _sim_heat_1d(self, n_points: int = 50, n_steps: int = 100,
                      alpha: float = 0.1, dt: float = 0.01, **kw) -> Dict:
        dx = 1.0 / n_points
        u = np.zeros(n_points)
        u[n_points // 2] = 1.0
        trajectory = [u.copy()]
        for _ in range(n_steps):
            u_new = u.copy()
            for i in range(1, n_points - 1):
                u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
            u = u_new
            trajectory.append(u.copy())
        return {"trajectory": np.array(trajectory), "final": u,
                "max_val": float(np.max(np.abs(u))), "n_steps": n_steps}

    def _sim_wave_1d(self, n_points: int = 50, n_steps: int = 100,
                      c: float = 1.0, dt: float = 0.01, **kw) -> Dict:
        dx = 1.0 / n_points
        u = np.zeros(n_points)
        u_prev = np.zeros(n_points)
        u[n_points // 2] = 1.0
        trajectory = [u.copy()]
        for _ in range(n_steps):
            u_new = np.zeros(n_points)
            for i in range(1, n_points - 1):
                u_new[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
            u_prev = u.copy()
            u = u_new
            trajectory.append(u.copy())
        return {"trajectory": np.array(trajectory), "final": u, "n_steps": n_steps}

    def _sim_logistic(self, r: float = 3.9, x0: float = 0.1,
                       n_steps: int = 200, **kw) -> Dict:
        x = x0
        trajectory = [x]
        for _ in range(n_steps):
            x = r * x * (1 - x)
            trajectory.append(x)
        traj = np.array(trajectory)
        return {"trajectory": traj, "final": float(traj[-1]),
                "mean": float(np.mean(traj[50:])), "std": float(np.std(traj[50:]))}

    def _sim_harmonic(self, omega: float = 1.0, x0: float = 1.0,
                       v0: float = 0.0, dt: float = 0.01,
                       n_steps: int = 1000, **kw) -> Dict:
        x, v = x0, v0
        trajectory = [(x, v)]
        for _ in range(n_steps):
            a = -omega**2 * x
            v += a * dt
            x += v * dt
            trajectory.append((x, v))
        traj = np.array(trajectory)
        energy = 0.5 * traj[:, 1]**2 + 0.5 * omega**2 * traj[:, 0]**2
        return {"trajectory": traj, "energy": energy,
                "energy_drift": float(np.std(energy) / (np.mean(energy) + 1e-15))}

    def summary(self) -> str:
        n_ok = sum(1 for r in self.results if r.success)
        return f"SimulationEngine: {len(self.results)} runs ({n_ok} successful)"


# ---- Module 6: Evidence Scoring ----

class EvidenceType(Enum):
    THEORETICAL = "theoretical"
    NUMERICAL = "numerical"
    EXPERIMENTAL = "experimental"
    COUNTEREXAMPLE = "counterexample"
    ANALOGY = "analogy"


@dataclass
class EvidenceItem:
    """A piece of evidence for or against a hypothesis."""
    evidence_type: EvidenceType
    description: str
    supports: bool  # True = supports, False = contradicts
    strength: float = 0.5  # 0 to 1
    source: str = ""


class EvidenceScorer:
    """Module 6: Evaluate hypothesis confidence from multiple evidence types."""

    def __init__(self):
        self.evidence_store: Dict[int, List[EvidenceItem]] = {}
        self.weights = {
            EvidenceType.THEORETICAL: 0.3,
            EvidenceType.NUMERICAL: 0.25,
            EvidenceType.EXPERIMENTAL: 0.25,
            EvidenceType.COUNTEREXAMPLE: 0.15,
            EvidenceType.ANALOGY: 0.05,
        }

    def add_evidence(self, hypothesis_index: int, evidence: EvidenceItem):
        if hypothesis_index not in self.evidence_store:
            self.evidence_store[hypothesis_index] = []
        self.evidence_store[hypothesis_index].append(evidence)

    def score(self, hypothesis_index: int) -> Dict:
        """Compute confidence score for a hypothesis."""
        items = self.evidence_store.get(hypothesis_index, [])
        if not items:
            return {"confidence": 0.5, "support": 0, "against": 0,
                    "breakdown": {}, "verdict": "insufficient_evidence"}

        by_type = {}
        for item in items:
            t = item.evidence_type
            if t not in by_type:
                by_type[t] = {"support": 0.0, "against": 0.0, "count": 0}
            if item.supports:
                by_type[t]["support"] += item.strength
            else:
                by_type[t]["against"] += item.strength
            by_type[t]["count"] += 1

        # Weighted confidence
        total_weight = 0.0
        weighted_score = 0.0
        breakdown = {}
        for t, data in by_type.items():
            total = data["support"] + data["against"]
            if total > 0:
                type_score = data["support"] / total
            else:
                type_score = 0.5
            w = self.weights.get(t, 0.1)
            weighted_score += w * type_score
            total_weight += w
            breakdown[t.value] = {
                "score": round(type_score, 3),
                "support": round(data["support"], 3),
                "against": round(data["against"], 3),
            }

        confidence = weighted_score / (total_weight + 1e-15)

        # Counterexample override
        counterexamples = [i for i in items
                           if i.evidence_type == EvidenceType.COUNTEREXAMPLE and not i.supports]
        if counterexamples:
            confidence = min(confidence, 0.2)

        if confidence >= 0.8:
            verdict = "strongly_supported"
        elif confidence >= 0.6:
            verdict = "moderately_supported"
        elif confidence >= 0.4:
            verdict = "inconclusive"
        elif confidence >= 0.2:
            verdict = "weakly_refuted"
        else:
            verdict = "refuted"

        return {
            "confidence": round(confidence, 3),
            "support": sum(1 for i in items if i.supports),
            "against": sum(1 for i in items if not i.supports),
            "breakdown": breakdown,
            "verdict": verdict,
        }

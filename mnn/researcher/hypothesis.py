"""mnn.researcher.hypothesis — Hypothesis Generator + Experiment Planner.

Module 3: Scientific creativity engine — observe anomalies, generate
multi-level hypotheses (parameter → equation → theorem → framework).

Module 4: Experiment Planner — design numerical, symbolic, and
counterexample experiments to test hypotheses.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class HypothesisLevel(Enum):
    PARAMETER = "parameter"       # Level 1: parameter relationship
    EQUATION = "equation"         # Level 2: equation relationship
    THEOREM = "theorem"           # Level 3: new theorem
    FRAMEWORK = "framework"       # Level 4: new mathematical framework


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    PROVED = "proved"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Hypothesis:
    """A scientific hypothesis at a specific level."""
    statement: str
    level: HypothesisLevel
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    source: str = ""
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def update_confidence(self):
        n_for = len(self.evidence_for)
        n_against = len(self.evidence_against)
        total = n_for + n_against
        if total > 0:
            self.confidence = n_for / total


class HypothesisGenerator:
    """Module 3: Generate multi-level hypotheses from observations."""

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []
        self.observation_history: List[Dict] = []

    def observe(self, observations: List[Dict]):
        """Record observations for pattern analysis."""
        self.observation_history.extend(observations)

    def generate(self, observations: List[Dict] = None) -> List[Hypothesis]:
        """Generate hypotheses from observations."""
        obs = observations or self.observation_history
        if not obs:
            return []

        new_hypotheses = []

        # Level 1: Parameter relationships
        new_hypotheses.extend(self._detect_parameter_patterns(obs))

        # Level 2: Equation relationships
        new_hypotheses.extend(self._detect_equation_patterns(obs))

        # Level 3: Structural patterns → theorems
        new_hypotheses.extend(self._detect_structural_patterns(obs))

        self.hypotheses.extend(new_hypotheses)
        return new_hypotheses

    def _detect_parameter_patterns(self, obs: List[Dict]) -> List[Hypothesis]:
        results = []
        try:
            xs = [float(o.get("input", 0)) for o in obs if o.get("input") is not None]
            ys = [float(o.get("output", 0)) for o in obs if o.get("output") is not None]
            if len(xs) >= 3 and len(ys) >= 3:
                xs, ys = np.array(xs[:len(ys)]), np.array(ys[:len(xs)])
                # Linear fit
                if len(xs) >= 2:
                    coeffs = np.polyfit(xs, ys, 1)
                    residual = np.mean((ys - np.polyval(coeffs, xs))**2)
                    if residual < 0.01 * (np.var(ys) + 1e-15):
                        results.append(Hypothesis(
                            f"Linear relationship: y ≈ {coeffs[0]:.4f}x + {coeffs[1]:.4f}",
                            HypothesisLevel.PARAMETER, confidence=0.8,
                            source="linear_regression"))
                # Quadratic fit
                if len(xs) >= 3:
                    coeffs2 = np.polyfit(xs, ys, 2)
                    residual2 = np.mean((ys - np.polyval(coeffs2, xs))**2)
                    if residual2 < 0.01 * (np.var(ys) + 1e-15) and abs(coeffs2[0]) > 1e-6:
                        results.append(Hypothesis(
                            f"Quadratic relationship: y ≈ {coeffs2[0]:.4f}x² + {coeffs2[1]:.4f}x + {coeffs2[2]:.4f}",
                            HypothesisLevel.PARAMETER, confidence=0.7,
                            source="quadratic_regression"))
        except Exception:
            pass
        return results

    def _detect_equation_patterns(self, obs: List[Dict]) -> List[Hypothesis]:
        results = []
        # Symmetry detection
        symmetric_obs = [o for o in obs if o.get("input") is not None and o.get("output") is not None]
        if len(symmetric_obs) >= 4:
            try:
                pairs = {}
                for o in symmetric_obs:
                    x = float(o["input"])
                    pairs[x] = float(o["output"])
                sym_count = sum(1 for x, y in pairs.items()
                               if -x in pairs and abs(y - pairs[-x]) < 1e-6)
                if sym_count >= 2:
                    results.append(Hypothesis(
                        "Function appears even: f(x) = f(-x)",
                        HypothesisLevel.EQUATION, confidence=0.65,
                        source="symmetry_detection"))
            except Exception:
                pass

        # Conservation detection
        if all("conserved" in str(o.get("tags", [])) for o in obs if o.get("tags")):
            results.append(Hypothesis(
                "A conservation law may govern this system",
                HypothesisLevel.EQUATION, confidence=0.6,
                source="conservation_detection"))
        return results

    def _detect_structural_patterns(self, obs: List[Dict]) -> List[Hypothesis]:
        results = []
        # Domain crossing
        domains = set(o.get("domain", "") for o in obs if o.get("domain"))
        if len(domains) >= 2:
            results.append(Hypothesis(
                f"Cross-domain connection detected between: {', '.join(domains)}",
                HypothesisLevel.THEOREM, confidence=0.4,
                source="cross_domain_detection"))
        return results

    def open_hypotheses(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses
                if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)]

    def summary(self) -> str:
        counts = {}
        for h in self.hypotheses:
            counts[h.status.value] = counts.get(h.status.value, 0) + 1
        return f"HypothesisGenerator: {len(self.hypotheses)} total, status={counts}"


# ---- Module 4: Experiment Planner ----

class ExperimentType(Enum):
    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    COUNTEREXAMPLE = "counterexample"
    SIMULATION = "simulation"
    COMPARISON = "comparison"


@dataclass
class Experiment:
    """A planned or executed experiment."""
    name: str
    experiment_type: ExperimentType
    hypothesis_index: int
    description: str
    parameters: Dict = field(default_factory=dict)
    status: str = "planned"  # planned, running, completed, failed
    results: Optional[Dict] = None
    conclusion: str = ""
    timestamp: float = field(default_factory=time.time)


class ExperimentPlanner:
    """Module 4: Design and manage experiments to test hypotheses."""

    def __init__(self):
        self.experiments: List[Experiment] = []

    def plan_experiments(self, hypothesis: Hypothesis,
                          hypothesis_index: int) -> List[Experiment]:
        """Design experiments to test a hypothesis."""
        experiments = []

        # Numerical test
        experiments.append(Experiment(
            f"numerical_test_{hypothesis_index}",
            ExperimentType.NUMERICAL, hypothesis_index,
            f"Numerical validation of: {hypothesis.statement[:80]}",
            {"n_samples": 100, "tolerance": 1e-6}))

        # Counterexample search
        experiments.append(Experiment(
            f"counterexample_{hypothesis_index}",
            ExperimentType.COUNTEREXAMPLE, hypothesis_index,
            f"Search for counterexamples to: {hypothesis.statement[:80]}",
            {"n_random": 1000, "domain": "[-10, 10]"}))

        # Symbolic verification
        if hypothesis.level in (HypothesisLevel.EQUATION, HypothesisLevel.THEOREM):
            experiments.append(Experiment(
                f"symbolic_test_{hypothesis_index}",
                ExperimentType.SYMBOLIC, hypothesis_index,
                f"Symbolic analysis of: {hypothesis.statement[:80]}"))

        self.experiments.extend(experiments)
        return experiments

    def run_experiment(self, experiment: Experiment,
                        test_fn: Callable = None) -> Dict:
        """Execute an experiment."""
        experiment.status = "running"
        result = {"success": True, "details": {}}

        if test_fn:
            try:
                output = test_fn(**experiment.parameters)
                result["details"] = output if isinstance(output, dict) else {"output": output}
                experiment.status = "completed"
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
                experiment.status = "failed"
        else:
            experiment.status = "completed"
            result["details"] = {"note": "No test function provided, marked complete"}

        experiment.results = result
        return result

    def completed_experiments(self) -> List[Experiment]:
        return [e for e in self.experiments if e.status == "completed"]

    def pending_experiments(self) -> List[Experiment]:
        return [e for e in self.experiments if e.status == "planned"]

    def summary(self) -> str:
        counts = {}
        for e in self.experiments:
            counts[e.status] = counts.get(e.status, 0) + 1
        return f"ExperimentPlanner: {len(self.experiments)} total, status={counts}"

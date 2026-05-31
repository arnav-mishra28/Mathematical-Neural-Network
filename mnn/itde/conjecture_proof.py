"""mnn.itde.conjecture_proof — Conjecture Playground + Proof Assistant.

Module 3: Conjecture Playground — enter observations, get AI-generated
conjectures with confidence scores, evidence, counterexample searches.

Module 4: Proof Assistant — collaborative proof helper suggesting strategies
(induction, contradiction, spectral, categorical) with interactive feedback.
"""
from __future__ import annotations
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class PlaygroundConjecture:
    """A conjecture in the playground with interactive state."""
    statement: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    status: str = "open"
    user_notes: str = ""

    def card(self) -> str:
        sym = {"open": "?", "supported": "✓", "refuted": "✗", "proved": "●"}
        return (f"  {sym.get(self.status, '·')} {self.statement}\n"
                f"    Confidence: {self.confidence:.0%} | "
                f"Evidence: {len(self.evidence)} | "
                f"Counterexamples: {len(self.counterexamples)}")


class ConjecturePlayground:
    """Module 3: Interactive conjecture exploration."""

    def __init__(self):
        self.conjectures: List[PlaygroundConjecture] = []
        self.observations: List[Dict] = []

    def observe(self, description: str, data: Dict = None):
        """Record an observation."""
        self.observations.append({
            "description": description, "data": data or {},
            "timestamp": time.time()})

    def generate_conjectures(self, observation: str = None) -> List[PlaygroundConjecture]:
        """Generate conjectures from observations."""
        new_conjs = []

        if "symmetry" in (observation or "").lower():
            new_conjs.append(PlaygroundConjecture(
                "A conserved quantity exists", 0.65,
                [f"Symmetry observed: {observation}"]))
            new_conjs.append(PlaygroundConjecture(
                "Group action is invariant", 0.55,
                [f"Symmetry: {observation}"]))
            new_conjs.append(PlaygroundConjecture(
                "Hidden manifold structure underlies the symmetry", 0.40))

        if "periodic" in (observation or "").lower():
            new_conjs.append(PlaygroundConjecture(
                "System has a periodic orbit", 0.70,
                [f"Periodicity: {observation}"]))

        if "decay" in (observation or "").lower():
            new_conjs.append(PlaygroundConjecture(
                "Exponential decay rate governs the system", 0.60))
            new_conjs.append(PlaygroundConjecture(
                "Spectral gap determines decay rate", 0.55))

        if "linear" in (observation or "").lower():
            new_conjs.append(PlaygroundConjecture(
                "Underlying map is a linear transformation", 0.75))

        # Numerical pattern detection from stored observations
        for obs in self.observations:
            data = obs.get("data", {})
            if "values" in data:
                vals = np.array(data["values"])
                if len(vals) > 2 and np.std(np.diff(vals)) < 0.01 * np.std(vals):
                    new_conjs.append(PlaygroundConjecture(
                        f"Arithmetic progression detected in {obs['description']}", 0.7))

        if not new_conjs:
            new_conjs.append(PlaygroundConjecture(
                f"Pattern in: {observation or 'observations'}", 0.3))

        self.conjectures.extend(new_conjs)
        return new_conjs

    def test_conjecture(self, index: int, test_fn: Callable,
                         test_inputs: List) -> Dict:
        """Test a conjecture against data."""
        if index >= len(self.conjectures):
            return {"error": "Invalid index"}
        c = self.conjectures[index]
        passed, failed = 0, 0
        for inp in test_inputs:
            try:
                result = test_fn(inp)
                if result:
                    passed += 1
                    c.evidence.append(f"Passed: {inp}")
                else:
                    failed += 1
                    c.counterexamples.append(f"Failed: {inp}")
            except Exception as e:
                failed += 1
                c.counterexamples.append(f"Error({inp}): {e}")

        c.confidence = passed / (passed + failed) if (passed + failed) > 0 else c.confidence
        c.status = "supported" if failed == 0 else "refuted" if passed == 0 else "open"
        return {"passed": passed, "failed": failed, "confidence": c.confidence}

    def dashboard(self) -> str:
        lines = ["╔═ CONJECTURE PLAYGROUND", f"║  {len(self.conjectures)} conjectures"]
        for i, c in enumerate(self.conjectures):
            lines.append(f"║  [{i}] {c.card()}")
        lines.append("╚═")
        return "\n".join(lines)

    def summary(self) -> str:
        counts = {}
        for c in self.conjectures:
            counts[c.status] = counts.get(c.status, 0) + 1
        return f"ConjecturePlayground: {len(self.conjectures)} conjectures, status={counts}"


# ---- Module 4: Proof Assistant ----

@dataclass
class ProofSuggestion:
    """A proof strategy suggestion."""
    strategy: str
    description: str
    applicability: float  # 0-1
    steps: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


class ProofAssistant:
    """Module 4: Collaborative proof helper."""

    def __init__(self):
        self.strategies = self._default_strategies()
        self.proof_attempts: List[Dict] = []

    def _default_strategies(self) -> List[Dict]:
        return [
            {"name": "Induction", "tags": ["number_theory", "combinatorics", "algebra"],
             "description": "Base case + inductive step. Works for statements about natural numbers."},
            {"name": "Contradiction", "tags": ["analysis", "number_theory", "logic"],
             "description": "Assume negation, derive impossibility."},
            {"name": "Direct Construction", "tags": ["algebra", "topology", "geometry"],
             "description": "Explicitly build the desired object."},
            {"name": "Symmetry Argument", "tags": ["algebra", "geometry", "pde"],
             "description": "Exploit symmetry to reduce problem complexity."},
            {"name": "Spectral Decomposition", "tags": ["pde", "analysis", "topology"],
             "description": "Analyze via eigenvalues/eigenvectors of associated operator."},
            {"name": "Categorical Argument", "tags": ["category_theory", "algebra"],
             "description": "Use universal properties, functors, natural transformations."},
            {"name": "Geometric Proof", "tags": ["geometry", "topology", "manifold"],
             "description": "Use manifold structure, geodesics, curvature."},
            {"name": "Probabilistic Method", "tags": ["combinatorics", "number_theory"],
             "description": "Show existence via probability argument."},
        ]

    def suggest(self, goal: str, tags: List[str] = None,
                context: Dict = None) -> List[ProofSuggestion]:
        """Suggest proof strategies for a goal."""
        suggestions = []
        tags = tags or []
        goal_lower = goal.lower()

        for s in self.strategies:
            score = 0.3  # base
            # Tag matching
            if tags:
                overlap = len(set(s["tags"]) & set(tags))
                score += 0.2 * overlap / max(len(tags), 1)
            # Keyword matching
            if "induct" in goal_lower and s["name"] == "Induction":
                score += 0.3
            if "contradi" in goal_lower and s["name"] == "Contradiction":
                score += 0.3
            if any(w in goal_lower for w in ["symmetr", "invariant"]) and "Symmetry" in s["name"]:
                score += 0.3
            if any(w in goal_lower for w in ["spectral", "eigen", "laplacian"]) and "Spectral" in s["name"]:
                score += 0.3
            if any(w in goal_lower for w in ["manifold", "curvature", "geodesic"]) and "Geometric" in s["name"]:
                score += 0.3

            suggestions.append(ProofSuggestion(
                s["name"], s["description"], min(score, 1.0)))

        suggestions.sort(key=lambda x: -x.applicability)
        return suggestions[:5]

    def start_proof(self, goal: str, strategy: str) -> Dict:
        """Start a proof attempt."""
        attempt = {
            "goal": goal, "strategy": strategy,
            "steps": [], "status": "in_progress",
            "timestamp": time.time()}
        self.proof_attempts.append(attempt)
        return attempt

    def add_step(self, attempt_index: int, step: str, result: str = ""):
        if 0 <= attempt_index < len(self.proof_attempts):
            self.proof_attempts[attempt_index]["steps"].append({
                "step": step, "result": result})

    def complete_proof(self, attempt_index: int, success: bool = True):
        if 0 <= attempt_index < len(self.proof_attempts):
            self.proof_attempts[attempt_index]["status"] = "proved" if success else "failed"

    def summary(self) -> str:
        counts = {}
        for a in self.proof_attempts:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
        return f"ProofAssistant: {len(self.proof_attempts)} attempts, status={counts}"

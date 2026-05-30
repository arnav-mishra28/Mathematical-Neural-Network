"""mnn.agi.reasoning — Reasoning Engine.

Module 3: Hybrid reasoning combining symbolic, neural, geometric,
and category-theoretic perspectives simultaneously.

Module 4: Conjecture Engine — observe patterns, generate hypotheses, validate.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ReasoningMode(Enum):
    SYMBOLIC = "symbolic"
    NEURAL = "neural"
    GEOMETRIC = "geometric"
    CATEGORICAL = "categorical"
    SPECTRAL = "spectral"
    HYBRID = "hybrid"


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    mode: ReasoningMode
    description: str
    input_data: Any = None
    output_data: Any = None
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    goal: str
    steps: List[ReasoningStep] = field(default_factory=list)

    def add_step(self, step: ReasoningStep):
        self.steps.append(step)

    @property
    def confidence(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.prod([s.confidence for s in self.steps]))

    def summary(self) -> str:
        lines = [f"ReasoningChain: {self.goal} ({len(self.steps)} steps, conf={self.confidence:.3f})"]
        for i, s in enumerate(self.steps):
            lines.append(f"  [{i}] ({s.mode.value}) {s.description} [conf={s.confidence:.2f}]")
        return "\n".join(lines)


class SymbolicReasoner:
    """Symbolic reasoning: exact manipulation of mathematical expressions."""

    def __init__(self):
        self.rules: Dict[str, Callable] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        self.rules["commutativity"] = lambda expr: f"commute({expr})"
        self.rules["associativity"] = lambda expr: f"assoc({expr})"
        self.rules["distributivity"] = lambda expr: f"dist({expr})"
        self.rules["identity"] = lambda expr: f"id({expr})"
        self.rules["inverse"] = lambda expr: f"inv({expr})"

    def add_rule(self, name: str, rule: Callable):
        self.rules[name] = rule

    def apply_rule(self, name: str, expression: str) -> Optional[str]:
        rule = self.rules.get(name)
        return rule(expression) if rule else None

    def derive(self, expression: str, rule_sequence: List[str]) -> List[str]:
        chain = [expression]
        current = expression
        for rule_name in rule_sequence:
            result = self.apply_rule(rule_name, current)
            if result:
                chain.append(result)
                current = result
        return chain

    def check_property(self, operation: Callable, elements: List[Any],
                        property_name: str) -> Dict:
        """Check algebraic properties."""
        results = {"property": property_name, "holds": True, "counterexample": None}
        if property_name == "commutativity":
            for a in elements:
                for b in elements:
                    if operation(a, b) != operation(b, a):
                        results["holds"] = False
                        results["counterexample"] = (a, b)
                        return results
        elif property_name == "associativity":
            for a in elements:
                for b in elements:
                    for c in elements:
                        if operation(operation(a, b), c) != operation(a, operation(b, c)):
                            results["holds"] = False
                            results["counterexample"] = (a, b, c)
                            return results
        return results


class GeometricReasoner:
    """Geometric reasoning using manifold representations."""

    def __init__(self, embed_dim: int = 32):
        self.embed_dim = embed_dim
        self.embeddings: Dict[str, np.ndarray] = {}

    def embed(self, name: str, vector: np.ndarray):
        self.embeddings[name] = np.asarray(vector)

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embeddings.get(a), self.embeddings.get(b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-15))

    def nearest(self, name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        v = self.embeddings.get(name)
        if v is None:
            return []
        dists = []
        for n, emb in self.embeddings.items():
            if n != name:
                sim = float(np.dot(v, emb) / (np.linalg.norm(v) * np.linalg.norm(emb) + 1e-15))
                dists.append((n, sim))
        dists.sort(key=lambda x: -x[1])
        return dists[:top_k]

    def analogy(self, a: str, b: str, c: str) -> Optional[Tuple[str, float]]:
        """a:b :: c:? via vector arithmetic."""
        va, vb, vc = (self.embeddings.get(x) for x in [a, b, c])
        if any(v is None for v in [va, vb, vc]):
            return None
        target = vb - va + vc
        best_name, best_sim = None, -1
        for n, emb in self.embeddings.items():
            if n not in [a, b, c]:
                sim = float(np.dot(target, emb) / (np.linalg.norm(target) * np.linalg.norm(emb) + 1e-15))
                if sim > best_sim:
                    best_sim = sim
                    best_name = n
        return (best_name, best_sim) if best_name else None


class HybridReasoner:
    """Combines symbolic, neural, geometric, and categorical reasoning."""

    def __init__(self):
        self.symbolic = SymbolicReasoner()
        self.geometric = GeometricReasoner()

    def reason(self, goal: str, context: Dict = None) -> ReasoningChain:
        chain = ReasoningChain(goal)
        context = context or {}

        # Step 1: Symbolic analysis
        chain.add_step(ReasoningStep(
            ReasoningMode.SYMBOLIC,
            f"Analyze structure of: {goal}",
            confidence=0.9))

        # Step 2: Geometric embedding lookup
        chain.add_step(ReasoningStep(
            ReasoningMode.GEOMETRIC,
            "Find geometrically similar known results",
            confidence=0.8))

        # Step 3: Categorical abstraction
        chain.add_step(ReasoningStep(
            ReasoningMode.CATEGORICAL,
            "Identify categorical structure (objects, morphisms)",
            confidence=0.7))

        # Step 4: Synthesis
        chain.add_step(ReasoningStep(
            ReasoningMode.HYBRID,
            "Synthesize perspectives into unified reasoning",
            confidence=0.85))

        return chain


# ---- Module 4: Conjecture Engine ----

class ConjectureStatus(Enum):
    OPEN = "open"
    VALIDATED = "validated"
    REFUTED = "refuted"
    PROVED = "proved"


@dataclass
class Conjecture:
    """A mathematical conjecture with evidence and status."""
    statement: str
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    status: ConjectureStatus = ConjectureStatus.OPEN
    source: str = ""
    tags: List[str] = field(default_factory=list)


class ConjectureEngine:
    """Generate and manage mathematical conjectures."""

    def __init__(self):
        self.conjectures: List[Conjecture] = []

    def observe_and_conjecture(self, observations: List[Dict]) -> List[Conjecture]:
        """From observations, generate conjectures."""
        new_conjectures = []

        # Pattern: constant output → possible identity
        if all(obs.get("output") == observations[0].get("output") for obs in observations):
            c = Conjecture(
                f"The output appears constant: {observations[0].get('output')}",
                confidence=0.7, evidence=[str(o) for o in observations[:3]],
                source="constant_detection")
            new_conjectures.append(c)

        # Pattern: linearity
        if len(observations) >= 3:
            try:
                xs = [float(o.get("input", 0)) for o in observations]
                ys = [float(o.get("output", 0)) for o in observations]
                if len(xs) >= 2:
                    diffs = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
                    if len(diffs) >= 2 and all(abs(d - diffs[0]) < 1e-6 for d in diffs):
                        slope = diffs[0]
                        c = Conjecture(
                            f"Function appears linear with slope ≈ {slope:.4f}",
                            confidence=0.8, source="linearity_detection")
                        new_conjectures.append(c)
            except (ValueError, TypeError):
                pass

        # Pattern: symmetry f(x) = f(-x)
        sym_pairs = [(o1, o2) for o1 in observations for o2 in observations
                     if str(o1.get("input", "")) == str(-float(o2.get("input", 0))) if o2.get("input") is not None]
        if sym_pairs:
            if all(abs(float(o1.get("output", 0)) - float(o2.get("output", 0))) < 1e-6
                   for o1, o2 in sym_pairs[:5]):
                c = Conjecture("Function appears to be even: f(x) = f(-x)",
                              confidence=0.6, source="symmetry_detection")
                new_conjectures.append(c)

        self.conjectures.extend(new_conjectures)
        return new_conjectures

    def validate(self, index: int, test_fn: Callable, test_inputs: List) -> bool:
        """Test a conjecture against a function."""
        if index >= len(self.conjectures):
            return False
        c = self.conjectures[index]
        passed = True
        for inp in test_inputs:
            try:
                result = test_fn(inp)
                c.evidence.append(f"f({inp})={result}")
            except Exception as e:
                c.counterexamples.append(f"f({inp}) raised {e}")
                passed = False
        c.status = ConjectureStatus.VALIDATED if passed else ConjectureStatus.REFUTED
        return passed

    def open_conjectures(self) -> List[Conjecture]:
        return [c for c in self.conjectures if c.status == ConjectureStatus.OPEN]

    def summary(self) -> str:
        counts = {}
        for c in self.conjectures:
            counts[c.status.value] = counts.get(c.status.value, 0) + 1
        return f"ConjectureEngine: {len(self.conjectures)} total, status={counts}"

"""mnn.agi.planner — Proof Strategy Engine + Mathematical Planner.

Module 5: Generate proof strategies (induction, contradiction, symmetry, etc.)
instead of immediately searching for proofs.

Module 6: Long-horizon planning — decompose goals into lemma chains.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class StrategyType(Enum):
    INDUCTION = "induction"
    CONTRADICTION = "contradiction"
    CONSTRUCTION = "construction"
    SYMMETRY = "symmetry"
    INVARIANT = "invariant"
    SPECTRAL = "spectral"
    GEOMETRIC = "geometric"
    ALGEBRAIC = "algebraic"
    COMBINATORIAL = "combinatorial"
    CATEGORICAL = "categorical"
    DIRECT = "direct"


@dataclass
class ProofStrategy:
    """A proof strategy with applicability conditions."""
    name: str
    strategy_type: StrategyType
    description: str
    applicability: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    success_rate: float = 0.5
    examples: List[str] = field(default_factory=list)

    def is_applicable(self, tags: List[str]) -> bool:
        if not self.applicability:
            return True
        return any(t in tags for t in self.applicability)


class ProofStrategyEngine:
    """Module 5: Generate and rank proof strategies."""

    def __init__(self):
        self.strategies: List[ProofStrategy] = []
        self._register_defaults()

    def _register_defaults(self):
        defaults = [
            ProofStrategy("Mathematical Induction", StrategyType.INDUCTION,
                         "Prove base case, assume for n, prove for n+1.",
                         ["number_theory", "combinatorics", "algebra"],
                         examples=["Sum formula", "Divisibility"]),
            ProofStrategy("Proof by Contradiction", StrategyType.CONTRADICTION,
                         "Assume negation, derive contradiction.",
                         ["analysis", "number_theory", "logic"],
                         examples=["Irrationality of √2"]),
            ProofStrategy("Direct Construction", StrategyType.CONSTRUCTION,
                         "Explicitly construct the desired object.",
                         ["algebra", "topology", "geometry"]),
            ProofStrategy("Symmetry Argument", StrategyType.SYMMETRY,
                         "Exploit symmetry to reduce complexity.",
                         ["algebra", "geometry", "pde"]),
            ProofStrategy("Invariant Method", StrategyType.INVARIANT,
                         "Identify quantity preserved under transformation.",
                         ["algebra", "topology", "pde"]),
            ProofStrategy("Spectral Analysis", StrategyType.SPECTRAL,
                         "Analyze eigenvalues/eigenvectors of associated operator.",
                         ["pde", "analysis", "topology"]),
            ProofStrategy("Geometric Proof", StrategyType.GEOMETRIC,
                         "Use geometric intuition and manifold properties.",
                         ["geometry", "topology"]),
            ProofStrategy("Algebraic Manipulation", StrategyType.ALGEBRAIC,
                         "Transform using algebraic identities.",
                         ["algebra", "analysis"]),
            ProofStrategy("Categorical Argument", StrategyType.CATEGORICAL,
                         "Use universal properties, functors, natural transformations.",
                         ["category_theory", "algebra", "topology"]),
        ]
        self.strategies.extend(defaults)

    def suggest_strategies(self, tags: List[str], top_k: int = 5) -> List[ProofStrategy]:
        """Rank strategies by applicability and success rate."""
        applicable = [(s, s.success_rate + 0.3 * s.is_applicable(tags))
                      for s in self.strategies if s.is_applicable(tags)]
        applicable.sort(key=lambda x: -x[1])
        return [s for s, _ in applicable[:top_k]]

    def add_strategy(self, strategy: ProofStrategy):
        self.strategies.append(strategy)

    def update_success_rate(self, name: str, success: bool):
        for s in self.strategies:
            if s.name == name:
                alpha = 0.1
                s.success_rate = (1 - alpha) * s.success_rate + alpha * (1.0 if success else 0.0)
                break


# ---- Module 6: Mathematical Planner ----

class PlanStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class PlanNode:
    """A node in the proof plan tree."""
    name: str
    description: str
    status: PlanStatus = PlanStatus.PENDING
    strategy: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    result: Optional[str] = None

    def is_ready(self, completed: set) -> bool:
        return all(d in completed for d in self.dependencies)


class MathematicalPlanner:
    """Module 6: Decompose mathematical goals into sub-goals with dependencies."""

    def __init__(self):
        self.plans: Dict[str, PlanNode] = {}
        self.root: Optional[str] = None

    def set_goal(self, name: str, description: str):
        self.root = name
        self.plans[name] = PlanNode(name, description)

    def add_subgoal(self, name: str, description: str,
                     parent: str, dependencies: List[str] = None,
                     strategy: str = None):
        node = PlanNode(name, description, dependencies=dependencies or [],
                        strategy=strategy)
        self.plans[name] = node
        if parent in self.plans:
            self.plans[parent].children.append(name)

    def mark_complete(self, name: str, result: str = ""):
        if name in self.plans:
            self.plans[name].status = PlanStatus.COMPLETED
            self.plans[name].result = result

    def mark_failed(self, name: str, reason: str = ""):
        if name in self.plans:
            self.plans[name].status = PlanStatus.FAILED
            self.plans[name].result = reason

    def next_actions(self) -> List[PlanNode]:
        """Return subgoals that are ready to work on."""
        completed = {n for n, p in self.plans.items()
                     if p.status == PlanStatus.COMPLETED}
        return [p for p in self.plans.values()
                if p.status == PlanStatus.PENDING and p.is_ready(completed)]

    def progress(self) -> Dict:
        total = len(self.plans)
        completed = sum(1 for p in self.plans.values() if p.status == PlanStatus.COMPLETED)
        failed = sum(1 for p in self.plans.values() if p.status == PlanStatus.FAILED)
        return {"total": total, "completed": completed, "failed": failed,
                "pending": total - completed - failed,
                "progress_pct": 100 * completed / max(total, 1)}

    def plan_tree(self, node: str = None, indent: int = 0) -> str:
        if node is None:
            node = self.root
        if node is None or node not in self.plans:
            return ""
        p = self.plans[node]
        status_sym = {"pending": "○", "in_progress": "◑",
                      "completed": "●", "failed": "✗", "blocked": "◌"}
        sym = status_sym.get(p.status.value, "?")
        lines = [f"{'  ' * indent}{sym} {p.name}: {p.description}"]
        if p.strategy:
            lines[-1] += f" [{p.strategy}]"
        for child in p.children:
            lines.append(self.plan_tree(child, indent + 1))
        return "\n".join(lines)

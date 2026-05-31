"""mnn.researcher.critique — Self-Critique Engine + Discovery Engine.

Module 7: Self-Critique — ask "Why might I be wrong?" Check invalid
assumptions, hidden biases, insufficient evidence, numerical artifacts.

Module 8: Discovery Engine — detect new equations, conjectures, algorithms,
geometric structures, mathematical connections.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class CritiqueType(Enum):
    ASSUMPTION_CHECK = "assumption_check"
    BIAS_CHECK = "bias_check"
    EVIDENCE_CHECK = "evidence_check"
    NUMERICAL_CHECK = "numerical_check"
    LOGICAL_CHECK = "logical_check"
    GENERALIZATION_CHECK = "generalization_check"


@dataclass
class Critique:
    """A self-critique of a hypothesis or result."""
    critique_type: CritiqueType
    target: str  # what is being critiqued
    issue: str
    severity: str = "warning"  # info, warning, critical
    suggestion: str = ""
    resolved: bool = False


class SelfCritiqueEngine:
    """Module 7: Systematic self-critique to prevent hallucinations."""

    def __init__(self):
        self.critiques: List[Critique] = []

    def critique_hypothesis(self, statement: str, evidence: List[Dict],
                             assumptions: List[str] = None) -> List[Critique]:
        """Generate critiques for a hypothesis."""
        critiques = []

        # Assumption check
        if assumptions:
            for a in assumptions:
                critiques.append(Critique(
                    CritiqueType.ASSUMPTION_CHECK, statement,
                    f"Relies on assumption: '{a}' — is this justified?",
                    "warning", f"Verify assumption: {a}"))

        # Evidence sufficiency
        if len(evidence) < 3:
            critiques.append(Critique(
                CritiqueType.EVIDENCE_CHECK, statement,
                f"Only {len(evidence)} pieces of evidence — may be insufficient",
                "warning", "Gather more evidence before concluding"))

        # Counterexample check
        has_counter = any(not e.get("supports", True) for e in evidence)
        if has_counter:
            critiques.append(Critique(
                CritiqueType.LOGICAL_CHECK, statement,
                "Counterexample exists in evidence",
                "critical", "Re-examine hypothesis in light of counterexample"))

        # Generalization check
        if any("specific" in str(e.get("scope", "")).lower() for e in evidence):
            critiques.append(Critique(
                CritiqueType.GENERALIZATION_CHECK, statement,
                "Evidence may be too specific to generalize",
                "warning", "Test on broader range of inputs"))

        self.critiques.extend(critiques)
        return critiques

    def critique_numerical(self, values: np.ndarray,
                            context: str = "") -> List[Critique]:
        """Check numerical results for artifacts."""
        critiques = []

        if np.any(np.isnan(values)):
            critiques.append(Critique(
                CritiqueType.NUMERICAL_CHECK, context,
                "NaN values detected in results", "critical",
                "Check for division by zero or overflow"))

        if np.any(np.isinf(values)):
            critiques.append(Critique(
                CritiqueType.NUMERICAL_CHECK, context,
                "Infinity detected in results", "critical",
                "Check for numerical instability"))

        if np.std(values) < 1e-15 and len(values) > 1:
            critiques.append(Critique(
                CritiqueType.NUMERICAL_CHECK, context,
                "All values are identical — possible numerical artifact",
                "warning", "Verify input data is not degenerate"))

        if np.max(np.abs(values)) > 1e15:
            critiques.append(Critique(
                CritiqueType.NUMERICAL_CHECK, context,
                f"Very large values detected (max={np.max(np.abs(values)):.2e})",
                "warning", "Check for blow-up or scaling issues"))

        self.critiques.extend(critiques)
        return critiques

    def unresolved(self) -> List[Critique]:
        return [c for c in self.critiques if not c.resolved]

    def critical_issues(self) -> List[Critique]:
        return [c for c in self.critiques if c.severity == "critical" and not c.resolved]

    def resolve(self, index: int, resolution: str = ""):
        if 0 <= index < len(self.critiques):
            self.critiques[index].resolved = True
            self.critiques[index].suggestion = resolution

    def summary(self) -> str:
        n_crit = sum(1 for c in self.critiques if c.severity == "critical")
        n_warn = sum(1 for c in self.critiques if c.severity == "warning")
        n_resolved = sum(1 for c in self.critiques if c.resolved)
        return (f"SelfCritiqueEngine: {len(self.critiques)} critiques "
                f"({n_crit} critical, {n_warn} warnings, {n_resolved} resolved)")


# ---- Module 8: Discovery Engine ----

class DiscoveryType(Enum):
    NEW_EQUATION = "new_equation"
    NEW_CONJECTURE = "new_conjecture"
    NEW_ALGORITHM = "new_algorithm"
    NEW_STRUCTURE = "new_structure"
    NEW_CONNECTION = "new_connection"
    UNIFYING_THEOREM = "unifying_theorem"


@dataclass
class Discovery:
    """A mathematical discovery."""
    discovery_type: DiscoveryType
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    domains: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    status: str = "proposed"  # proposed, validated, published

    def report(self) -> str:
        lines = [
            f"DISCOVERY: {self.title}",
            f"Type: {self.discovery_type.value}",
            f"Confidence: {self.confidence:.1%}",
            f"Description: {self.description}",
        ]
        if self.domains:
            lines.append(f"Domains: {', '.join(self.domains)}")
        if self.evidence:
            lines.append("Evidence:")
            for e in self.evidence:
                lines.append(f"  • {e}")
        if self.implications:
            lines.append("Implications:")
            for imp in self.implications:
                lines.append(f"  → {imp}")
        return "\n".join(lines)


class DiscoveryEngine:
    """Module 8: Detect and formalize mathematical discoveries."""

    def __init__(self):
        self.discoveries: List[Discovery] = []

    def analyze_patterns(self, patterns: List[Dict]) -> List[Discovery]:
        """Generate discoveries from detected patterns."""
        new_discoveries = []

        for pattern in patterns:
            ptype = pattern.get("type", "")
            desc = pattern.get("description", "")
            domains = pattern.get("domains", [])
            evidence = pattern.get("evidence", [])

            if ptype == "symmetry":
                d = Discovery(DiscoveryType.NEW_EQUATION,
                             f"Symmetry in {', '.join(domains)}",
                             desc, evidence, 0.6, domains,
                             ["Possible conservation law"])
                new_discoveries.append(d)

            elif ptype == "cross_domain":
                d = Discovery(DiscoveryType.NEW_CONNECTION,
                             f"Connection: {' ↔ '.join(domains)}",
                             desc, evidence, 0.5, domains,
                             ["May lead to unifying framework"])
                new_discoveries.append(d)

            elif ptype == "invariant":
                d = Discovery(DiscoveryType.NEW_CONJECTURE,
                             f"Invariant detected",
                             desc, evidence, 0.7, domains,
                             ["Investigate topological or algebraic origin"])
                new_discoveries.append(d)

            elif ptype == "unifying":
                d = Discovery(DiscoveryType.UNIFYING_THEOREM,
                             f"Unifying structure across {', '.join(domains)}",
                             desc, evidence, 0.4, domains,
                             ["Requires formal proof"])
                new_discoveries.append(d)

            else:
                d = Discovery(DiscoveryType.NEW_CONJECTURE,
                             f"Pattern: {desc[:60]}",
                             desc, evidence, 0.5, domains)
                new_discoveries.append(d)

        self.discoveries.extend(new_discoveries)
        return new_discoveries

    def validated_discoveries(self) -> List[Discovery]:
        return [d for d in self.discoveries if d.status == "validated"]

    def summary(self) -> str:
        type_counts = {}
        for d in self.discoveries:
            type_counts[d.discovery_type.value] = type_counts.get(d.discovery_type.value, 0) + 1
        return f"DiscoveryEngine: {len(self.discoveries)} discoveries, types={type_counts}"

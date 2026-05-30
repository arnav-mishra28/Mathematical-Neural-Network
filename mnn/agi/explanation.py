"""mnn.agi.explanation — Explanation Engine.

Module 7: Teach mathematics at multiple levels.
Beginner (intuitive), Undergraduate (formal), Research (rigorous).
"""
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class AudienceLevel(Enum):
    BEGINNER = "beginner"
    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    RESEARCH = "research"


@dataclass
class Explanation:
    """A mathematical explanation at a specific audience level."""
    topic: str
    level: AudienceLevel
    summary: str
    details: str
    examples: List[str]
    prerequisites: List[str]
    key_insights: List[str]
    formal_statement: Optional[str] = None
    proof_sketch: Optional[str] = None


class ExplanationEngine:
    """Generate explanations at different audience levels."""

    def __init__(self):
        self.templates: Dict[str, Dict[AudienceLevel, Dict]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.templates["group"] = {
            AudienceLevel.BEGINNER: {
                "summary": "A group is a set with an operation that follows specific rules.",
                "details": ("Think of symmetries of a shape. Rotating a square by 90° is an "
                            "operation. You can combine rotations. There's a 'do nothing' rotation. "
                            "And every rotation can be undone. That's a group!"),
                "examples": ["Rotations of a square", "Clock arithmetic (mod 12)"],
                "key_insights": ["Groups capture the idea of symmetry",
                                  "The operation must be reversible"],
            },
            AudienceLevel.UNDERGRADUATE: {
                "summary": "A group (G,*) is a set with an associative binary operation, identity, and inverses.",
                "details": ("Definition: (G,*) is a group if * is closed and associative on G, "
                            "∃e∈G: e*a = a*e = a ∀a, and ∀a∈G ∃a⁻¹: a*a⁻¹ = a⁻¹*a = e."),
                "examples": ["(ℤ,+)", "(ℤ/nℤ, +)", "S_n (symmetric group)", "GL_n(ℝ)"],
                "key_insights": ["Groups abstract symmetry operations",
                                  "Lagrange: |H| divides |G| for H ≤ G"],
                "formal_statement": "(G,*) is a group ⟺ associativity + identity + inverses",
            },
            AudienceLevel.RESEARCH: {
                "summary": "Groups as objects in the category Grp with homomorphisms as morphisms.",
                "details": ("Group theory connects to topology (fundamental groups), "
                            "geometry (Lie groups), number theory (Galois groups), "
                            "and physics (gauge symmetries). Key theorems: Sylow, "
                            "classification of finite simple groups."),
                "examples": ["Lie groups", "Galois groups", "Homotopy groups π_n(X)"],
                "key_insights": ["Groups unify via category theory",
                                  "Classification of finite simple groups is complete"],
            },
        }

    def explain(self, topic: str, level: AudienceLevel = AudienceLevel.UNDERGRADUATE,
                context: Dict = None) -> Explanation:
        """Generate explanation for a topic at the given level."""
        template = self.templates.get(topic, {}).get(level)
        if template:
            return Explanation(
                topic=topic, level=level,
                summary=template.get("summary", f"Explanation of {topic}"),
                details=template.get("details", ""),
                examples=template.get("examples", []),
                prerequisites=template.get("prerequisites", []),
                key_insights=template.get("key_insights", []),
                formal_statement=template.get("formal_statement"),
                proof_sketch=template.get("proof_sketch"),
            )
        # Generate a generic explanation
        return Explanation(
            topic=topic, level=level,
            summary=f"{topic} ({level.value} level explanation)",
            details=f"A {level.value}-level treatment of {topic}.",
            examples=[], prerequisites=[], key_insights=[],
        )

    def add_explanation(self, topic: str, level: AudienceLevel, data: Dict):
        if topic not in self.templates:
            self.templates[topic] = {}
        self.templates[topic][level] = data

    def available_topics(self) -> List[str]:
        return list(self.templates.keys())

    def format_explanation(self, explanation: Explanation) -> str:
        lines = [
            f"{'=' * 60}",
            f"  {explanation.topic.upper()} ({explanation.level.value})",
            f"{'=' * 60}",
            f"\n{explanation.summary}\n",
            f"Details:\n{explanation.details}\n",
        ]
        if explanation.formal_statement:
            lines.append(f"Formal: {explanation.formal_statement}\n")
        if explanation.examples:
            lines.append("Examples:")
            for ex in explanation.examples:
                lines.append(f"  • {ex}")
        if explanation.key_insights:
            lines.append("\nKey Insights:")
            for ins in explanation.key_insights:
                lines.append(f"  ★ {ins}")
        if explanation.proof_sketch:
            lines.append(f"\nProof Sketch:\n{explanation.proof_sketch}")
        return "\n".join(lines)

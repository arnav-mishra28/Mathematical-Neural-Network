"""mnn.researcher.publication — Publication Engine + Research Roadmap.

Module 9: Publication Engine — automatically produce abstracts, introductions,
methods, results, and discussion. Output as papers, reports, or notebooks.

Module 10: Research Roadmap Engine — autonomously decide what to investigate
next. Current result → open question → new experiment → new result → repeat.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class PublicationFormat(Enum):
    PAPER = "paper"
    REPORT = "report"
    NOTEBOOK = "notebook"
    ABSTRACT = "abstract"


@dataclass
class Publication:
    """An automatically generated publication."""
    title: str
    format: PublicationFormat
    abstract: str = ""
    introduction: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusion: str = ""
    references: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def render(self) -> str:
        sections = [f"{'=' * 70}", f"  {self.title}", f"{'=' * 70}"]
        if self.abstract:
            sections.extend(["", "ABSTRACT", "-" * 40, self.abstract])
        if self.introduction:
            sections.extend(["", "1. INTRODUCTION", "-" * 40, self.introduction])
        if self.methods:
            sections.extend(["", "2. METHODS", "-" * 40, self.methods])
        if self.results:
            sections.extend(["", "3. RESULTS", "-" * 40, self.results])
        if self.discussion:
            sections.extend(["", "4. DISCUSSION", "-" * 40, self.discussion])
        if self.conclusion:
            sections.extend(["", "5. CONCLUSION", "-" * 40, self.conclusion])
        if self.references:
            sections.extend(["", "REFERENCES", "-" * 40])
            for i, ref in enumerate(self.references, 1):
                sections.append(f"  [{i}] {ref}")
        return "\n".join(sections)


class PublicationEngine:
    """Module 9: Generate publications from research results."""

    def __init__(self):
        self.publications: List[Publication] = []

    def generate(self, title: str, investigation: Dict,
                  fmt: PublicationFormat = PublicationFormat.REPORT) -> Publication:
        """Generate a publication from investigation results."""
        pub = Publication(title, fmt)

        # Abstract
        findings = investigation.get("findings", [])
        conjectures = investigation.get("conjectures", [])
        pub.abstract = self._generate_abstract(title, findings, conjectures)

        # Introduction
        objective = investigation.get("objective", "")
        background = investigation.get("background", "")
        pub.introduction = self._generate_intro(title, objective, background)

        # Methods
        steps = investigation.get("steps", [])
        pub.methods = self._generate_methods(steps)

        # Results
        pub.results = self._generate_results(findings)

        # Discussion
        pub.discussion = self._generate_discussion(findings, conjectures)

        # Conclusion
        pub.conclusion = self._generate_conclusion(findings, conjectures)

        # References
        pub.references = investigation.get("references", [])

        self.publications.append(pub)
        return pub

    def _generate_abstract(self, title: str, findings: List, conjectures: List) -> str:
        parts = [f"We investigate {title.lower()}."]
        if findings:
            parts.append(f"Our analysis reveals {len(findings)} key findings.")
        if conjectures:
            parts.append(f"We propose {len(conjectures)} new conjectures.")
        parts.append("This work contributes to the mathematical understanding of the studied structures.")
        return " ".join(parts)

    def _generate_intro(self, title: str, objective: str, background: str) -> str:
        parts = [f"The study of {title.lower()} is a fundamental area of mathematical research."]
        if objective:
            parts.append(f"Our objective is to {objective.lower()}.")
        if background:
            parts.append(background)
        return " ".join(parts)

    def _generate_methods(self, steps: List) -> str:
        if not steps:
            return "Standard mathematical and computational methods were employed."
        lines = ["The following methodology was employed:"]
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict):
                lines.append(f"  Step {i}: {step.get('action', '')} — {step.get('description', '')}")
            else:
                lines.append(f"  Step {i}: {step}")
        return "\n".join(lines)

    def _generate_results(self, findings: List) -> str:
        if not findings:
            return "No significant findings were obtained in this investigation."
        lines = [f"The investigation yielded {len(findings)} findings:"]
        for i, f in enumerate(findings, 1):
            lines.append(f"  Finding {i}: {f}")
        return "\n".join(lines)

    def _generate_discussion(self, findings: List, conjectures: List) -> str:
        parts = []
        if findings:
            parts.append(f"The {len(findings)} findings suggest significant mathematical structure.")
        if conjectures:
            parts.append(f"The {len(conjectures)} conjectures require further investigation.")
        parts.append("Future work should address the open questions raised by this research.")
        return " ".join(parts)

    def _generate_conclusion(self, findings: List, conjectures: List) -> str:
        parts = [f"This investigation produced {len(findings)} findings "
                 f"and {len(conjectures)} conjectures."]
        parts.append("The results contribute to the mathematical understanding of the studied domain.")
        return " ".join(parts)

    def summary(self) -> str:
        return f"PublicationEngine: {len(self.publications)} publications"


# ---- Module 10: Research Roadmap Engine ----

class ResearchPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPLORATORY = "exploratory"


@dataclass
class ResearchQuestion:
    """An automatically generated research question."""
    question: str
    priority: ResearchPriority
    source: str = ""  # what generated this question
    domain: str = ""
    estimated_difficulty: float = 0.5
    status: str = "open"  # open, investigating, resolved, deferred
    linked_discoveries: List[str] = field(default_factory=list)


class ResearchRoadmap:
    """Module 10: Autonomously decide what to investigate next."""

    def __init__(self):
        self.questions: List[ResearchQuestion] = []
        self.completed_topics: List[str] = []
        self.research_log: List[Dict] = []

    def generate_questions(self, findings: List[str] = None,
                            conjectures: List[str] = None,
                            discoveries: List[Dict] = None) -> List[ResearchQuestion]:
        """Generate next research questions from current state."""
        new_questions = []

        if findings:
            for f in findings:
                new_questions.append(ResearchQuestion(
                    f"Can we generalize: {f[:80]}?",
                    ResearchPriority.MEDIUM, source="finding_generalization"))
                new_questions.append(ResearchQuestion(
                    f"What are the boundary conditions for: {f[:80]}?",
                    ResearchPriority.LOW, source="finding_boundary"))

        if conjectures:
            for c in conjectures:
                new_questions.append(ResearchQuestion(
                    f"Can we prove or disprove: {c[:80]}?",
                    ResearchPriority.HIGH, source="conjecture_resolution"))

        if discoveries:
            for d in discoveries:
                dtype = d.get("type", "")
                title = d.get("title", "")
                if dtype == "new_connection":
                    new_questions.append(ResearchQuestion(
                        f"What is the deeper structure behind: {title}?",
                        ResearchPriority.CRITICAL, source="discovery_investigation"))
                else:
                    new_questions.append(ResearchQuestion(
                        f"What are the implications of: {title}?",
                        ResearchPriority.MEDIUM, source="discovery_implication"))

        self.questions.extend(new_questions)
        return new_questions

    def next_investigation(self) -> Optional[ResearchQuestion]:
        """Select the highest priority open question."""
        priority_order = [ResearchPriority.CRITICAL, ResearchPriority.HIGH,
                         ResearchPriority.MEDIUM, ResearchPriority.LOW,
                         ResearchPriority.EXPLORATORY]
        open_qs = [q for q in self.questions if q.status == "open"]
        for priority in priority_order:
            candidates = [q for q in open_qs if q.priority == priority]
            if candidates:
                return candidates[0]
        return None

    def start_investigation(self, question: ResearchQuestion):
        question.status = "investigating"
        self.research_log.append({
            "action": "start", "question": question.question,
            "timestamp": time.time()})

    def complete_investigation(self, question: ResearchQuestion, result: str = ""):
        question.status = "resolved"
        self.completed_topics.append(question.question)
        self.research_log.append({
            "action": "complete", "question": question.question,
            "result": result, "timestamp": time.time()})

    def open_questions(self) -> List[ResearchQuestion]:
        return [q for q in self.questions if q.status == "open"]

    def research_cycle(self) -> Dict:
        """Execute one full research cycle: select → plan → (return for execution)."""
        next_q = self.next_investigation()
        if next_q is None:
            return {"status": "no_open_questions"}
        self.start_investigation(next_q)
        return {
            "status": "investigating",
            "question": next_q.question,
            "priority": next_q.priority.value,
            "source": next_q.source,
        }

    def summary(self) -> str:
        counts = {}
        for q in self.questions:
            counts[q.status] = counts.get(q.status, 0) + 1
        return (f"ResearchRoadmap: {len(self.questions)} questions, "
                f"status={counts}, completed={len(self.completed_topics)}")

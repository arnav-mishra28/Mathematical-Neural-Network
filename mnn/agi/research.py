"""mnn.agi.research — Research Assistant + Dialogue System.

Module 9: Research Assistant Mode — study PDEs, build models, test hypotheses,
generate conjectures, produce reports.

Module 10: Mathematical Dialogue System — discuss definitions, proofs,
experiments, simulations while maintaining mathematical consistency.
"""
from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class InvestigationStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class InvestigationStep:
    """A single step in a research investigation."""
    action: str
    description: str
    result: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Investigation:
    """A mathematical research investigation."""
    topic: str
    objective: str
    status: InvestigationStatus = InvestigationStatus.ACTIVE
    steps: List[InvestigationStep] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    conjectures: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)

    def add_step(self, action: str, description: str, result: str = None):
        self.steps.append(InvestigationStep(action, description, result))

    def add_finding(self, finding: str):
        self.findings.append(finding)

    def add_conjecture(self, conjecture: str):
        self.conjectures.append(conjecture)

    def report(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  RESEARCH REPORT: {self.topic}",
            f"{'=' * 60}",
            f"\nObjective: {self.objective}",
            f"Status: {self.status.value}",
            f"\nSteps ({len(self.steps)}):",
        ]
        for i, s in enumerate(self.steps):
            lines.append(f"  [{i+1}] {s.action}: {s.description}")
            if s.result:
                lines.append(f"      → {s.result}")
        if self.findings:
            lines.append(f"\nFindings ({len(self.findings)}):")
            for f in self.findings:
                lines.append(f"  • {f}")
        if self.conjectures:
            lines.append(f"\nConjectures ({len(self.conjectures)}):")
            for c in self.conjectures:
                lines.append(f"  ? {c}")
        if self.open_questions:
            lines.append(f"\nOpen Questions:")
            for q in self.open_questions:
                lines.append(f"  ◇ {q}")
        return "\n".join(lines)


class ResearchAssistant:
    """Module 9: Conduct mathematical investigations."""

    def __init__(self):
        self.investigations: List[Investigation] = []
        self.current: Optional[Investigation] = None

    def start_investigation(self, topic: str, objective: str) -> Investigation:
        inv = Investigation(topic, objective)
        self.investigations.append(inv)
        self.current = inv
        inv.add_step("initialize", f"Begin investigation of: {topic}")
        return inv

    def literature_review(self, sources: List[str]) -> str:
        if self.current is None:
            return "No active investigation."
        review = f"Reviewed {len(sources)} sources: {', '.join(sources[:5])}"
        self.current.add_step("literature_review", review)
        return review

    def formulate_hypothesis(self, hypothesis: str) -> str:
        if self.current is None:
            return "No active investigation."
        self.current.add_conjecture(hypothesis)
        self.current.add_step("hypothesis", f"Proposed: {hypothesis}")
        return f"Hypothesis recorded: {hypothesis}"

    def run_experiment(self, name: str, experiment_fn: Callable = None,
                        params: Dict = None) -> Dict:
        if self.current is None:
            return {"error": "No active investigation."}
        result = {}
        if experiment_fn:
            try:
                result = experiment_fn(**(params or {}))
                self.current.add_step("experiment", f"Ran: {name}",
                                       str(result)[:200])
            except Exception as e:
                result = {"error": str(e)}
                self.current.add_step("experiment", f"Failed: {name}", str(e))
        else:
            self.current.add_step("experiment", f"Planned: {name}")
        return result

    def record_finding(self, finding: str):
        if self.current:
            self.current.add_finding(finding)
            self.current.add_step("finding", finding)

    def complete_investigation(self) -> Optional[str]:
        if self.current:
            self.current.status = InvestigationStatus.COMPLETED
            report = self.current.report()
            self.current = None
            return report
        return None


# ---- Module 10: Mathematical Dialogue System ----

class DialogueMode(Enum):
    EXPLORATION = "exploration"
    PROBLEM_SOLVING = "problem_solving"
    PROOF_DISCUSSION = "proof_discussion"
    CONCEPT_LEARNING = "concept_learning"
    SIMULATION = "simulation"


@dataclass
class DialogueTurn:
    """A single turn in mathematical dialogue."""
    role: str  # "user" or "assistant"
    content: str
    mode: DialogueMode = DialogueMode.EXPLORATION
    references: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MathDialogueSystem:
    """Module 10: Mathematical dialogue maintaining consistency."""

    def __init__(self):
        self.history: List[DialogueTurn] = []
        self.mode = DialogueMode.EXPLORATION
        self.context: Dict[str, Any] = {}
        self.defined_objects: Dict[str, str] = {}
        self.assumptions: List[str] = []

    def set_mode(self, mode: DialogueMode):
        self.mode = mode
        self.history.append(DialogueTurn(
            "system", f"Mode changed to: {mode.value}", mode))

    def user_input(self, message: str) -> str:
        self.history.append(DialogueTurn("user", message, self.mode))
        return self._process(message)

    def _process(self, message: str) -> str:
        msg = message.lower()

        # Definition handling
        if msg.startswith("define "):
            parts = message[7:].split(" as ", 1)
            if len(parts) == 2:
                name, defn = parts[0].strip(), parts[1].strip()
                self.defined_objects[name] = defn
                response = f"Defined: {name} := {defn}"
                self.history.append(DialogueTurn("assistant", response, self.mode))
                return response

        # Assumption handling
        if msg.startswith("assume "):
            assumption = message[7:].strip()
            self.assumptions.append(assumption)
            response = f"Assumption recorded: {assumption}"
            self.history.append(DialogueTurn("assistant", response, self.mode))
            return response

        # Query handling
        if msg.startswith("what is "):
            term = message[8:].strip().rstrip("?")
            if term in self.defined_objects:
                response = f"{term} := {self.defined_objects[term]}"
            else:
                response = f"{term} is not yet defined in this session."
            self.history.append(DialogueTurn("assistant", response, self.mode))
            return response

        # List context
        if "show definitions" in msg or "show context" in msg:
            lines = ["Current Definitions:"]
            for name, defn in self.defined_objects.items():
                lines.append(f"  {name} := {defn}")
            if self.assumptions:
                lines.append("Assumptions:")
                for a in self.assumptions:
                    lines.append(f"  • {a}")
            response = "\n".join(lines)
            self.history.append(DialogueTurn("assistant", response, self.mode))
            return response

        # Default
        response = f"[{self.mode.value}] Processing: {message}"
        self.history.append(DialogueTurn("assistant", response, self.mode))
        return response

    def transcript(self, last_n: int = None) -> str:
        turns = self.history[-last_n:] if last_n else self.history
        lines = []
        for t in turns:
            prefix = "USER" if t.role == "user" else "AGI"
            lines.append(f"[{prefix}] {t.content}")
        return "\n".join(lines)

    def consistency_check(self) -> Dict:
        """Check for obvious contradictions in assumptions."""
        issues = []
        for i, a1 in enumerate(self.assumptions):
            for j, a2 in enumerate(self.assumptions):
                if i < j and ("not " + a1 == a2 or a1 == "not " + a2):
                    issues.append(f"Contradiction: '{a1}' vs '{a2}'")
        return {"consistent": len(issues) == 0, "issues": issues}

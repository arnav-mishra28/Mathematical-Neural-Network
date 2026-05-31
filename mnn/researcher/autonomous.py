"""mnn.researcher.autonomous — Unified Autonomous Scientific Researcher.

Integrates all 10 modules into a single autonomous research loop:
Observe → Hypothesize → Experiment → Analyze → Publish → Generate New Questions → Repeat.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from .literature import LiteratureEngine, ResearchKnowledgeGraph, LiteratureSource, SourceType, ResearchNode, ResearchEdge, RelationType
from .hypothesis import HypothesisGenerator, ExperimentPlanner, HypothesisLevel
from .simulation import SimulationEngine, EvidenceScorer, EvidenceItem, EvidenceType, SimulationType
from .critique import SelfCritiqueEngine, DiscoveryEngine
from .publication import PublicationEngine, ResearchRoadmap, PublicationFormat


class AutonomousResearcher:
    """Autonomous Mathematical & Scientific Research System (AMSR).

    Unlike the AGI Assistant which waits for tasks, the AMSR:
    - Generates tasks autonomously
    - Creates hypotheses from observations
    - Designs and runs experiments
    - Critiques its own work
    - Decides what to investigate next
    """

    def __init__(self, name: str = "AMSR"):
        self.name = name
        # Module 1: Literature
        self.literature = LiteratureEngine()
        # Module 2: Knowledge Graph
        self.knowledge = ResearchKnowledgeGraph()
        # Module 3: Hypothesis Generator
        self.hypotheses = HypothesisGenerator()
        # Module 4: Experiment Planner
        self.experiments = ExperimentPlanner()
        # Module 5: Simulation Engine
        self.simulation = SimulationEngine()
        # Module 6: Evidence Scorer
        self.evidence = EvidenceScorer()
        # Module 7: Self-Critique
        self.critique = SelfCritiqueEngine()
        # Module 8: Discovery Engine
        self.discovery = DiscoveryEngine()
        # Module 9: Publication
        self.publication = PublicationEngine()
        # Module 10: Roadmap
        self.roadmap = ResearchRoadmap()
        # Research state
        self._cycle_count = 0

    def ingest_source(self, title: str, source_type: str, abstract: str = "",
                       concepts: List[str] = None, theorems: List[str] = None,
                       open_problems: List[str] = None, **kwargs):
        """Module 1: Ingest a literature source."""
        st = SourceType(source_type) if source_type in [e.value for e in SourceType] else SourceType.PAPER
        src = LiteratureSource(title, st, abstract=abstract,
                                key_concepts=concepts or [],
                                theorems=theorems or [],
                                open_problems=open_problems or [],
                                **kwargs)
        self.literature.add_source(src)
        # Auto-add to knowledge graph
        for thm in (theorems or []):
            self.knowledge.add_node(ResearchNode(thm, "theorem", "", thm, source=title))
        for concept in (concepts or []):
            self.knowledge.add_node(ResearchNode(concept, "concept", "", concept, source=title))

    def observe(self, observations: List[Dict]):
        """Module 3: Record observations and generate hypotheses."""
        self.hypotheses.observe(observations)
        return self.hypotheses.generate(observations)

    def plan_experiments(self, hypothesis_index: int = 0):
        """Module 4: Design experiments for a hypothesis."""
        if hypothesis_index < len(self.hypotheses.hypotheses):
            h = self.hypotheses.hypotheses[hypothesis_index]
            return self.experiments.plan_experiments(h, hypothesis_index)
        return []

    def run_simulation(self, name: str, sim_type: str = "custom",
                        parameters: Dict = None):
        """Module 5: Run a simulation."""
        st = SimulationType(sim_type) if sim_type in [e.value for e in SimulationType] else SimulationType.CUSTOM
        return self.simulation.run(name, st, parameters or {})

    def score_evidence(self, hypothesis_index: int) -> Dict:
        """Module 6: Score evidence for a hypothesis."""
        return self.evidence.score(hypothesis_index)

    def self_critique(self, statement: str, evidence: List[Dict],
                       assumptions: List[str] = None):
        """Module 7: Critique a hypothesis or result."""
        return self.critique.critique_hypothesis(statement, evidence, assumptions)

    def detect_discoveries(self, patterns: List[Dict]):
        """Module 8: Analyze patterns for discoveries."""
        return self.discovery.analyze_patterns(patterns)

    def publish(self, title: str, investigation: Dict,
                 fmt: str = "report"):
        """Module 9: Generate a publication."""
        pfmt = PublicationFormat(fmt) if fmt in [e.value for e in PublicationFormat] else PublicationFormat.REPORT
        return self.publication.generate(title, investigation, pfmt)

    def next_question(self):
        """Module 10: Get next research question."""
        return self.roadmap.next_investigation()

    def research_cycle(self, observations: List[Dict] = None) -> Dict:
        """Execute one full autonomous research cycle."""
        self._cycle_count += 1
        cycle = {"cycle": self._cycle_count, "steps": []}

        # Step 1: Generate hypotheses from observations
        if observations:
            hyps = self.observe(observations)
            cycle["steps"].append({"action": "hypothesize",
                                    "count": len(hyps)})
        else:
            hyps = self.hypotheses.open_hypotheses()

        # Step 2: Plan experiments
        all_experiments = []
        for i, h in enumerate(hyps[:3]):  # top 3
            idx = self.hypotheses.hypotheses.index(h) if h in self.hypotheses.hypotheses else i
            exps = self.plan_experiments(idx)
            all_experiments.extend(exps)
        cycle["steps"].append({"action": "plan_experiments",
                                "count": len(all_experiments)})

        # Step 3: Critique
        for h in hyps[:3]:
            critiques = self.self_critique(h.statement, [], [])
            cycle["steps"].append({"action": "critique",
                                    "issues": len(critiques)})

        # Step 4: Generate new questions
        findings = [h.statement for h in hyps if h.confidence > 0.6]
        new_qs = self.roadmap.generate_questions(findings=findings)
        cycle["steps"].append({"action": "generate_questions",
                                "count": len(new_qs)})

        # Step 5: Select next investigation
        next_q = self.roadmap.next_investigation()
        if next_q:
            cycle["next_investigation"] = next_q.question

        return cycle

    def status(self) -> Dict:
        return {
            "name": self.name,
            "cycle_count": self._cycle_count,
            "literature": self.literature.summary(),
            "knowledge_graph": self.knowledge.summary(),
            "hypotheses": self.hypotheses.summary(),
            "experiments": self.experiments.summary(),
            "simulations": self.simulation.summary(),
            "critiques": self.critique.summary(),
            "discoveries": self.discovery.summary(),
            "publications": self.publication.summary(),
            "roadmap": self.roadmap.summary(),
        }

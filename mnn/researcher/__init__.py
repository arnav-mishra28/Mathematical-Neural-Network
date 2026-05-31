"""
mnn.researcher — Autonomous Scientific Researcher

An autonomous mathematical & scientific research system (AMSR).
Generates tasks, creates hypotheses, designs experiments, critiques work,
and decides what to investigate next.

Submodules:
  literature   — Literature Engine + Research Knowledge Graph (Modules 1-2)
  hypothesis   — Hypothesis Generator + Experiment Planner (Modules 3-4)
  simulation   — Simulation Engine + Evidence Scoring (Modules 5-6)
  critique     — Self-Critique Engine + Discovery Engine (Modules 7-8)
  publication  — Publication Engine + Research Roadmap (Modules 9-10)
  autonomous   — Unified AutonomousResearcher
"""
from .literature import (
    LiteratureEngine, LiteratureSource, SourceType,
    ResearchKnowledgeGraph, ResearchNode, ResearchEdge, RelationType,
)
from .hypothesis import (
    HypothesisGenerator, Hypothesis, HypothesisLevel, HypothesisStatus,
    ExperimentPlanner, Experiment, ExperimentType,
)
from .simulation import (
    SimulationEngine, SimulationResult, SimulationType,
    EvidenceScorer, EvidenceItem, EvidenceType,
)
from .critique import (
    SelfCritiqueEngine, Critique, CritiqueType,
    DiscoveryEngine, Discovery, DiscoveryType,
)
from .publication import (
    PublicationEngine, Publication, PublicationFormat,
    ResearchRoadmap, ResearchQuestion, ResearchPriority,
)
from .autonomous import AutonomousResearcher

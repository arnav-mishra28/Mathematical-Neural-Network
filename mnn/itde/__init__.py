"""
mnn.itde — Interactive Theorem-Discovery Environment

A collaborative mathematical laboratory where humans and AI discover
mathematics together. Combines workspace, theorem canvas, conjecture
playground, proof assistant, knowledge graph explorer, research notebook,
AI co-researcher, multi-agent system, and discovery dashboard.

Submodules:
  workspace        — Mathematical Workspace + Live Theorem Canvas (Modules 1-2)
  conjecture_proof — Conjecture Playground + Proof Assistant (Modules 3-4)
  explorer         — Knowledge Graph Explorer + Research Notebook (Modules 6-7)
  collaboration    — AI Co-Researcher + Multi-Agent + Dashboard (Modules 8-10)
  environment      — Unified TheoremDiscoveryEnvironment
"""
from .workspace import MathWorkspace, MathObject, TheoremCanvas, CanvasNode
from .conjecture_proof import ConjecturePlayground, ProofAssistant, ProofSuggestion
from .explorer import KnowledgeGraphExplorer, ResearchNotebook
from .collaboration import (
    AICoResearcher, MultiAgentSystem, MathAgent, AgentSpecialty,
    DiscoveryDashboard, DashboardEntry,
)
from .environment import TheoremDiscoveryEnvironment

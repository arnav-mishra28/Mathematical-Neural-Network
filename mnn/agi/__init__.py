"""
mnn.agi — Mathematical AGI Assistant

A mathematical research agent capable of learning, reasoning, discovering,
verifying, explaining, planning, and collaborating.

Submodules:
  knowledge   — Knowledge Layer: math knowledge graphs (Module 1)
  memory      — Mathematical Memory: concepts, proofs, research (Module 2)
  reasoning   — Reasoning + Conjecture Engines (Modules 3-4)
  planner     — Proof Strategy + Mathematical Planner (Modules 5-6)
  explanation — Explanation Engine: multi-level teaching (Module 7)
  research    — Research Assistant + Dialogue System (Modules 9-10)
  assistant   — Unified MathAGIAssistant (all modules)
"""
from .knowledge import MathKnowledgeGraph, KnowledgeNode, KnowledgeType, Domain
from .memory import MathematicalMemory, ConceptMemory, ProofMemory, ResearchMemory
from .reasoning import (
    HybridReasoner, SymbolicReasoner, GeometricReasoner,
    ConjectureEngine, Conjecture, ReasoningChain,
)
from .planner import ProofStrategyEngine, MathematicalPlanner, ProofStrategy
from .explanation import ExplanationEngine, AudienceLevel, Explanation
from .research import ResearchAssistant, MathDialogueSystem, Investigation
from .assistant import MathAGIAssistant

"""mnn.agi.assistant — Mathematical AGI Assistant (Unified Agent).

Module 8 (Visualization) is integrated from existing mnn.visualization.

This module unifies all 10 components into a single Mathematical AGI Assistant.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from .knowledge import MathKnowledgeGraph, KnowledgeNode, KnowledgeType, Domain
from .memory import MathematicalMemory
from .reasoning import HybridReasoner, ConjectureEngine
from .planner import ProofStrategyEngine, MathematicalPlanner
from .explanation import ExplanationEngine, AudienceLevel
from .research import ResearchAssistant, MathDialogueSystem, DialogueMode


class MathAGIAssistant:
    """Mathematical AGI Assistant — a unified mathematical research agent.

    Capabilities:
      1. Learn: ingest mathematical knowledge
      2. Reason: derive implications (symbolic + geometric + categorical)
      3. Discover: generate conjectures
      4. Verify: test hypotheses
      5. Explain: teach at multiple levels
      6. Research: investigate open problems
      7. Plan: decompose goals into lemma chains
      8. Dialogue: mathematical conversation with consistency
    """

    def __init__(self, name: str = "MathAGI"):
        self.name = name
        # Module 1: Knowledge
        self.knowledge = MathKnowledgeGraph(name)
        # Module 2: Memory
        self.memory = MathematicalMemory()
        # Module 3: Reasoning
        self.reasoner = HybridReasoner()
        # Module 4: Conjecture
        self.conjectures = ConjectureEngine()
        # Module 5: Proof Strategy
        self.strategies = ProofStrategyEngine()
        # Module 6: Planner
        self.planner = MathematicalPlanner()
        # Module 7: Explanation
        self.explainer = ExplanationEngine()
        # Module 9: Research
        self.research = ResearchAssistant()
        # Module 10: Dialogue
        self.dialogue = MathDialogueSystem()

    def learn(self, name: str, kind: str, domain: str, statement: str,
              dependencies: List[str] = None, tags: List[str] = None):
        """Ingest mathematical knowledge."""
        kind_enum = KnowledgeType(kind) if kind in [e.value for e in KnowledgeType] else KnowledgeType.DEFINITION
        domain_enum = Domain(domain) if domain in [e.value for e in Domain] else Domain.ALGEBRA
        node = KnowledgeNode(name, kind_enum, domain_enum, statement,
                              dependencies or [], tags or [domain])
        self.knowledge.add_node(node)
        self.memory.concepts.store(name, statement, tags or [domain])

    def reason(self, goal: str, context: Dict = None):
        """Derive implications using hybrid reasoning."""
        chain = self.reasoner.reason(goal, context)
        return chain

    def conjecture(self, observations: List[Dict]):
        """Generate conjectures from observations."""
        return self.conjectures.observe_and_conjecture(observations)

    def suggest_proof(self, tags: List[str], top_k: int = 3):
        """Suggest proof strategies for a given domain."""
        return self.strategies.suggest_strategies(tags, top_k)

    def plan_proof(self, theorem: str, lemmas: List[Dict]):
        """Create a proof plan with sub-goals."""
        self.planner.set_goal(theorem, f"Prove: {theorem}")
        for lem in lemmas:
            self.planner.add_subgoal(
                lem["name"], lem.get("description", ""),
                lem.get("parent", theorem),
                lem.get("dependencies", []),
                lem.get("strategy"))
        return self.planner.plan_tree()

    def explain(self, topic: str, level: str = "undergraduate"):
        """Explain a topic at a given audience level."""
        level_enum = AudienceLevel(level)
        expl = self.explainer.explain(topic, level_enum)
        return self.explainer.format_explanation(expl)

    def investigate(self, topic: str, objective: str):
        """Start a research investigation."""
        return self.research.start_investigation(topic, objective)

    def chat(self, message: str) -> str:
        """Mathematical dialogue."""
        return self.dialogue.user_input(message)

    def status(self) -> Dict:
        """Full system status."""
        return {
            "name": self.name,
            "knowledge": self.knowledge.summary(),
            "memory": self.memory.summary(),
            "conjectures": self.conjectures.summary(),
            "planner": self.planner.progress(),
            "investigations": len(self.research.investigations),
            "dialogue_turns": len(self.dialogue.history),
        }

    def initialize(self):
        """Seed with standard knowledge and proof strategies."""
        self.knowledge.add_standard_knowledge()
        # Store strategies in memory
        for s in self.strategies.strategies:
            self.memory.proofs.store_strategy(
                s.name, s.description, s.applicability)
        return self

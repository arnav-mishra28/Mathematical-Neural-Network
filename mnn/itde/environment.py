"""mnn.itde.environment — Unified Interactive Theorem-Discovery Environment.

Integrates all 10 modules into a single collaborative mathematical laboratory.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from .workspace import MathWorkspace, TheoremCanvas
from .conjecture_proof import ConjecturePlayground, ProofAssistant
from .explorer import KnowledgeGraphExplorer, ResearchNotebook
from .collaboration import AICoResearcher, MultiAgentSystem, DiscoveryDashboard


class TheoremDiscoveryEnvironment:
    """Interactive Theorem-Discovery Environment (ITDE).

    A collaborative mathematical laboratory where humans and AI
    discover mathematics together.

    Capabilities:
        1. Workspace: create/inspect math objects
        2. Canvas: live theorem graphs
        3. Conjectures: interactive conjecture playground
        4. Proofs: collaborative proof assistant
        5. Visualization: (delegates to mnn.visualization)
        6. Explorer: knowledge graph navigation
        7. Notebook: reproducible research notebook
        8. CoResearcher: AI-driven investigation
        9. MultiAgent: specialized agent collaboration
        10. Dashboard: discovery tracking
    """

    def __init__(self, name: str = "ITDE"):
        self.name = name
        # Module 1: Workspace
        self.workspace = MathWorkspace(name)
        # Module 2: Theorem Canvas
        self.canvas = TheoremCanvas()
        # Module 3: Conjecture Playground
        self.conjectures = ConjecturePlayground()
        # Module 4: Proof Assistant
        self.proofs = ProofAssistant()
        # Module 6: Knowledge Graph Explorer
        self.explorer = KnowledgeGraphExplorer()
        # Module 7: Research Notebook
        self.notebook = ResearchNotebook(f"{name} Notebook")
        # Module 8: AI Co-Researcher
        self.co_researcher = AICoResearcher()
        # Module 9: Multi-Agent System
        self.agents = MultiAgentSystem()
        # Module 10: Discovery Dashboard
        self.dashboard = DiscoveryDashboard()

    # ---- Workspace (Module 1) ----
    def create_object(self, name: str, obj_type: str, description: str = "",
                       **props):
        """Create a mathematical object in the workspace."""
        obj = self.workspace.create(name, obj_type, description, **props)
        self.notebook.add_markdown(f"Created {obj_type}: {name}")
        return obj

    def inspect(self, name: str) -> str:
        """Inspect a workspace object."""
        return self.workspace.inspect(name)

    # ---- Theorem Canvas (Module 2) ----
    def add_theorem(self, name: str, statement: str, depends: List[str] = None):
        """Add a theorem to the canvas."""
        node = self.canvas.add_theorem(name, statement, depends)
        self.explorer.add_node(name, "theorem", "", statement)
        return node

    def add_definition(self, name: str, statement: str):
        """Add a definition to the canvas."""
        node = self.canvas.add_definition(name, statement)
        self.explorer.add_node(name, "definition", "", statement)
        return node

    def view_canvas(self) -> str:
        """View the theorem canvas."""
        return self.canvas.render_tree()

    # ---- Conjecture Playground (Module 3) ----
    def hypothesize(self, observation: str):
        """Generate conjectures from an observation."""
        conjs = self.conjectures.generate_conjectures(observation)
        for c in conjs:
            self.dashboard.add_entry(
                c.statement[:60], "conjecture", c.confidence)
        return conjs

    # ---- Proof Assistant (Module 4) ----
    def suggest_proof(self, goal: str, tags: List[str] = None):
        """Get proof strategy suggestions."""
        return self.proofs.suggest(goal, tags)

    # ---- Knowledge Graph (Module 6) ----
    def explore_topic(self, name: str) -> Dict:
        """Explore a topic in the knowledge graph."""
        return self.explorer.explore(name)

    # ---- AI Co-Researcher (Module 8) ----
    def investigate(self, topic: str, context: Dict = None):
        """AI investigates a topic autonomously."""
        report = self.co_researcher.investigate(topic, context)
        self.notebook.add_experiment(f"Investigation: {topic}", report.render())
        return report

    # ---- Multi-Agent (Module 9) ----
    def discuss(self, topic: str) -> List[str]:
        """All agents discuss a topic."""
        return self.agents.discuss(topic)

    def debate(self, proposition: str) -> Dict:
        """Agents debate a proposition."""
        return self.agents.debate(proposition)

    # ---- Status ----
    def status(self) -> Dict:
        return {
            "name": self.name,
            "workspace": self.workspace.summary(),
            "canvas": self.canvas.summary(),
            "conjectures": self.conjectures.summary(),
            "proofs": self.proofs.summary(),
            "explorer": self.explorer.summary(),
            "notebook": self.notebook.summary(),
            "co_researcher": self.co_researcher.summary(),
            "agents": self.agents.summary(),
            "dashboard": self.dashboard.summary(),
        }

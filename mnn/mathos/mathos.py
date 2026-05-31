"""mnn.mathos.mathos — Unified Mathematical Operating System.

The culmination: unifies kernel, memory, agents, knowledge graph,
simulation, discovery, proof, visualization, apps, and cloud into
a single platform for mathematical intelligence.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from .kernel import MathKernel, MathMemorySystem
from .agents import AgentRuntime, KnowledgeGraphOS
from .simulation import SimulationSubsystem, DiscoverySubsystem
from .proof import ProofSubsystem, VisualizationEngine
from .apps import AppEcosystem, MathApp, AppCategory, MathCloud


class MathOS:
    """AI-Powered Mathematical Operating System.

    An operating system for mathematical intelligence that manages:
    theorems, proofs, conjectures, simulations, agents, and research.

    Layers:
        1. Kernel — mathematical object model
        2. Memory — accumulated mathematical experience
        3. Agents — specialized mathematical agents
        4. Knowledge Graph — evolving mathematical universe
        5. Simulation — dynamical systems, PDEs, chaos, geometry
        6. Discovery — autonomous research loop
        7. Proof — interactive/autonomous theorem proving
        8. Visualization — 3D manifolds, networks, dynamics
        9. Apps — research, education, engineering, physics, AI
        10. Cloud — distributed mathematical intelligence
    """

    def __init__(self, name: str = "MathOS"):
        self.name = name
        self.version = "1.3.0"
        # Layer 1: Kernel
        self.kernel = MathKernel()
        # Layer 2: Memory
        self.memory = MathMemorySystem()
        # Layer 3: Agent Runtime
        self.agents = AgentRuntime()
        # Layer 4: Knowledge Graph
        self.knowledge = KnowledgeGraphOS()
        # Layer 5: Simulation
        self.simulation = SimulationSubsystem()
        # Layer 6: Discovery
        self.discovery = DiscoverySubsystem()
        # Layer 7: Proof
        self.proof = ProofSubsystem()
        # Layer 8: Visualization
        self.viz = VisualizationEngine()
        # Layer 9: Apps
        self.apps = AppEcosystem()
        # Layer 10: Cloud
        self.cloud = MathCloud(name)

    # ---- Layer 1: Kernel ----
    def create(self, name: str, math_type: str, data: Any = None,
               domain: str = "", **props):
        """Create a mathematical object."""
        entity = self.kernel.create(name, math_type, data, domain, **props)
        self.memory.store("concept", name, f"{math_type}: {name}", tags=[domain])
        self.knowledge.add_node(name, math_type, domain, str(props)[:100])
        return entity

    # ---- Layer 3: Agents ----
    def dispatch(self, task: str) -> Dict[str, str]:
        """Send a task to all agents."""
        return self.agents.dispatch(task)

    # ---- Layer 5: Simulation ----
    def simulate(self, name: str, domain: str = "custom",
                  params: Dict = None):
        """Run a simulation."""
        return self.simulation.run(name, domain, params)

    # ---- Layer 6: Discovery ----
    def research_cycle(self, data: Dict = None) -> Dict:
        """Run one autonomous research cycle."""
        return self.discovery.research_cycle(data)

    # ---- Layer 7: Proof ----
    def prove(self, theorem: str, strategy: str = "induction",
              mode: str = "interactive"):
        """Start a proof attempt."""
        return self.proof.plan_proof(theorem, strategy, mode)

    # ---- Layer 8: Visualization ----
    def visualize_manifold(self, name: str = "torus", params: Dict = None,
                            save_path: str = None):
        """Render a 3D manifold."""
        return self.viz.plot_manifold_3d(name, params, save_path)

    def visualize_network(self, save_path: str = None):
        """Render the knowledge graph as a network."""
        nodes = {n: {"domain": node.domain} for n, node in self.knowledge.nodes.items()}
        edges = [{"source": e.source, "target": e.target}
                 for e in self.knowledge.edges]
        return self.viz.plot_network(nodes, edges, save_path)

    def visualize_dynamics(self, trajectory: np.ndarray, name: str = "dynamics",
                            save_path: str = None):
        """Plot a dynamical system trajectory."""
        return self.viz.plot_dynamics(trajectory, name, save_path)

    # ---- Layer 9: Apps ----
    def install_app(self, app: MathApp):
        self.apps.install(app)

    def run_app(self, name: str, **kwargs):
        return self.apps.run_app(name, **kwargs)

    # ---- Layer 10: Cloud ----
    def cloud_submit(self, description: str, capability: str = ""):
        return self.cloud.submit_task(description, capability)

    def cloud_share(self, key: str, value: Any):
        self.cloud.share_knowledge(key, value)

    # ---- Status ----
    def boot(self) -> str:
        """Boot MathOS — initialize all layers."""
        lines = [
            f"╔{'═' * 58}╗",
            f"║  MathOS v{self.version} — Mathematical Operating System     ║",
            f"╠{'═' * 58}╣",
            f"║  Layer 1: {self.kernel.summary():<47}║",
            f"║  Layer 2: {self.memory.summary():<47}║",
            f"║  Layer 3: {self.agents.summary():<47}║",
            f"║  Layer 4: {self.knowledge.summary():<47}║",
            f"║  Layer 5: {self.simulation.summary():<47}║",
            f"║  Layer 6: {self.discovery.summary():<47}║",
            f"║  Layer 7: {self.proof.summary():<47}║",
            f"║  Layer 8: {self.viz.summary():<47}║",
            f"║  Layer 9: {self.apps.summary():<47}║",
            f"║  Layer 10: {self.cloud.summary():<46}║",
            f"╚{'═' * 58}╝",
        ]
        return "\n".join(lines)

    def status(self) -> Dict:
        return {
            "name": self.name, "version": self.version,
            "kernel": self.kernel.summary(),
            "memory": self.memory.summary(),
            "agents": self.agents.summary(),
            "knowledge": self.knowledge.summary(),
            "simulation": self.simulation.summary(),
            "discovery": self.discovery.summary(),
            "proof": self.proof.summary(),
            "visualization": self.viz.summary(),
            "apps": self.apps.summary(),
            "cloud": self.cloud.summary(),
        }

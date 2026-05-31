"""
mnn.mathos — AI-Powered Mathematical Operating System

The culmination of the MNN framework. An operating system for mathematical
intelligence that manages theorems, proofs, conjectures, simulations,
agents, and scientific research programs.

Layers:
  kernel       — Mathematical Kernel + Memory System (Layers 1-2)
  agents       — Agent Runtime + Knowledge Graph OS (Layers 3-4)
  simulation   — Simulation + Discovery Subsystems (Layers 5-6)
  proof        — Proof Subsystem + Visualization Engine (Layers 7-8)
  apps         — Application Ecosystem + Mathematical Cloud (Layers 9-10)
  mathos       — Unified MathOS
"""
from .kernel import MathKernel, MathEntity, MathType, MathMemorySystem, MemoryRecord
from .agents import AgentRuntime, MathOSAgent, AgentRole, KnowledgeGraphOS, KGNode, KGEdge
from .simulation import SimulationSubsystem, SimResult, DiscoverySubsystem, DiscoveryItem
from .proof import ProofSubsystem, ProofAttempt, ProofMode, VisualizationEngine
from .apps import AppEcosystem, MathApp, AppCategory, MathCloud, CloudTask
from .mathos import MathOS

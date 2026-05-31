"""mnn.mathos.agents — Agent Runtime (Layer 3) + Knowledge Graph OS (Layer 4).

Layer 3: Agent Runtime — specialized mathematical agents that communicate
via messages, proofs, conjectures, and evidence.

Layer 4: Knowledge Graph OS — continuously evolving mathematical universe.
"""
from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============= Layer 3: Agent Runtime =============

class AgentRole(Enum):
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    PDE = "pde"
    TOPOLOGY = "topology"
    DISCOVERY = "discovery"
    PROOF = "proof"
    ANALYSIS = "analysis"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentMsg:
    """Message between mathematical agents."""
    sender: str
    receiver: str
    content: str
    msg_type: str = "statement"  # statement, proof, conjecture, evidence, query
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class MathOSAgent:
    """A specialized mathematical agent in the runtime."""
    name: str
    role: AgentRole
    status: str = "idle"  # idle, working, waiting
    knowledge: List[str] = field(default_factory=list)
    inbox: List[AgentMsg] = field(default_factory=list)
    outbox: List[AgentMsg] = field(default_factory=list)
    task_history: List[Dict] = field(default_factory=list)

    def receive(self, msg: AgentMsg):
        self.inbox.append(msg)

    def send(self, receiver: str, content: str, msg_type: str = "statement") -> AgentMsg:
        msg = AgentMsg(self.name, receiver, content, msg_type)
        self.outbox.append(msg)
        return msg

    def process_task(self, task: str) -> str:
        """Process a task based on agent's specialty."""
        self.status = "working"
        task_lower = task.lower()
        result = ""

        if self.role == AgentRole.ALGEBRA:
            if any(w in task_lower for w in ["group", "ring", "field", "symmetry"]):
                result = f"Algebraic analysis: identified structure in '{task}'. Check group actions."
            else:
                result = f"Algebraic perspective on '{task}': formulate algebraically."
        elif self.role == AgentRole.GEOMETRY:
            if any(w in task_lower for w in ["manifold", "curvature", "surface", "geodesic"]):
                result = f"Geometric analysis: manifold structure in '{task}'. Compute curvature."
            else:
                result = f"Geometric interpretation of '{task}'."
        elif self.role == AgentRole.PDE:
            if any(w in task_lower for w in ["equation", "differential", "heat", "wave"]):
                result = f"PDE analysis: classify and solve '{task}' using spectral methods."
            else:
                result = f"Differential formulation of '{task}'."
        elif self.role == AgentRole.DISCOVERY:
            result = f"Scanning for patterns and novel connections in '{task}'."
        elif self.role == AgentRole.PROOF:
            result = f"Proof search for '{task}': evaluating induction, contradiction, spectral methods."
        else:
            result = f"[{self.role.value}] Processing: {task}"

        self.task_history.append({"task": task, "result": result, "timestamp": time.time()})
        self.status = "idle"
        return result


class AgentRuntime:
    """Layer 3: Runtime environment for mathematical agents."""

    def __init__(self):
        self.agents: Dict[str, MathOSAgent] = {}
        self.message_bus: List[AgentMsg] = []
        self._init_default_agents()

    def _init_default_agents(self):
        defaults = [
            ("AlgebraAgent", AgentRole.ALGEBRA),
            ("GeometryAgent", AgentRole.GEOMETRY),
            ("PDEAgent", AgentRole.PDE),
            ("DiscoveryAgent", AgentRole.DISCOVERY),
            ("ProofAgent", AgentRole.PROOF),
        ]
        for name, role in defaults:
            self.agents[name] = MathOSAgent(name, role)

    def register_agent(self, name: str, role: str) -> MathOSAgent:
        r = AgentRole(role) if role in [e.value for e in AgentRole] else AgentRole.ORCHESTRATOR
        agent = MathOSAgent(name, r)
        self.agents[name] = agent
        return agent

    def dispatch(self, task: str) -> Dict[str, str]:
        """Dispatch a task to all relevant agents."""
        results = {}
        for name, agent in self.agents.items():
            result = agent.process_task(task)
            results[name] = result
        return results

    def send_message(self, sender: str, receiver: str, content: str,
                      msg_type: str = "statement"):
        msg = AgentMsg(sender, receiver, content, msg_type)
        self.message_bus.append(msg)
        if receiver in self.agents:
            self.agents[receiver].receive(msg)

    def broadcast(self, sender: str, content: str, msg_type: str = "statement"):
        for name in self.agents:
            if name != sender:
                self.send_message(sender, name, content, msg_type)

    def summary(self) -> str:
        agent_info = {n: a.role.value for n, a in self.agents.items()}
        return f"AgentRuntime: {len(self.agents)} agents={agent_info}, msgs={len(self.message_bus)}"


# ============= Layer 4: Knowledge Graph OS =============

@dataclass
class KGNode:
    """A node in the knowledge graph."""
    name: str
    kind: str  # definition, theorem, proof, paper, simulation
    domain: str
    content: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class KGEdge:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation: str  # implies, depends_on, generalizes, contradicts, equivalent
    weight: float = 1.0


class KnowledgeGraphOS:
    """Layer 4: A continuously evolving mathematical universe."""

    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self._fwd: Dict[str, List[Tuple[str, str]]] = {}
        self._rev: Dict[str, List[Tuple[str, str]]] = {}

    def add_node(self, name: str, kind: str, domain: str, content: str,
                  tags: List[str] = None):
        self.nodes[name] = KGNode(name, kind, domain, content, tags or [])
        self._fwd.setdefault(name, [])
        self._rev.setdefault(name, [])

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0):
        self.edges.append(KGEdge(source, target, relation, weight))
        self._fwd.setdefault(source, []).append((target, relation))
        self._rev.setdefault(target, []).append((source, relation))

    def neighbors(self, name: str) -> List[Tuple[str, str]]:
        return self._fwd.get(name, [])

    def predecessors(self, name: str) -> List[Tuple[str, str]]:
        return self._rev.get(name, [])

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        if start == end:
            return [start]
        visited = {start}
        queue = [(start, [start])]
        while queue:
            current, path = queue.pop(0)
            for nb, _ in self._fwd.get(current, []):
                if nb == end:
                    return path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def cross_domain(self) -> List[Dict]:
        links = []
        for e in self.edges:
            s, t = self.nodes.get(e.source), self.nodes.get(e.target)
            if s and t and s.domain != t.domain:
                links.append({"source": e.source, "target": e.target,
                              "domains": [s.domain, t.domain],
                              "relation": e.relation})
        return links

    def by_domain(self, domain: str) -> List[KGNode]:
        return [n for n in self.nodes.values() if n.domain == domain]

    def search(self, query: str) -> List[KGNode]:
        q = query.lower()
        return [n for n in self.nodes.values()
                if q in n.name.lower() or q in n.content.lower()
                or any(q in t for t in n.tags)]

    def domains(self) -> List[str]:
        return list(set(n.domain for n in self.nodes.values()))

    def summary(self) -> str:
        domain_counts = {}
        for n in self.nodes.values():
            domain_counts[n.domain] = domain_counts.get(n.domain, 0) + 1
        return (f"KGOS: {len(self.nodes)} nodes, {len(self.edges)} edges, "
                f"domains={domain_counts}")

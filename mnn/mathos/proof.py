"""mnn.mathos.proof — Proof Subsystem (Layer 7) + Visualization Engine (Layer 8).

Layer 7: Proof Subsystem — interactive, semi-autonomous, and autonomous
proof modes. Proof planning, counterexample search, symbolic reasoning.

Layer 8: Visualization Engine — render manifolds, networks, dynamics,
theorem graphs using matplotlib. Reference: 3D manifold gallery style.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============= Layer 7: Proof Subsystem =============

class ProofMode(Enum):
    INTERACTIVE = "interactive"      # Human + AI
    SEMI_AUTONOMOUS = "semi_autonomous"  # AI-guided
    AUTONOMOUS = "autonomous"        # Agent-driven


class ProofStatus(Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    PROVED = "proved"
    FAILED = "failed"
    COUNTEREXAMPLE_FOUND = "counterexample_found"


@dataclass
class ProofStep:
    """A step in a proof."""
    description: str
    justification: str = ""
    status: str = "pending"  # pending, verified, invalid


@dataclass
class ProofAttempt:
    """A proof attempt with strategy and steps."""
    theorem: str
    strategy: str
    mode: ProofMode
    steps: List[ProofStep] = field(default_factory=list)
    status: ProofStatus = ProofStatus.PLANNING
    counterexamples: List[str] = field(default_factory=list)

    def add_step(self, description: str, justification: str = "") -> int:
        self.steps.append(ProofStep(description, justification))
        return len(self.steps) - 1

    def verify_step(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps[index].status = "verified"

    def render(self) -> str:
        lines = [f"╔═ PROOF: {self.theorem}",
                 f"║  Strategy: {self.strategy} | Mode: {self.mode.value}",
                 f"║  Status: {self.status.value}"]
        for i, s in enumerate(self.steps):
            sym = {"pending": "○", "verified": "●", "invalid": "✗"}
            lines.append(f"║  {sym.get(s.status, '·')} [{i}] {s.description}")
            if s.justification:
                lines.append(f"║       ↳ {s.justification}")
        if self.counterexamples:
            lines.append(f"║  Counterexamples: {self.counterexamples}")
        lines.append("╚═")
        return "\n".join(lines)


class ProofSubsystem:
    """Layer 7: Proof planning, search, and verification."""

    def __init__(self):
        self.attempts: List[ProofAttempt] = []
        self.strategies = [
            "induction", "contradiction", "construction", "symmetry",
            "spectral", "categorical", "geometric", "probabilistic",
        ]

    def plan_proof(self, theorem: str, strategy: str = "induction",
                    mode: str = "interactive") -> ProofAttempt:
        pm = ProofMode(mode) if mode in [e.value for e in ProofMode] else ProofMode.INTERACTIVE
        attempt = ProofAttempt(theorem, strategy, pm, status=ProofStatus.IN_PROGRESS)
        self.attempts.append(attempt)
        return attempt

    def suggest_strategies(self, theorem: str, tags: List[str] = None) -> List[str]:
        """Suggest strategies based on theorem and tags."""
        suggestions = list(self.strategies)
        theorem_lower = theorem.lower()
        priority = []
        if any(w in theorem_lower for w in ["natural", "integer", "n+1"]):
            priority.append("induction")
        if any(w in theorem_lower for w in ["no", "impossible", "cannot"]):
            priority.append("contradiction")
        if any(w in theorem_lower for w in ["manifold", "curvature"]):
            priority.append("geometric")
        if any(w in theorem_lower for w in ["eigenvalue", "spectrum"]):
            priority.append("spectral")
        return priority + [s for s in suggestions if s not in priority]

    def search_counterexample(self, predicate: Callable, domain: List,
                                theorem: str = "") -> Optional[Any]:
        """Search for counterexamples."""
        for x in domain:
            try:
                if not predicate(x):
                    return x
            except Exception:
                pass
        return None

    def complete(self, index: int, success: bool = True):
        if 0 <= index < len(self.attempts):
            self.attempts[index].status = (
                ProofStatus.PROVED if success else ProofStatus.FAILED)

    def summary(self) -> str:
        counts = {}
        for a in self.attempts:
            counts[a.status.value] = counts.get(a.status.value, 0) + 1
        return f"ProofSubsystem: {len(self.attempts)} attempts, status={counts}"


# ============= Layer 8: Visualization Engine =============

class VisualizationEngine:
    """Layer 8: Mathematical visualization — manifolds, networks, dynamics.

    Produces matplotlib figures. Reference style: 3D manifold gallery with
    rainbow/spectral colormaps on dark backgrounds.
    """

    def __init__(self):
        self.figures: List[Dict] = []

    def plot_manifold_3d(self, name: str = "torus", params: Dict = None,
                          save_path: str = None) -> Dict:
        """Generate 3D manifold surface plot."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            return {"error": "matplotlib not available"}

        params = params or {}
        fig = plt.figure(figsize=(10, 8), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, 2 * np.pi, 100)
        U, V = np.meshgrid(u, v)

        if name == "torus":
            R = params.get("R", 3); r = params.get("r", 1)
            X = (R + r * np.cos(V)) * np.cos(U)
            Y = (R + r * np.cos(V)) * np.sin(U)
            Z = r * np.sin(V)
        elif name == "sphere":
            r = params.get("radius", 2)
            X = r * np.sin(V) * np.cos(U)
            Y = r * np.sin(V) * np.sin(U)
            Z = r * np.cos(V)
        elif name == "klein_bottle":
            a = params.get("a", 2)
            X = (a + np.cos(U/2) * np.sin(V) - np.sin(U/2) * np.sin(2*V)) * np.cos(U)
            Y = (a + np.cos(U/2) * np.sin(V) - np.sin(U/2) * np.sin(2*V)) * np.sin(U)
            Z = np.sin(U/2) * np.sin(V) + np.cos(U/2) * np.sin(2*V)
        elif name == "mobius":
            t = np.linspace(0, 2*np.pi, 100)
            s = np.linspace(-0.5, 0.5, 30)
            T, S = np.meshgrid(t, s)
            X = (1 + S * np.cos(T/2)) * np.cos(T)
            Y = (1 + S * np.cos(T/2)) * np.sin(T)
            Z = S * np.sin(T/2)
        else:  # hyperboloid
            t = np.linspace(-2, 2, 100)
            th = np.linspace(0, 2*np.pi, 100)
            T, TH = np.meshgrid(t, th)
            X = np.cosh(T) * np.cos(TH)
            Y = np.cosh(T) * np.sin(TH)
            Z = np.sinh(T)

        norm = np.sqrt(X**2 + Y**2 + Z**2)
        colors = plt.cm.Spectral((norm - norm.min()) / (norm.max() - norm.min() + 1e-15))
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.9, shade=True)
        ax.set_title(f"{name.replace('_', ' ').title()}", color="white", fontsize=14)
        ax.set_axis_off()

        path = save_path or f"{name}_3d.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        record = {"type": "manifold_3d", "name": name, "path": path}
        self.figures.append(record)
        return record

    def plot_network(self, nodes: Dict[str, Dict], edges: List[Dict],
                      save_path: str = None) -> Dict:
        """Plot knowledge graph / theorem network."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"error": "matplotlib not available"}

        fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor="black")
        ax.set_facecolor("black")
        n = len(nodes)
        if n == 0:
            plt.close(fig)
            return {"error": "No nodes"}

        # Layout: circular
        names = list(nodes.keys())
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        pos = {name: (np.cos(a), np.sin(a)) for name, a in zip(names, angles)}

        # Domain colors
        domain_colors = {}
        palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
        for name, props in nodes.items():
            d = props.get("domain", "default")
            if d not in domain_colors:
                domain_colors[d] = palette[len(domain_colors) % len(palette)]

        # Draw edges
        for e in edges:
            s, t = e.get("source"), e.get("target")
            if s in pos and t in pos:
                ax.annotate("", xy=pos[t], xytext=pos[s],
                             arrowprops=dict(arrowstyle="->", color="#555555",
                                             lw=1.5, alpha=0.6))

        # Draw nodes
        for name, (x, y) in pos.items():
            d = nodes[name].get("domain", "default")
            color = domain_colors.get(d, "#FFFFFF")
            ax.scatter(x, y, s=300, c=color, zorder=5, edgecolors="white", linewidths=0.5)
            ax.text(x, y + 0.12, name, ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Mathematical Knowledge Network", color="white", fontsize=14)

        path = save_path or "knowledge_network.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        record = {"type": "network", "path": path, "nodes": n, "edges": len(edges)}
        self.figures.append(record)
        return record

    def plot_dynamics(self, trajectory: np.ndarray, name: str = "dynamics",
                       save_path: str = None) -> Dict:
        """Plot dynamical system trajectory (2D or 3D)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"error": "matplotlib not available"}

        fig = plt.figure(figsize=(10, 8), facecolor="black")
        if trajectory.shape[1] >= 3:
            ax = fig.add_subplot(111, projection="3d", facecolor="black")
            n = len(trajectory)
            colors = plt.cm.plasma(np.linspace(0, 1, n))
            for i in range(n - 1):
                ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                        trajectory[i:i+2, 2], color=colors[i], alpha=0.7, lw=0.5)
            ax.set_axis_off()
        else:
            ax = fig.add_subplot(111, facecolor="black")
            n = len(trajectory)
            colors = plt.cm.plasma(np.linspace(0, 1, n))
            ax.scatter(trajectory[:, 0], trajectory[:, 1],
                       c=colors, s=1, alpha=0.7)
            ax.set_axis_off()

        ax.set_title(name.replace("_", " ").title(), color="white", fontsize=14)
        path = save_path or f"{name}_dynamics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        record = {"type": "dynamics", "name": name, "path": path}
        self.figures.append(record)
        return record

    def plot_heatmap(self, data: np.ndarray, name: str = "heatmap",
                      save_path: str = None) -> Dict:
        """Plot 2D heatmap (e.g., PDE solution)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"error": "matplotlib not available"}

        fig, ax = plt.subplots(figsize=(8, 6), facecolor="black")
        ax.set_facecolor("black")
        im = ax.imshow(data, cmap="inferno", interpolation="bilinear")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(name.replace("_", " ").title(), color="white", fontsize=14)
        ax.tick_params(colors="white")

        path = save_path or f"{name}_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close(fig)
        record = {"type": "heatmap", "name": name, "path": path}
        self.figures.append(record)
        return record

    def summary(self) -> str:
        type_counts = {}
        for f in self.figures:
            type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
        return f"VisualizationEngine: {len(self.figures)} figures, types={type_counts}"

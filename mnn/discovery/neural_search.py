"""mnn.discovery.neural_search — Neural-Guided Theorem Search.

Engine 5: Combines symbolic search with neural guidance.
The neural network predicts which transformations/proof steps
are promising, replacing brute-force search with learned heuristics.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TheoremStep:
    """A single step in a proof/derivation."""
    rule_name: str
    input_state: Any
    output_state: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Step({self.rule_name}, conf={self.confidence:.3f})"


@dataclass
class ProofPath:
    """A sequence of theorem steps forming a proof attempt."""
    steps: List[TheoremStep] = field(default_factory=list)
    score: float = 0.0
    complete: bool = False

    def add_step(self, step: TheoremStep):
        self.steps.append(step)
        self.score += step.confidence

    @property
    def length(self):
        return len(self.steps)

    def __repr__(self):
        status = "COMPLETE" if self.complete else "PARTIAL"
        return f"ProofPath({status}, {self.length} steps, score={self.score:.3f})"


class TransformationRule:
    """A mathematical transformation rule that can be applied to expressions.

    Examples: commutativity (a+b -> b+a), distribution (a*(b+c) -> a*b + a*c),
    factoring, trigonometric identities, etc.
    """
    def __init__(self, name: str, pattern_fn: Callable,
                 transform_fn: Callable, priority: float = 0.5):
        self.name = name
        self.pattern_fn = pattern_fn    # returns True if applicable
        self.transform_fn = transform_fn  # applies the transformation
        self.priority = priority
        self.success_count = 0
        self.fail_count = 0

    def applicable(self, state: Any) -> bool:
        try:
            return self.pattern_fn(state)
        except Exception:
            return False

    def apply(self, state: Any) -> Any:
        return self.transform_fn(state)

    @property
    def empirical_success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return self.priority
        return self.success_count / total

    def __repr__(self):
        return f"Rule({self.name}, p={self.priority:.2f})"


class TransformationLibrary:
    """Library of standard mathematical transformation rules."""

    @staticmethod
    def algebraic_rules() -> List[TransformationRule]:
        import sympy as sp
        rules = []

        # Expand
        rules.append(TransformationRule(
            "expand", lambda e: isinstance(e, sp.Expr) and e.args,
            lambda e: sp.expand(e), priority=0.4,
        ))
        # Factor
        rules.append(TransformationRule(
            "factor", lambda e: isinstance(e, sp.Expr),
            lambda e: sp.factor(e), priority=0.5,
        ))
        # Simplify
        rules.append(TransformationRule(
            "simplify", lambda e: isinstance(e, sp.Expr),
            lambda e: sp.simplify(e), priority=0.6,
        ))
        # Trigonometric simplify
        rules.append(TransformationRule(
            "trig_simplify", lambda e: isinstance(e, sp.Expr),
            lambda e: sp.trigsimp(e), priority=0.3,
        ))
        # Collect
        rules.append(TransformationRule(
            "collect", lambda e: isinstance(e, sp.Expr) and e.free_symbols,
            lambda e: sp.collect(e, list(e.free_symbols)[0]) if e.free_symbols else e,
            priority=0.3,
        ))
        # Cancel
        rules.append(TransformationRule(
            "cancel", lambda e: isinstance(e, sp.Expr),
            lambda e: sp.cancel(e), priority=0.4,
        ))
        # Rationalize
        rules.append(TransformationRule(
            "radsimp", lambda e: isinstance(e, sp.Expr),
            lambda e: sp.radsimp(e), priority=0.2,
        ))
        return rules

    @staticmethod
    def calculus_rules() -> List[TransformationRule]:
        import sympy as sp
        rules = []

        # Differentiate
        rules.append(TransformationRule(
            "differentiate",
            lambda e: isinstance(e, sp.Expr) and e.free_symbols,
            lambda e: sp.diff(e, list(e.free_symbols)[0]) if e.free_symbols else e,
            priority=0.3,
        ))
        # Integrate
        rules.append(TransformationRule(
            "integrate",
            lambda e: isinstance(e, sp.Expr) and e.free_symbols,
            lambda e: sp.integrate(e, list(e.free_symbols)[0]) if e.free_symbols else e,
            priority=0.2,
        ))
        return rules


class ProofGuidanceNetwork(nn.Module):
    """Neural network that scores transformation rules for a given state.

    Input: encoded expression state
    Output: scores for each candidate rule
    """
    def __init__(self, state_dim: int, n_rules: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_rules),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec)

    def predict_best(self, state_vec: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            scores = self(torch.tensor(state_vec, dtype=torch.float32))
            return F.softmax(scores, dim=-1).numpy()


class NeuralTheoremSearcher:
    """Neural-guided search for proof paths.

    Uses beam search with neural scoring to find proof paths
    more efficiently than brute-force exploration.
    """
    def __init__(self, rules: List[TransformationRule],
                 state_encoder: Optional[Callable] = None,
                 beam_width: int = 5, max_depth: int = 10):
        self.rules = rules
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.state_encoder = state_encoder or self._default_encoder
        self.guidance = ProofGuidanceNetwork(64, len(rules))
        self.search_history: List[Dict] = []

    def _default_encoder(self, state: Any) -> np.ndarray:
        """Encode a sympy expression or string into a fixed-size vector."""
        s = str(state)
        # Simple hash-based encoding
        vec = np.zeros(64, dtype=np.float32)
        for i, ch in enumerate(s[:64]):
            vec[i] = ord(ch) / 128.0
        vec[min(len(s), 63)] = len(s) / 100.0
        return vec

    def search(self, start_state: Any, goal_fn: Callable,
               verbose: bool = False) -> Optional[ProofPath]:
        """Beam search for a proof path from start to goal.

        goal_fn: returns True if the state satisfies the target condition.
        """
        beam = [ProofPath()]
        beam[0].steps.append(TheoremStep("start", None, start_state, 1.0))

        for depth in range(self.max_depth):
            candidates = []

            for path in beam:
                current_state = path.steps[-1].output_state

                if goal_fn(current_state):
                    path.complete = True
                    if verbose:
                        print(f"  [Search] Goal reached at depth {depth}!")
                    return path

                # Score rules using neural guidance
                state_vec = self.state_encoder(current_state)
                scores = self.guidance.predict_best(state_vec)

                for i, rule in enumerate(self.rules):
                    if rule.applicable(current_state):
                        try:
                            new_state = rule.apply(current_state)
                            if str(new_state) != str(current_state):  # avoid no-ops
                                new_path = ProofPath(steps=list(path.steps))
                                step = TheoremStep(
                                    rule.name, current_state, new_state,
                                    confidence=float(scores[i]),
                                )
                                new_path.add_step(step)
                                candidates.append(new_path)
                        except Exception:
                            pass

            if not candidates:
                break

            # Keep top beam_width paths
            candidates.sort(key=lambda p: p.score, reverse=True)
            beam = candidates[:self.beam_width]

            if verbose:
                print(f"  [Search] Depth {depth+1}: {len(candidates)} candidates, "
                      f"best score={beam[0].score:.3f}")

        self.search_history.append({
            "start": str(start_state),
            "found": any(p.complete for p in beam),
            "paths_explored": sum(p.length for p in beam),
        })

        return beam[0] if beam else None

    def train_from_examples(self, examples: List[Tuple[Any, str]],
                             n_epochs: int = 200, lr: float = 1e-3):
        """Train guidance network from (state, correct_rule_name) pairs."""
        rule_names = [r.name for r in self.rules]
        optimizer = torch.optim.Adam(self.guidance.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(n_epochs):
            total_loss = 0
            for state, rule_name in examples:
                vec = torch.tensor(self.state_encoder(state), dtype=torch.float32).unsqueeze(0)
                target = torch.tensor([rule_names.index(rule_name)], dtype=torch.long)
                pred = self.guidance(vec)
                loss = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def summary(self) -> str:
        lines = [f"NeuralTheoremSearcher: {len(self.rules)} rules, "
                 f"beam={self.beam_width}, max_depth={self.max_depth}"]
        for r in self.rules:
            lines.append(f"  {r}")
        if self.search_history:
            n_found = sum(1 for h in self.search_history if h["found"])
            lines.append(f"  History: {len(self.search_history)} searches, "
                        f"{n_found} successful")
        return "\n".join(lines)

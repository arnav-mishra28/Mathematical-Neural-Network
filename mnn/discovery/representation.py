"""mnn.discovery.representation — Mathematical Representation Engine.

Engine 1: Makes mathematics machine-readable. Encodes equations as
expression trees, groups as operation tables, topologies as graphs,
tensors as arrays. Enables traversal, learning, and transformation.
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class NodeType(Enum):
    NUMBER = "number"
    SYMBOL = "symbol"
    OPERATOR = "operator"
    FUNCTION = "function"


@dataclass
class ExprNode:
    """Node in a mathematical expression tree.

    Represents: constants, variables, binary ops (+,-,*,/,**),
    unary functions (sin, cos, exp, log, sqrt, abs).
    """
    node_type: NodeType
    value: Any  # number, variable name, op symbol, or function name
    children: List["ExprNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def constant(cls, val: float) -> "ExprNode":
        return cls(NodeType.NUMBER, float(val))

    @classmethod
    def symbol(cls, name: str) -> "ExprNode":
        return cls(NodeType.SYMBOL, name)

    @classmethod
    def op(cls, operator: str, left: "ExprNode", right: "ExprNode") -> "ExprNode":
        return cls(NodeType.OPERATOR, operator, [left, right])

    @classmethod
    def func(cls, name: str, arg: "ExprNode") -> "ExprNode":
        return cls(NodeType.FUNCTION, name, [arg])

    def evaluate(self, env: Dict[str, float]) -> float:
        if self.node_type == NodeType.NUMBER:
            return self.value
        elif self.node_type == NodeType.SYMBOL:
            return env[self.value]
        elif self.node_type == NodeType.OPERATOR:
            l = self.children[0].evaluate(env)
            r = self.children[1].evaluate(env)
            ops = {"+": lambda a, b: a+b, "-": lambda a, b: a-b,
                   "*": lambda a, b: a*b, "/": lambda a, b: a/(b+1e-15),
                   "**": lambda a, b: a**b}
            return ops[self.value](l, r)
        elif self.node_type == NodeType.FUNCTION:
            a = self.children[0].evaluate(env)
            fns = {"sin": np.sin, "cos": np.cos, "exp": np.exp,
                   "log": lambda x: np.log(abs(x)+1e-15), "sqrt": lambda x: np.sqrt(abs(x)),
                   "abs": abs, "tan": np.tan, "tanh": np.tanh}
            return fns[self.value](a)
        return 0.0

    def to_sympy(self) -> sp.Expr:
        if self.node_type == NodeType.NUMBER:
            return sp.Float(self.value)
        elif self.node_type == NodeType.SYMBOL:
            return sp.Symbol(self.value)
        elif self.node_type == NodeType.OPERATOR:
            l = self.children[0].to_sympy()
            r = self.children[1].to_sympy()
            ops = {"+": sp.Add, "-": lambda a, b: a-b, "*": sp.Mul,
                   "/": lambda a, b: a/b, "**": sp.Pow}
            return ops[self.value](l, r)
        elif self.node_type == NodeType.FUNCTION:
            a = self.children[0].to_sympy()
            fns = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
                   "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
                   "tan": sp.tan, "tanh": sp.tanh}
            return fns[self.value](a)
        return sp.S.Zero

    @classmethod
    def from_sympy(cls, expr: sp.Expr) -> "ExprNode":
        if isinstance(expr, sp.Number):
            return cls.constant(float(expr))
        elif isinstance(expr, sp.Symbol):
            return cls.symbol(str(expr))
        elif isinstance(expr, sp.Add):
            args = list(expr.args)
            node = cls.from_sympy(args[0])
            for a in args[1:]:
                node = cls.op("+", node, cls.from_sympy(a))
            return node
        elif isinstance(expr, sp.Mul):
            args = list(expr.args)
            node = cls.from_sympy(args[0])
            for a in args[1:]:
                node = cls.op("*", node, cls.from_sympy(a))
            return node
        elif isinstance(expr, sp.Pow):
            base, exp = expr.args
            return cls.op("**", cls.from_sympy(base), cls.from_sympy(exp))
        elif isinstance(expr, sp.Function):
            fname = type(expr).__name__
            return cls.func(fname, cls.from_sympy(expr.args[0]))
        return cls.constant(0)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def variables(self) -> Set[str]:
        if self.node_type == NodeType.SYMBOL:
            return {self.value}
        result = set()
        for c in self.children:
            result |= c.variables()
        return result

    def to_string(self) -> str:
        if self.node_type == NodeType.NUMBER:
            v = self.value
            return str(int(v)) if v == int(v) else f"{v:.4g}"
        elif self.node_type == NodeType.SYMBOL:
            return self.value
        elif self.node_type == NodeType.OPERATOR:
            l = self.children[0].to_string()
            r = self.children[1].to_string()
            return f"({l} {self.value} {r})"
        elif self.node_type == NodeType.FUNCTION:
            return f"{self.value}({self.children[0].to_string()})"
        return "?"

    def __repr__(self):
        return self.to_string()

    def to_vector(self, max_depth: int = 8) -> np.ndarray:
        """Encode tree as fixed-size vector for neural processing."""
        vec = []
        self._encode_recursive(vec, max_depth, 0)
        return np.array(vec, dtype=np.float32)

    def _encode_recursive(self, vec: list, max_depth: int, current: int):
        type_enc = {"number": 0, "symbol": 1, "operator": 2, "function": 3}
        op_enc = {"+": 0.1, "-": 0.2, "*": 0.3, "/": 0.4, "**": 0.5}
        fn_enc = {"sin": 0.6, "cos": 0.7, "exp": 0.8, "log": 0.9, "sqrt": 1.0}

        if current >= max_depth:
            vec.extend([0, 0, 0])
            return
        vec.append(type_enc.get(self.node_type.value, 0))
        if self.node_type == NodeType.NUMBER:
            vec.append(float(np.tanh(self.value)))
            vec.append(0)
        elif self.node_type == NodeType.SYMBOL:
            vec.append(hash(self.value) % 100 / 100.0)
            vec.append(0)
        elif self.node_type == NodeType.OPERATOR:
            vec.append(op_enc.get(self.value, 0))
            vec.append(len(self.children))
        elif self.node_type == NodeType.FUNCTION:
            vec.append(fn_enc.get(self.value, 0))
            vec.append(1)

        for c in self.children:
            c._encode_recursive(vec, max_depth, current + 1)
        for _ in range(2 - len(self.children)):
            for _ in range(3 * (max_depth - current - 1) + 3):
                vec.append(0)


@dataclass
class MathObject:
    """Unified representation of a mathematical object for the discovery engine."""
    kind: str  # "equation", "sequence", "group", "tensor", "manifold", "graph"
    data: Any
    name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return hashlib.md5(f"{self.kind}:{self.name}:{str(self.data)[:200]}".encode()).hexdigest()[:16]


class MathEncoder:
    """Encode diverse mathematical objects into uniform representations."""

    @staticmethod
    def encode_sequence(seq: List[float]) -> MathObject:
        diffs = np.diff(seq)
        ratios = [seq[i+1]/(seq[i]+1e-15) for i in range(len(seq)-1)]
        return MathObject("sequence", np.array(seq), properties={
            "length": len(seq), "diffs": diffs.tolist(),
            "ratios": ratios, "mean": float(np.mean(seq)),
            "std": float(np.std(seq)),
        })

    @staticmethod
    def encode_function(expr: sp.Expr, variables: List[str]) -> MathObject:
        tree = ExprNode.from_sympy(expr)
        return MathObject("equation", tree, name=str(expr), properties={
            "variables": variables, "depth": tree.depth(),
            "size": tree.size(), "sympy_expr": expr,
        })

    @staticmethod
    def encode_group_table(table: np.ndarray, name: str = "G") -> MathObject:
        n = table.shape[0]
        is_abelian = np.allclose(table, table.T)
        has_identity = any(np.allclose(table[i], np.arange(n)) for i in range(n))
        return MathObject("group", table, name=name, properties={
            "order": n, "abelian": is_abelian, "has_identity": has_identity,
        })

    @staticmethod
    def encode_tensor(arr: np.ndarray, name: str = "T") -> MathObject:
        return MathObject("tensor", arr, name=name, properties={
            "shape": arr.shape, "rank": arr.ndim,
            "norm": float(np.linalg.norm(arr)),
            "symmetric": bool(arr.ndim == 2 and np.allclose(arr, arr.T)),
        })

    @staticmethod
    def encode_graph(adjacency: np.ndarray, name: str = "G") -> MathObject:
        n = adjacency.shape[0]
        degrees = adjacency.sum(axis=1)
        return MathObject("graph", adjacency, name=name, properties={
            "n_vertices": n, "n_edges": int(adjacency.sum()) // 2,
            "degrees": degrees.tolist(), "connected": bool(np.all(degrees > 0)),
        })

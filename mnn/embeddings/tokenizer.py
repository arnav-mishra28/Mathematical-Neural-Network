"""mnn.embeddings.tokenizer — Mathematical Tokenization Engine.

Part 2: Tokenize theorems into structured sequences that capture
operators, variables, algebraic types, graph structure, and
categorical roles — beyond simple text tokenization.
"""
from __future__ import annotations
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TokenType(Enum):
    VARIABLE = "variable"
    CONSTANT = "constant"
    OPERATOR = "operator"
    FUNCTION = "function"
    RELATION = "relation"
    QUANTIFIER = "quantifier"
    GROUPING = "grouping"
    STRUCTURE = "structure"
    KEYWORD = "keyword"


@dataclass
class MathToken:
    """A token from a mathematical expression with rich metadata."""
    text: str
    token_type: TokenType
    position: int = 0
    algebraic_type: str = ""      # e.g., "binary_op", "unary_fn", "element"
    categorical_role: str = ""    # e.g., "object", "morphism", "functor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_index(self, vocab: Dict[str, int]) -> int:
        return vocab.get(self.text, vocab.get("<UNK>", 0))

    def __repr__(self):
        return f"Token({self.text}, {self.token_type.value})"


class MathVocabulary:
    """Vocabulary of mathematical symbols with type annotations."""

    OPERATORS = {
        "+": ("binary_op", "morphism"), "-": ("binary_op", "morphism"),
        "*": ("binary_op", "morphism"), "/": ("binary_op", "morphism"),
        "**": ("binary_op", "morphism"), "^": ("binary_op", "morphism"),
        "@": ("binary_op", "composition"),
    }
    RELATIONS = {
        "=": ("equivalence", "isomorphism"), "!=": ("negation", "morphism"),
        "<": ("order", "morphism"), ">": ("order", "morphism"),
        "<=": ("order", "morphism"), ">=": ("order", "morphism"),
        "in": ("membership", "morphism"), "subset": ("inclusion", "morphism"),
        "~": ("equivalence", "isomorphism"), "->": ("mapping", "morphism"),
    }
    FUNCTIONS = {
        "sin": ("trig", "endomorphism"), "cos": ("trig", "endomorphism"),
        "tan": ("trig", "endomorphism"), "exp": ("exponential", "morphism"),
        "log": ("logarithmic", "morphism"), "sqrt": ("radical", "morphism"),
        "abs": ("norm", "functor"), "det": ("determinant", "functor"),
        "tr": ("trace", "functor"), "dim": ("dimension", "functor"),
    }
    QUANTIFIERS = {
        "forall": ("universal", "natural_transformation"),
        "exists": ("existential", "natural_transformation"),
    }
    STRUCTURES = {
        "group": ("algebraic", "category"), "ring": ("algebraic", "category"),
        "field": ("algebraic", "category"), "space": ("topological", "category"),
        "manifold": ("geometric", "category"), "functor": ("categorical", "functor"),
        "morphism": ("categorical", "morphism"),
    }

    def __init__(self):
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self._next_id = 4
        self._type_map: Dict[str, Tuple[str, str]] = {}
        self._build_vocab()

    def _build_vocab(self):
        for sym, (alg, cat) in self.OPERATORS.items():
            self._add(sym, alg, cat)
        for sym, (alg, cat) in self.RELATIONS.items():
            self._add(sym, alg, cat)
        for sym, (alg, cat) in self.FUNCTIONS.items():
            self._add(sym, alg, cat)
        for sym, (alg, cat) in self.QUANTIFIERS.items():
            self._add(sym, alg, cat)
        for sym, (alg, cat) in self.STRUCTURES.items():
            self._add(sym, alg, cat)
        # Common variables
        for v in list("abcdefghijklmnopqrstuvwxyz") + ["alpha", "beta", "gamma",
                 "delta", "epsilon", "theta", "phi", "psi", "omega"]:
            self._add(v, "element", "object")
        # Common constants
        for c in ["0", "1", "2", "pi", "e", "i", "inf"]:
            self._add(c, "constant", "object")

    def _add(self, sym: str, alg_type: str, cat_role: str):
        if sym not in self.vocab:
            self.vocab[sym] = self._next_id
            self._next_id += 1
        self._type_map[sym] = (alg_type, cat_role)

    def get_types(self, sym: str) -> Tuple[str, str]:
        return self._type_map.get(sym, ("unknown", "unknown"))

    @property
    def size(self) -> int:
        return len(self.vocab)


class MathTokenizer:
    """Tokenize mathematical expressions with type annotations."""

    def __init__(self, vocab: Optional[MathVocabulary] = None):
        self.vocab = vocab or MathVocabulary()

    def tokenize(self, expression: str) -> List[MathToken]:
        """Tokenize a mathematical expression string."""
        # Normalize
        expr = expression.strip()
        # Split into tokens
        raw_tokens = self._split(expr)
        tokens = []
        for i, raw in enumerate(raw_tokens):
            token_type = self._classify(raw)
            alg_type, cat_role = self.vocab.get_types(raw)
            tokens.append(MathToken(
                text=raw, token_type=token_type, position=i,
                algebraic_type=alg_type, categorical_role=cat_role,
            ))
        return tokens

    def _split(self, expr: str) -> List[str]:
        """Split expression preserving multi-char tokens."""
        # Replace multi-char operators/relations
        expr = expr.replace("!=", " != ").replace("<=", " <= ").replace(">=", " >= ")
        expr = expr.replace("->", " -> ").replace("**", " ** ")
        expr = expr.replace("forall", " forall ").replace("exists", " exists ")
        # Split on whitespace and common delimiters
        pattern = r'(\w+|[+\-*/^=<>!~@(),\[\]{}|]|\*\*|!=|<=|>=|->)'
        tokens = re.findall(pattern, expr)
        return [t.strip() for t in tokens if t.strip()]

    def _classify(self, token: str) -> TokenType:
        if token in self.vocab.OPERATORS or token in {"^"}:
            return TokenType.OPERATOR
        elif token in self.vocab.RELATIONS:
            return TokenType.RELATION
        elif token in self.vocab.FUNCTIONS:
            return TokenType.FUNCTION
        elif token in self.vocab.QUANTIFIERS:
            return TokenType.QUANTIFIER
        elif token in self.vocab.STRUCTURES:
            return TokenType.STRUCTURE
        elif token in {"(", ")", "[", "]", "{", "}", "|"}:
            return TokenType.GROUPING
        elif re.match(r'^-?\d+\.?\d*$', token):
            return TokenType.CONSTANT
        elif token in {"pi", "e", "i", "inf"}:
            return TokenType.CONSTANT
        else:
            return TokenType.VARIABLE

    def encode(self, expression: str, max_len: int = 64) -> np.ndarray:
        """Encode expression as integer sequence."""
        tokens = self.tokenize(expression)
        indices = [self.vocab.vocab.get("<BOS>", 2)]
        for t in tokens[:max_len - 2]:
            indices.append(t.to_index(self.vocab.vocab))
        indices.append(self.vocab.vocab.get("<EOS>", 3))
        # Pad
        while len(indices) < max_len:
            indices.append(0)
        return np.array(indices[:max_len], dtype=np.int64)

    def batch_encode(self, expressions: List[str], max_len: int = 64) -> np.ndarray:
        return np.stack([self.encode(e, max_len) for e in expressions])

    def decode_tokens(self, tokens: List[MathToken]) -> str:
        return " ".join(t.text for t in tokens)

    def structural_encoding(self, expression: str) -> Dict:
        """Rich structural encoding beyond sequence."""
        tokens = self.tokenize(expression)
        return {
            "tokens": tokens,
            "n_tokens": len(tokens),
            "n_variables": sum(1 for t in tokens if t.token_type == TokenType.VARIABLE),
            "n_operators": sum(1 for t in tokens if t.token_type == TokenType.OPERATOR),
            "n_functions": sum(1 for t in tokens if t.token_type == TokenType.FUNCTION),
            "has_quantifiers": any(t.token_type == TokenType.QUANTIFIER for t in tokens),
            "has_relations": any(t.token_type == TokenType.RELATION for t in tokens),
            "algebraic_types": list(set(t.algebraic_type for t in tokens)),
            "categorical_roles": list(set(t.categorical_role for t in tokens)),
            "depth_estimate": expression.count("("),
        }

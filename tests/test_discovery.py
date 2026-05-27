"""Tests for the Mathematical Discovery Engine (mnn.discovery)."""
import numpy as np
import sympy as sp
import pytest


# ===== Engine 1: Representation Tests =====

class TestExprNode:
    def test_constant(self):
        from mnn.discovery.representation import ExprNode
        c = ExprNode.constant(3.14)
        assert abs(c.evaluate({}) - 3.14) < 1e-10

    def test_symbol(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        assert abs(x.evaluate({"x": 5.0}) - 5.0) < 1e-10

    def test_operator(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        y = ExprNode.symbol("y")
        expr = ExprNode.op("+", x, y)
        assert abs(expr.evaluate({"x": 3, "y": 4}) - 7) < 1e-10

    def test_function(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        expr = ExprNode.func("sin", x)
        assert abs(expr.evaluate({"x": np.pi/2}) - 1.0) < 1e-10

    def test_depth_and_size(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        sq = ExprNode.op("**", x, ExprNode.constant(2))
        assert sq.depth() == 1
        assert sq.size() == 3

    def test_sympy_roundtrip(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        expr = ExprNode.op("+", ExprNode.op("**", x, ExprNode.constant(2)),
                           ExprNode.constant(1))
        sp_expr = expr.to_sympy()
        back = ExprNode.from_sympy(sp_expr)
        assert abs(back.evaluate({"x": 3}) - 10) < 1e-6

    def test_variables(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        y = ExprNode.symbol("y")
        expr = ExprNode.op("*", x, y)
        assert expr.variables() == {"x", "y"}

    def test_to_vector(self):
        from mnn.discovery.representation import ExprNode
        x = ExprNode.symbol("x")
        vec = x.to_vector(max_depth=4)
        assert len(vec) > 0


class TestMathEncoder:
    def test_encode_sequence(self):
        from mnn.discovery.representation import MathEncoder
        obj = MathEncoder.encode_sequence([1, 4, 9, 16])
        assert obj.kind == "sequence"
        assert obj.properties["length"] == 4

    def test_encode_tensor(self):
        from mnn.discovery.representation import MathEncoder
        obj = MathEncoder.encode_tensor(np.eye(3), "I")
        assert obj.properties["symmetric"]
        assert obj.properties["rank"] == 2

    def test_encode_graph(self):
        from mnn.discovery.representation import MathEncoder
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        obj = MathEncoder.encode_graph(adj)
        assert obj.properties["n_vertices"] == 3


# ===== Engine 2: Pattern Discovery Tests =====

class TestSymmetryDetector:
    def test_commutativity(self):
        from mnn.discovery.patterns import SymmetryDetector
        floats = list(range(-5, 6))
        r = SymmetryDetector.test_commutativity(lambda a, b: a+b, floats)
        assert r["commutative"]

    def test_non_commutativity(self):
        from mnn.discovery.patterns import SymmetryDetector
        floats = list(range(-5, 6))
        r = SymmetryDetector.test_commutativity(lambda a, b: a-b, floats)
        assert not r["commutative"]

    def test_associativity(self):
        from mnn.discovery.patterns import SymmetryDetector
        floats = list(range(-3, 4))
        r = SymmetryDetector.test_associativity(lambda a, b: a*b, floats)
        assert r["associative"]

    def test_even_function(self):
        from mnn.discovery.patterns import SymmetryDetector
        sym = SymmetryDetector.detect_function_symmetry(
            lambda x: x[0]**2, dim=1)
        assert sym.get("even", False)

    def test_odd_function(self):
        from mnn.discovery.patterns import SymmetryDetector
        sym = SymmetryDetector.detect_function_symmetry(
            lambda x: x[0]**3, dim=1)
        assert sym.get("odd", False)


class TestSequenceAnalyzer:
    def test_arithmetic(self):
        from mnn.discovery.patterns import SequenceAnalyzer
        info = SequenceAnalyzer.analyze([3, 7, 11, 15, 19])
        assert info["pattern"] == "arithmetic"

    def test_geometric(self):
        from mnn.discovery.patterns import SequenceAnalyzer
        info = SequenceAnalyzer.analyze([2, 6, 18, 54])
        assert info["pattern"] == "geometric"

    def test_quadratic(self):
        from mnn.discovery.patterns import SequenceAnalyzer
        info = SequenceAnalyzer.analyze([1, 4, 9, 16, 25])
        assert info["pattern"] == "quadratic"

    def test_predict_next(self):
        from mnn.discovery.patterns import SequenceAnalyzer
        pred = SequenceAnalyzer.predict_next([2, 4, 6, 8], 2)
        assert abs(pred[0] - 10) < 1e-6
        assert abs(pred[1] - 12) < 1e-6


class TestInvariantFinder:
    def test_conservation_harmonic(self):
        from mnn.discovery.patterns import InvariantFinder
        t = np.linspace(0, 20, 500)
        traj = np.column_stack([np.cos(t), -np.sin(t)])
        conserved = InvariantFinder.find_conservation_laws(traj)
        # ||x||^2 = cos^2 + sin^2 = 1 should be conserved
        names = [c["name"] for c in conserved]
        assert any("||x||" in n for n in names)


# ===== Engine 3: Conjecture Tests =====

class TestConjectureGenerator:
    def test_generate_algebraic(self):
        from mnn.discovery.conjectures import ConjectureGenerator
        gen = ConjectureGenerator()
        floats = list(range(-5, 6))
        conj = gen.analyze_operation(lambda a, b: a+b, floats, "add")
        assert len(conj) >= 2  # commutative + associative at minimum

    def test_generate_sequence(self):
        from mnn.discovery.conjectures import ConjectureGenerator
        gen = ConjectureGenerator()
        conj = gen.analyze_sequence([1, 4, 9, 16, 25])
        assert len(conj) >= 1
        assert "quadratic" in conj[0].statement.lower() or "pattern" in conj[0].statement.lower()

    def test_generate_function_symmetry(self):
        from mnn.discovery.conjectures import ConjectureGenerator
        gen = ConjectureGenerator()
        conj = gen.analyze_function_symmetry(
            lambda x: np.cos(x[0]), dim=1, name="cos")
        types = [c.statement for c in conj]
        assert any("even" in t for t in types)


# ===== Engine 4: Validation Tests =====

class TestNumericalValidator:
    def test_validate_equality(self):
        from mnn.discovery.validation import NumericalValidator
        domain = np.random.randn(100, 1)
        r = NumericalValidator.validate_equality(
            lambda x: x[0]**2, lambda x: x[0]**2, domain)
        assert r["valid"]

    def test_validate_inequality(self):
        from mnn.discovery.validation import NumericalValidator
        domain = np.random.randn(100, 1)
        r = NumericalValidator.validate_inequality(
            lambda x: x[0]**2, lambda x: x[0]**2 + 1, domain)
        assert r["valid"]


class TestCounterexampleSearcher:
    def test_no_counterexample(self):
        from mnn.discovery.validation import CounterexampleSearcher
        r = CounterexampleSearcher.random_search(
            lambda x: x[0]**2 >= 0, dim=1, n_samples=1000)
        assert not r["found"]

    def test_finds_counterexample(self):
        from mnn.discovery.validation import CounterexampleSearcher
        r = CounterexampleSearcher.random_search(
            lambda x: x[0] > 0, dim=1, n_samples=1000)
        assert r["found"]


class TestSymbolicValidator:
    def test_identity(self):
        from mnn.discovery.validation import SymbolicValidator
        a, b = sp.symbols("a b")
        r = SymbolicValidator.verify_identity(a + b, b + a, ["a", "b"])
        assert r["proven"]

    def test_trig_identity(self):
        from mnn.discovery.validation import SymbolicValidator
        x = sp.Symbol("x")
        r = SymbolicValidator.verify_identity(
            sp.sin(x)**2 + sp.cos(x)**2, sp.Integer(1), ["x"])
        assert r["proven"]


# ===== Engine 5: Neural Search Tests =====

class TestNeuralSearch:
    def test_transformation_library(self):
        from mnn.discovery.neural_search import TransformationLibrary
        rules = TransformationLibrary.algebraic_rules()
        assert len(rules) >= 5

    def test_simplification_search(self):
        from mnn.discovery.neural_search import NeuralTheoremSearcher, TransformationLibrary
        rules = TransformationLibrary.algebraic_rules()
        searcher = NeuralTheoremSearcher(rules, beam_width=3, max_depth=5)
        x = sp.Symbol("x")
        start = (x + 1)**2 - x**2 - 2*x
        path = searcher.search(start, lambda e: sp.simplify(e - 1) == 0)
        assert path is not None


# ===== Engine 6: Categorical Discovery Tests =====

class TestCategoricalDiscovery:
    def test_morphism_invariants(self):
        from mnn.discovery.categorical import MorphismInvariantFinder
        f = lambda x: 2 * x  # scaling preserves sign
        props = {
            "sign": lambda x: float(np.sign(x[0])),
            "norm": lambda x: float(np.linalg.norm(x)),
        }
        data = [np.random.randn(2) for _ in range(100)]
        preserved = MorphismInvariantFinder.find_preserved_properties(
            f, props, data)
        names = [p["property"] for p in preserved]
        assert "sign" in names  # scaling preserves sign

    def test_equivalence_classes(self):
        from mnn.discovery.categorical import EquivalenceDiscovery
        elements = list(range(10))
        # Equivalence: same parity
        classes = EquivalenceDiscovery.find_equivalence_classes(
            elements, lambda a, b: a % 2 == b % 2)
        assert len(classes) == 2  # even and odd

    def test_orbit(self):
        from mnn.discovery.categorical import EquivalenceDiscovery
        orbit = EquivalenceDiscovery.find_orbit(
            0, [lambda x: (x + 1) % 5])
        assert len(orbit) == 5  # Z_5

    def test_quotient(self):
        from mnn.discovery.categorical import EquivalenceDiscovery
        elements = list(range(12))
        result = EquivalenceDiscovery.quotient_structure(
            elements, lambda a, b: a % 3 == b % 3,
            op=lambda a, b: (a + b) % 12)
        assert result["n_classes"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

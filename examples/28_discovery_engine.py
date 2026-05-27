"""
Example 28 — Mathematical Discovery Engine: Full Pipeline
Demonstrates all 6 engines working together: representation, pattern discovery,
conjecture generation, validation, neural search, and categorical discovery.
"""
import numpy as np
import sympy as sp


def main():
    print("=" * 70)
    print("  MATHEMATICAL DISCOVERY ENGINE — Full Pipeline")
    print("=" * 70)

    # ---- Engine 1: Mathematical Representation ----
    print("\n[Engine 1] Mathematical Representation")
    from mnn.discovery.representation import ExprNode, MathEncoder

    # Build expression tree for x^2 + sin(y)
    x = ExprNode.symbol("x")
    y = ExprNode.symbol("y")
    x_sq = ExprNode.op("**", x, ExprNode.constant(2))
    sin_y = ExprNode.func("sin", y)
    expr = ExprNode.op("+", x_sq, sin_y)
    print(f"  Expression tree: {expr}")
    print(f"  Depth: {expr.depth()}, Size: {expr.size()}")
    print(f"  Variables: {expr.variables()}")
    print(f"  Evaluate at x=3, y=pi/2: {expr.evaluate({'x': 3, 'y': np.pi/2}):.4f}")

    # Roundtrip through sympy
    sp_expr = expr.to_sympy()
    print(f"  SymPy: {sp_expr}")
    back = ExprNode.from_sympy(sp_expr)
    print(f"  Roundtrip: {back}")

    # Encode a sequence
    seq_obj = MathEncoder.encode_sequence([1, 4, 9, 16, 25, 36])
    print(f"  Sequence props: {seq_obj.properties}")

    # ---- Engine 2: Pattern Discovery ----
    print("\n[Engine 2] Pattern Discovery")
    from mnn.discovery.patterns import SymmetryDetector, InvariantFinder, SequenceAnalyzer

    # Sequence pattern discovery
    print("  Sequence analysis:")
    seqs = {
        "squares": [1, 4, 9, 16, 25],
        "fibonacci": [1, 1, 2, 3, 5, 8, 13],
        "powers_of_2": [1, 2, 4, 8, 16, 32],
        "arithmetic": [3, 7, 11, 15, 19],
    }
    for name, seq in seqs.items():
        info = SequenceAnalyzer.analyze(seq)
        pred = SequenceAnalyzer.predict_next(seq, 2)
        print(f"    {name}: {info['pattern']}, next = {pred}")

    # Function symmetry
    print("\n  Function symmetry detection:")
    sym_cos = SymmetryDetector.detect_function_symmetry(
        lambda x: np.cos(x[0]), dim=1)
    print(f"    cos(x): {sym_cos}")
    sym_sin = SymmetryDetector.detect_function_symmetry(
        lambda x: np.sin(x[0]), dim=1)
    print(f"    sin(x): {sym_sin}")

    # Operation symmetry
    print("\n  Operation symmetry:")
    floats = [float(i) for i in range(-5, 6)]
    comm = SymmetryDetector.test_commutativity(lambda a, b: a + b, floats)
    print(f"    Addition commutative: {comm['commutative']}")
    assoc = SymmetryDetector.test_associativity(lambda a, b: a + b, floats)
    print(f"    Addition associative: {assoc['associative']}")

    # Conservation laws in a harmonic oscillator trajectory
    print("\n  Conservation law detection (harmonic oscillator):")
    t = np.linspace(0, 20, 500)
    trajectory = np.column_stack([np.cos(t), -np.sin(t)])  # x, v
    conserved = InvariantFinder.find_conservation_laws(trajectory)
    for c in conserved:
        print(f"    Conserved: {c['name']} = {c['mean_value']:.6f} "
              f"(var: {c['relative_variation']:.2e})")

    # ---- Engine 3: Conjecture Generation ----
    print("\n[Engine 3] Conjecture Generation")
    from mnn.discovery.conjectures import ConjectureGenerator

    gen = ConjectureGenerator()

    # Analyze addition
    conj = gen.analyze_operation(lambda a, b: a + b, floats, "addition")
    for c in conj:
        print(f"  {c}")

    # Analyze sequence
    conj2 = gen.analyze_sequence([1, 4, 9, 16, 25, 36], "perfect_squares")
    for c in conj2:
        print(f"  {c}")

    # Analyze function symmetry
    conj3 = gen.analyze_function_symmetry(
        lambda x: np.cos(x[0]), dim=1, name="cos")
    for c in conj3:
        print(f"  {c}")

    # Analyze dynamical system
    conj4 = gen.analyze_dynamical_system(trajectory, "harmonic_oscillator")
    for c in conj4:
        print(f"  {c}")

    # Analyze invariance under rotation
    data_2d = np.random.randn(100, 2)
    rotation = lambda x: np.array([x[0]*np.cos(0.3) - x[1]*np.sin(0.3),
                                    x[0]*np.sin(0.3) + x[1]*np.cos(0.3)])
    conj5 = gen.analyze_transformation_invariance(
        lambda x: np.linalg.norm(x),
        {"rotation": rotation},
        data_2d, name="||x||"
    )
    for c in conj5:
        print(f"  {c}")

    # ---- Engine 4: Validation ----
    print("\n[Engine 4] Proof / Validation")
    from mnn.discovery.validation import ConjectureValidator, SymbolicValidator

    validator = ConjectureValidator()

    # Validate commutativity conjecture
    comm_conj = conj[0]  # addition is commutative
    validated = validator.validate_conjecture(
        comm_conj,
        test_fn=lambda x: abs((x[0]+x[1]) - (x[1]+x[0])) < 1e-8,
        domain=np.random.randn(500, 2), dim=2,
    )
    print(f"  Validated: {validated}")

    # Symbolic verification
    a, b = sp.symbols("a b")
    result = SymbolicValidator.verify_identity(a + b, b + a, ["a", "b"])
    print(f"  Symbolic: a+b = b+a? {result}")

    # Verify sin^2 + cos^2 = 1
    x_sym = sp.Symbol("x")
    result2 = SymbolicValidator.verify_identity(
        sp.sin(x_sym)**2 + sp.cos(x_sym)**2, sp.Integer(1), ["x"])
    print(f"  Symbolic: sin^2+cos^2 = 1? {result2}")

    # ---- Engine 5: Neural-Guided Theorem Search ----
    print("\n[Engine 5] Neural-Guided Theorem Search")
    from mnn.discovery.neural_search import (
        NeuralTheoremSearcher, TransformationLibrary,
    )

    rules = TransformationLibrary.algebraic_rules()
    searcher = NeuralTheoremSearcher(rules, beam_width=3, max_depth=5)
    print(f"  {searcher.summary()}")

    # Search: simplify (x+1)^2 - x^2 - 2*x to 1
    x_sym = sp.Symbol("x")
    start = (x_sym + 1)**2 - x_sym**2 - 2*x_sym
    goal = lambda e: sp.simplify(e - 1) == 0

    path = searcher.search(start, goal, verbose=True)
    if path:
        print(f"  Result: {path}")
        for step in path.steps:
            print(f"    {step.rule_name}: {step.output_state}")

    # ---- Engine 6: Category-Theoretic Discovery ----
    print("\n[Engine 6] Category-Theoretic Discovery")
    from mnn.discovery.categorical import CategoricalDiscoveryEngine

    cat_engine = CategoricalDiscoveryEngine()

    # Analyze a linear transformation
    A = np.array([[1, 0], [0, 2]])
    linear_map = lambda x: A @ x
    properties = {
        "norm": lambda x: float(np.linalg.norm(x)),
        "sign_x0": lambda x: float(np.sign(x[0])),
        "direction": lambda x: float(np.arctan2(x[1], x[0])),
        "quadrant": lambda x: int(np.sign(x[0]) + np.sign(x[1])),
    }
    test_vecs = [np.random.randn(2) for _ in range(200)]

    conj6 = cat_engine.analyze_morphism(linear_map, properties, test_vecs, "Ax")
    for c in conj6:
        print(f"  {c}")

    # Discover equivalence classes
    elements = [np.array([i, j]) for i in range(-3, 4) for j in range(-3, 4)]
    conj7 = cat_engine.analyze_equivalences(
        elements,
        lambda a, b: np.linalg.norm(a) == np.linalg.norm(b),
        "same_norm"
    )
    for c in conj7:
        print(f"  {c}")

    # ---- Full Report ----
    print("\n" + "=" * 70)
    print("  FULL CONJECTURE REPORT")
    print("=" * 70)
    print(gen.report())
    print("\n  Category-theoretic:")
    print(cat_engine.report())

    print("\n" + "=" * 70)
    print("  MATHEMATICAL DISCOVERY ENGINE — All 6 engines verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()

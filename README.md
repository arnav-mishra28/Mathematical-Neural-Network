# MNN — Mathematical Neural Network Framework

> Research-grade framework fusing neural networks with symbolic mathematics,
> differential geometry, tensor calculus, group theory, and chaos theory.

## Architecture
```
mnn/
├── core/          → Math Engine (tensors, operators, symbolic functions)
├── neural/        → Learning Engine (constraint-aware neural networks)
├── geometry/      → Geometry & Topology (manifolds, hyperspheres, hyperplanes)
├── algebra/       → Algebraic Engine (Group Theory, Abelian Theory, tensor calculus)
├── chaos/         → Chaos Engine (attractors, dynamics, Lyapunov analysis)
├── advanced/      → Deep group theory, elliptic curves, fractals
├── intelligence/  → Advanced Mathematical Intelligence Layer (Phase 3)
│   ├── dynamical.py     → Flow map learning, stability, bifurcations
│   ├── group_algebra.py → Neural group ops, equivariant nets, invariants
│   ├── neural_pde.py    → Generalized PINN for arbitrary PDEs
│   └── discovery.py     → Equation discovery (SINDy + neural smoothing)
├── category/      → Category Theory Engine (Phase 4 — Unifying Abstraction)
│   ├── core.py          → Objects, morphisms, categories, composition
│   ├── functors.py      → Functors, natural transformations, bridges
│   └── neural.py        → Neural morphisms, learnable functors, pipelines
├── quantum/       → Quantum-Inspired Neural Math (Phase 5)
│   ├── hilbert.py       → Hilbert spaces, quantum states, gates
│   ├── complex_nn.py    → Complex-valued neural networks
│   ├── unitary.py       → Unitary transformations, norm-preserving layers
│   ├── attention.py     → Quantum-inspired attention mechanisms
│   ├── geometric.py     → Fubini-Study manifold, curvature-aware learning
│   └── chaos.py         → Random matrices, spectral statistics, OTOC
├── discovery/     → Mathematical Discovery Engine (Phase 6)
│   ├── representation.py → Expression trees, math object encoding
│   ├── patterns.py       → Symmetry detection, invariant finding
│   ├── conjectures.py    → Hypothesis generation
│   ├── validation.py     → Numerical/symbolic verification
│   ├── neural_search.py  → Neural-guided theorem search
│   └── categorical.py    → Category-theoretic discovery
├── embeddings/    → Quantum-Inspired Theorem Embeddings (Phase 7)
│   ├── theorem_states.py → Quantum state representation of theorems
│   ├── tokenizer.py      → Mathematical tokenization with type annotations
│   ├── complex_embed.py  → Complex-valued embeddings + similarity
│   ├── proof_trajectories.py → Proof paths through theorem space
│   ├── categorical_embed.py  → Category-theoretic embeddings
│   └── training.py       → Training objectives (4 losses)
├── quantum_pde/   → Quantum PDE Solvers (Phase 8)
│   ├── states.py         → Quantum state PDE representation + operators
│   ├── neural_operator.py → Neural operators + quantum evolution
│   ├── spectral.py       → Spectral solving + geometric regularization
│   ├── pde_discovery.py  → Discover governing PDEs from data
│   └── categorical_pde.py → Category-theoretic PDE structure
└── visualization/ → Visualization Engine
```

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start
```python
from mnn.core.math_engine    import MathEngine
from mnn.geometry.manifolds  import RiemannianManifold
from mnn.chaos.attractors    import LorenzAttractor
from mnn.algebra.groups      import Group, LieGroup
from mnn.algebra.abelian     import AbelianFunction
from mnn.algebra.tensors     import TensorEngine

eng = MathEngine()
f   = eng.define_function("sin(x)*exp(-x**2)", ["x"])
print(eng.differentiate(f, "x"))

S2  = RiemannianManifold.sphere_S2()
print(S2.ricci_scalar())          # → 2

lor  = LorenzAttractor()
traj = lor.integrate(t_span=(0,50), dt=0.01)

so3  = LieGroup.SO(3)
print(so3.is_semisimple())        # → True
```

## Examples
```bash
python examples/01_basic_math_engine.py
python examples/02_neural_constraints.py
python examples/03_geometry_topology.py
python examples/04_tensors_vectors.py
python examples/05_group_theory.py
python examples/06_chaos_theory.py
python examples/07_full_pipeline.py
```

## Tests
```bash
cd tests && python run_all_tests.py
```

---

## Phase 2 — Advanced Modules

### Advanced Group Theory (`mnn.advanced.group_theory`)
```python
from mnn.advanced.group_theory import FiniteGroupAnalyzer, RepresentationTheory, GroupHomomorphism, GroupExtension

# Deep structural analysis
G   = Group.symmetric(4)
ana = FiniteGroupAnalyzer(G)
print(ana.structural_report())
# → solvable, derived series, Sylow theorems, composition series

# Representation theory
RT   = RepresentationTheory(Group.cyclic(6))
tab, _ = RT.character_table_abelian()
print(RT.verify_orthogonality())   # True

# Homomorphisms
hom = GroupHomomorphism(Z4, Z2, lambda x: x%2)
print(hom.first_isomorphism_theorem())

# Semi-direct product
D3 = GroupExtension.semidirect_product(Z3, Z2, action)
```

### Advanced Abelian Theory (`mnn.advanced.abelian`)
```python
from mnn.advanced.abelian import EllipticCurve, JacobiVariety, AbelianVariety, ThetaDivisor

# Elliptic curves with full group law
E  = EllipticCurve(a=-1, b=0)
P  = Point(1., 0.)
print(E.scalar_mult(5, P))          # 5P
print(E.points_over_Fp(7))          # E(F₇)
print(E.j_invariant())

# Abelian varieties
A  = AbelianVariety(np.array([[1.1j, 0.3j],[0.3j, 1.4j]]))
print(A.theta(z))                   # Riemann theta
TD = ThetaDivisor(A)
print(TD.riemann_roch())
```

### Advanced Chaos (`mnn.advanced.chaos_advanced`)
```python
from mnn.advanced.chaos_advanced import FractalAnalyzer, ChaoticMap, CoupledOscillators

# Fractal dimensions
d, _, _ = FractalAnalyzer.box_counting_dimension(points)
H       = FractalAnalyzer.hurst_exponent_dfa(signal)
M       = FractalAnalyzer.mandelbrot_set()
sier    = FractalAnalyzer.sierpinski_triangle()

# Chaotic maps
pts  = ChaoticMap.standard_map(K=1.5)
ik   = ChaoticMap.ikeda_map()
cat  = ChaoticMap.arnolds_cat_map()

# Coupled oscillators
t, theta = CoupledOscillators.kuramoto_model(N=10, K=3.)
r        = CoupledOscillators.order_parameter(theta)  # synchronization
```

### Prototype — Derivative-Constrained Network (`mnn.advanced.prototype`)
```python
from mnn.advanced.prototype import run_prototype

# Train f(x) such that df/dx = 2x → learns f(x) = x²
result = run_prototype(n_epochs=3000, width=64, depth=5)
print(result.summary())
# MSE(f vs x²)  ≈ tiny
# MSE(f' vs 2x) ≈ tiny
```

---

## Phase 3 — Advanced Mathematical Intelligence Layer

### Pillar 1: Dynamical + Nonlinear Systems (`mnn.intelligence.dynamical`)

Learns **flow maps** `x(t+Δt) = F(x(t))` instead of derivatives — more stable and captures long-term dynamics better.

```python
from mnn.intelligence.dynamical import FlowMapLearner, StabilityAnalyzer, BifurcationDetector

# Learn the Lorenz flow map
x_now, x_next = FlowMapLearner.generate_training_data(lorenz.ode, x0, (0,30), dt=0.01)
learner = FlowMapLearner(state_dim=3, width=128, depth=4, dt=0.01)
learner.train(x_now, x_next, n_epochs=2000)
trajectory = learner.predict(x0, n_steps=5000)

# Analyze stability
results = StabilityAnalyzer.analyze_system(lorenz.ode, dim=3)
# → fixed points, eigenvalues, saddle/stable/unstable classification

# Detect bifurcations (Hopf, saddle-node)
hopf = BifurcationDetector.detect_hopf(rhs_factory, (0,30), dim=3, fp_guess=origin)
```

### Pillar 2: Abstract Algebra Engine (`mnn.intelligence.group_algebra`)

Neural networks that **respect algebraic structure** — group axioms enforced as differentiable constraints.

```python
from mnn.intelligence.group_algebra import NeuralGroupOperator, EquivariantNetwork, InvariantLearner

# Learn a group operation with constraint enforcement
op = NeuralGroupOperator(element_dim=8, abelian=True)
# Constraints: identity, inverse, associativity, commutativity

# Build rotation-equivariant network: f(g·x) = g·f(x)
eq_net = EquivariantNetwork(2, 2, rotation_group_actions)

# Learn invariant quantities: I(g·x) = I(x) for all g
inv = InvariantLearner(2, 1, group_actions=rotations)
inv.train(data, targets=norms)
```

### Pillar 3: Neural PDE Solvers (`mnn.intelligence.neural_pde`)

Generalized **Physics-Informed Neural Network** for arbitrary PDEs — extends beyond physics into all mathematics.

```python
from mnn.intelligence.neural_pde import NeuralPDESolver, heat_1d, poisson_2d, wave_1d

# Solve ∇²u = f(x,y) with zero Dirichlet BC
solver = NeuralPDESolver(poisson_2d(source_fn), width=128, depth=5)
solver.train(n_epochs=5000, n_collocation=2000)
u = solver.predict(grid_points)

# Pre-built problems: heat_1d, wave_1d, burgers_1d, poisson_2d
```

### Pillar 4: Scientific Discovery Engine (`mnn.intelligence.discovery`)

Automatically **discover governing equations from data** — hybrid neural-symbolic regression.

```python
from mnn.intelligence.discovery import HybridDiscovery, ScientificDiscoveryEngine

# From Lorenz trajectory → discover dx/dt = 10(y-x), dy/dt = x(28-z)-y, ...
discovery = HybridDiscovery(state_dim=3, poly_order=2, threshold=0.5)
result = discovery.discover(t, trajectory, var_names=["dx/dt","dy/dt","dz/dt"])
# → prints discovered symbolic equations with R² score

# Automated multi-threshold sweep with complexity-accuracy tradeoff
engine = ScientificDiscoveryEngine(state_dim=3)
engine.auto_discover(t, trajectory)
```

### Full Pipeline Example
```bash
python examples/23_full_intelligence_pipeline.py
```

### Intelligence Layer Tests
```bash
python -m pytest tests/test_intelligence.py -v
```

---

## Phase 4 — Category Theory Layer (Unifying Abstraction)

Category theory is not just another math module — it is the **unifying language** for the entire MNN system. Instead of `function → output`, you now think `object → morphism → object`.

### Core Concepts (`mnn.category.core`)

Objects, morphisms, composition, identity, products, coproducts, and diagram verification.

```python
from mnn.category.core import CatObject, Morphism, Category, make_vect_category

# Build the category of vector spaces
Vect = make_vect_category([1, 2, 3])

# Objects are typed mathematical entities
A = CatObject("S²", "manifold", data=sphere, dim=2)
B = CatObject("R³", "vector", data=np.zeros(3), dim=3)

# Morphisms are structure-preserving maps: f: A → B
f = Morphism(A, B, fn=embed_fn, name="embed")

# Composition: g ∘ f  (apply f first, then g)
h = g @ f

# Verify commutative diagrams
C.verify_commutative_diagram([["f","g"], ["h"]], test_input=x)
```

### Functors (`mnn.category.functors`)

Functors map between categories — the **glue layer** of the entire system.

```python
from mnn.category.functors import (
    GeometryToAlgebraFunctor,      # manifold → metric tensor
    AlgebraToComputationFunctor,   # group → Cayley table matrix
    DynamicsToLearningFunctor,     # ODE → neural flow map
    ForgetfulFunctor,              # forgets structure, keeps data
    UniversalBridgeFunctor,        # generic inter-module converter
)

# Natural transformations: compare different models
alpha = NaturalTransformation(F, G, components)
alpha.verify_all(test_input=x)  # checks naturality condition
```

### Neural Categories (`mnn.category.neural`)

Neural networks **are** morphisms in a category. Train entire compositions end-to-end.

```python
from mnn.category.neural import NeuralMorphism, NeuralCategory, CategoricalPipeline

# Neural morphism: NN as a map between mathematical objects
f_nn = NeuralMorphism(V2, V3, width=64, depth=3, name="embed")
f_nn.train(x_train, y_train, n_epochs=1000)

# Neural category: train chains of morphisms end-to-end
NC = NeuralCategory("MathCat")
NC.add_neural_morphism(A, B, name="encode")
NC.add_neural_morphism(B, C, name="decode")
NC.train_composition(["encode", "decode"], x_start, y_end)

# Categorical pipeline: chain morphisms across different categories
pipe = CategoricalPipeline("FullPipeline")
pipe.add_stage("normalize", normalize_morphism)
pipe.add_stage("embed", neural_embed_morphism)
result = pipe.run(input_data)
```

### Category Theory Examples
```bash
python examples/24_category_theory_core.py
python examples/25_functors_neural_categories.py
```

### Category Theory Tests
```bash
python -m pytest tests/test_category.py -v
```

---

## Phase 5 — Quantum-Inspired Neural Math

Borrows mathematical structures from quantum mechanics for richer neural representations. NOT a quantum computer — uses complex-valued state spaces, unitary dynamics, and quantum geometry.

### Part 1: Hilbert Space Layer (`mnn.quantum.hilbert`)

Complex-valued state spaces with amplitude/phase representation.

```python
from mnn.quantum.hilbert import QuantumState, HilbertSpace, QuantumGates

psi = QuantumState.uniform_superposition(4)    # |psi> = equal amplitudes
qubit = QuantumState.from_bloch(np.pi/3, 0)    # Bloch sphere
bell = psi.tensor_product(phi)                 # tensor product
S = bell.von_neumann_entropy((2, 2))           # entanglement entropy
psi_h = psi.evolve(QuantumGates.H)             # Hadamard gate
```

### Part 2: Complex-Valued Neural Networks (`mnn.quantum.complex_nn`)

Process magnitude AND phase instead of only scalar activations.

```python
from mnn.quantum.complex_nn import ComplexNeuralNetwork, ComplexTrainer

cnn = ComplexNeuralNetwork(3, 1, width=64, depth=4, activation="modrelu")
trainer = ComplexTrainer(cnn)
trainer.train(x, y, n_epochs=2000)
mag, phase = cnn.magnitude_phase(x_tensor)  # internal complex features
```

### Part 3: Unitary Transformations (`mnn.quantum.unitary`)

Norm-preserving layers (U†U = I) for stable, reversible dynamics.

```python
from mnn.quantum.unitary import UnitaryNetwork, UnitaryTrainer

un = UnitaryNetwork(3, 2, hidden_dim=64, n_blocks=4)
trainer = UnitaryTrainer(un, unitarity_weight=0.01)
trainer.train(x, y, n_epochs=1000)
print(un.total_unitarity_error())  # should be ~0
```

### Part 4: Quantum-Inspired Attention (`mnn.quantum.attention`)

Attention via amplitude overlap <φ|ψ> instead of dot products.

```python
from mnn.quantum.attention import QuantumTransformer

qt = QuantumTransformer(input_dim=8, output_dim=2, d_model=64, n_heads=4)
output, attention_weights = qt(x)
```

### Part 5: Quantum Geometric Learning (`mnn.quantum.geometric`)

Embeddings on curved quantum state manifolds (Fubini-Study geometry).

```python
from mnn.quantum.geometric import FubiniStudyMetric, QuantumGeometricNetwork

dist = FubiniStudyMetric.distance(psi, phi)     # geodesic distance
bp = FubiniStudyMetric.berry_phase(state_loop)  # Berry phase
qgn = QuantumGeometricNetwork(4, 1, state_dim=32, n_geo_layers=4)
```

### Part 6: Quantum Chaos (`mnn.quantum.chaos`)

Random matrix theory, spectral statistics, chaotic quantum dynamics.

```python
from mnn.quantum.chaos import RandomMatrixEnsemble, SpectralAnalyzer, QuantumKickedTop

goe = RandomMatrixEnsemble.goe(200)             # Gaussian Orthogonal Ensemble
stats = SpectralAnalyzer.classify_dynamics(evals)  # integrable vs chaotic
qkt = QuantumKickedTop(j=10, k=3.0)             # paradigmatic chaos model
```

### Quantum Examples
```bash
python examples/26_quantum_hilbert_complex.py
python examples/27_quantum_attention_geometry_chaos.py
```

### Quantum Tests
```bash
python -m pytest tests/test_quantum.py -v
```

---

## Phase 6 — Mathematical Discovery Engine

Automated theorem discovery: observe structures → detect patterns → generate hypotheses → validate.

### Engine 1: Mathematical Representation (`mnn.discovery.representation`)

Makes mathematics machine-readable via expression trees.

```python
from mnn.discovery.representation import ExprNode, MathEncoder

# Build expression tree: x² + sin(y)
expr = ExprNode.op("+", ExprNode.op("**", ExprNode.symbol("x"), ExprNode.constant(2)),
                        ExprNode.func("sin", ExprNode.symbol("y")))
expr.evaluate({"x": 3, "y": 1.57})   # 10.0
sp_expr = expr.to_sympy()              # SymPy roundtrip
vec = expr.to_vector()                 # neural-ready encoding
```

### Engine 2: Pattern Discovery (`mnn.discovery.patterns`)

Search for symmetries, invariants, and sequence patterns.

```python
from mnn.discovery.patterns import SymmetryDetector, SequenceAnalyzer, InvariantFinder

SequenceAnalyzer.analyze([1, 4, 9, 16, 25])           # "quadratic"
SymmetryDetector.test_commutativity(op, elements)      # True/False
InvariantFinder.find_conservation_laws(trajectory)     # conserved quantities
```

### Engine 3: Conjecture Generation (`mnn.discovery.conjectures`)

Generate hypotheses about algebraic, geometric, dynamical properties.

```python
from mnn.discovery.conjectures import ConjectureGenerator

gen = ConjectureGenerator()
gen.analyze_operation(lambda a,b: a+b, elements, "add")  # finds commutativity
gen.analyze_sequence([1, 4, 9, 16, 25])                  # finds n² pattern
gen.analyze_dynamical_system(trajectory)                  # finds conservation
print(gen.report())
```

### Engine 4: Validation (`mnn.discovery.validation`)

Numerical verification, counterexample search, symbolic proofs.

```python
from mnn.discovery.validation import ConjectureValidator, SymbolicValidator

validator = ConjectureValidator()
validator.validate_conjecture(conjecture, test_fn, domain)  # full pipeline
SymbolicValidator.verify_identity(a+b, b+a, ["a","b"])     # symbolic proof
```

### Engine 5: Neural-Guided Theorem Search (`mnn.discovery.neural_search`)

Neural network predicts promising proof steps.

```python
from mnn.discovery.neural_search import NeuralTheoremSearcher, TransformationLibrary

rules = TransformationLibrary.algebraic_rules()
searcher = NeuralTheoremSearcher(rules, beam_width=5)
path = searcher.search(start_expr, goal_fn)  # beam search with neural scoring
```

### Engine 6: Category-Theoretic Discovery (`mnn.discovery.categorical`)

Discover morphism invariants, equivalence classes, and functorial structure.

```python
from mnn.discovery.categorical import CategoricalDiscoveryEngine

engine = CategoricalDiscoveryEngine()
engine.analyze_morphism(f, properties, data)        # preserved properties
engine.analyze_equivalences(elements, equiv_fn)     # equivalence classes
engine.discover_functorial_structure(obj_map, mor_map, ...)  # functor check
```

### Discovery Examples
```bash
python examples/28_discovery_engine.py
```

### Discovery Tests
```bash
python -m pytest tests/test_discovery.py -v
```

---

## Phase 7 — Quantum-Inspired Theorem Embeddings

Bridge between symbolic mathematics, geometric learning, and quantum-inspired representation. Theorems become continuous geometric entities in a complex Hilbert space.

### Part 1: Theorem State Representation (`mnn.embeddings.theorem_states`)

Represent theorems as quantum states |T⟩ = Σ c_i |b_i⟩ over 32 basis mathematical concepts.

```python
from mnn.embeddings.theorem_states import ConceptBasis, TheoremState

basis = ConceptBasis()  # 32 math concepts
thm = TheoremState.from_concepts({"commutativity": 1.0, "symmetry": 0.8}, basis)
thm.similarity(other)   # |<T1|T2>|^2
thm.concept_entropy()   # Shannon entropy of concept distribution
```

### Part 2: Mathematical Tokenization (`mnn.embeddings.tokenizer`)

Tokenize expressions with algebraic type and categorical role annotations.

```python
from mnn.embeddings.tokenizer import MathTokenizer

tok = MathTokenizer()
tokens = tok.tokenize("forall x : sin(x)**2 + cos(x)**2 = 1")
encoded = tok.encode(expr, max_len=64)   # integer sequence
info = tok.structural_encoding(expr)      # rich structural analysis
```

### Parts 3-4: Complex Embeddings & Similarity (`mnn.embeddings.complex_embed`)

Complex-valued neural encoder with magnitude+phase and quantum inner products.

```python
from mnn.embeddings.complex_embed import TheoremEncoder, TheoremSimilarity

encoder = TheoremEncoder(vocab_size, embed_dim=64, n_heads=4)
r, i = encoder(token_ids)   # complex embedding
TheoremSimilarity.overlap(r1, i1, r2, i2)         # |<T1|T2>|^2
TheoremSimilarity.nearest_theorems(q_r, q_i, db_r, db_i, names)
```

### Part 5: Proof Trajectories (`mnn.embeddings.proof_trajectories`)

Model proofs as continuous paths through theorem space.

```python
from mnn.embeddings.proof_trajectories import ProofTrajectory, ProofNavigator

traj = ProofTrajectory(name="proof")
traj.add_step(real, imag, "step_1")
ProofNavigator.interpolate(start_r, start_i, end_r, end_i, 10)  # geodesic
ProofNavigator.analogy(a_r, a_i, b_r, b_i, c_r, c_i)  # A:B :: C:?
```

### Part 6: Category-Theoretic Embeddings (`mnn.embeddings.categorical_embed`)

Theorems as objects, proofs as morphisms — compositional + geometric.

```python
from mnn.embeddings.categorical_embed import CategoricalTheoremSpace

cat = CategoricalTheoremSpace(embed_dim, "MathTheorems")
cat.add_theorem("T1", real, imag)
cat.add_proof_morphism("T1", "T2", name="proof")
cat.compose_morphisms("f", "g")   # g ∘ f
cat.connected_components()         # proof-connected theorems
```

### Part 7: Training Objectives (`mnn.embeddings.training`)

Four training losses for mathematically structured embeddings.

```python
from mnn.embeddings.training import TheoremEmbeddingTrainer

trainer = TheoremEmbeddingTrainer(encoder)
trainer.train_structural(pairs_1, pairs_2, labels, n_epochs=100)
# Losses: structural similarity, proof continuity, algebraic consistency, topological regularization
```

### Embedding Examples
```bash
python examples/29_theorem_embeddings.py
```

### Embedding Tests
```bash
python -m pytest tests/test_embeddings.py -v
```

---

## Phase 8 — Quantum PDE Solvers

Learn PDE solution spaces geometrically using quantum-inspired representations. Intersection of PDEs, quantum computing, and geometric deep learning.

### Parts 1-2: Quantum States + Operators (`mnn.quantum_pde.states`)

PDE solutions as quantum states |Ψ(x,t)⟩ with differential operators.

```python
from mnn.quantum_pde.states import QuantumPDEState, PDEOperator

psi = QuantumPDEState.gaussian_packet(grid, center=0, width=1.0, k0=3.0)
lap = PDEOperator.laplacian_1d(128, dx)
psi_evolved = psi.evolve(lap.propagator(dt=0.001), 0.001)
```

### Parts 3-4: Neural Operator + Quantum Evolution (`mnn.quantum_pde.neural_operator`)

Learn function space → function space with unitary evolution.

```python
from mnn.quantum_pde.neural_operator import QuantumPDENet

net = QuantumPDENet(n_grid=64, width=32, n_operator_layers=4, n_evolution_steps=2)
output = net(input_field)  # learns entire PDE families
```

### Parts 5-6: Spectral Solving + Geometric Regularization (`mnn.quantum_pde.spectral`)

FFT-based exact solvers + geometric loss constraints.

```python
from mnn.quantum_pde.spectral import SpectralPDESolver, GeometricPDERegularizer

solver = SpectralPDESolver(128)
heat_traj = solver.solve_heat(u0, alpha=0.1, dt=0.01, n_steps=100)
schro_traj = solver.solve_schrodinger(psi0, V, dt=0.01, n_steps=50)
reg = GeometricPDERegularizer()  # norm + smoothness + curvature
```

### Part 7: PDE Discovery (`mnn.quantum_pde.pde_discovery`)

Discover governing equations from observed data.

```python
from mnn.quantum_pde.pde_discovery import PDEDiscoveryEngine

engine = PDEDiscoveryEngine(poly_order=3, deriv_order=3)
result = engine.discover(u_data, dx, dt)  # finds u_t = 0.5*u_xx
```

### Part 8: Category-Theoretic PDE (`mnn.quantum_pde.categorical_pde`)

Solution spaces as objects, operators as morphisms, discretization as functor.

```python
from mnn.quantum_pde.categorical_pde import PDECategory, DiscretizationFunctor

cat = PDECategory("DiffusionCat")
functor = DiscretizationFunctor(64, (0, 2*np.pi))
lap = functor.discretize_laplacian()
```

### Quantum PDE Examples
```bash
python examples/30_quantum_pde_solvers.py
```

### Quantum PDE Tests
```bash
python -m pytest tests/test_quantum_pde.py -v
```

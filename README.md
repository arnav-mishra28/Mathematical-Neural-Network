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



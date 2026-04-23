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

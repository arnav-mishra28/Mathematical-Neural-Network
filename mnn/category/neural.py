"""mnn.category.neural — Neural morphisms, learnable functors, categorical pipelines.

This is where category theory meets neural networks:
  - NeuralMorphism: a morphism whose map is a trained MNN network
  - NeuralCategory: a category where all morphisms are neural networks
  - LearnableFunctor: a functor whose obj/mor maps are themselves learnable
  - CategoricalPipeline: compose morphisms across categories via functors
  - NeuralNaturalTransformation: compare two neural models categorically
"""
from __future__ import annotations
import torch, torch.nn as nn, numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

from mnn.neural.base_network import MNNNetwork
from mnn.neural.training import MNNTrainer
from .core import CatObject, Morphism, IdentityMorphism, Category
from .functors import Functor, NaturalTransformation


# ---------------------------------------------------------------------------
# Neural Morphism
# ---------------------------------------------------------------------------

class NeuralMorphism(Morphism):
    """A morphism whose map is a neural network.

    f: A → B  where f = MNNNetwork(dim_A → dim_B).

    The network is trainable — you supply (input, target) pairs and
    the morphism updates its internal weights.
    """
    def __init__(self, domain: CatObject, codomain: CatObject,
                 width: int = 64, depth: int = 3,
                 name: str = "f_nn", device: str = "cpu"):
        dim_in = domain.dim or 1
        dim_out = codomain.dim or 1
        self._network = MNNNetwork(dim_in, dim_out, width=width, depth=depth).to(device)
        self._device = device
        self._trained = False
        self._history: Dict[str, List[float]] = {}

        def nn_fn(x):
            self._network.eval()
            with torch.no_grad():
                xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32).to(device)
                return self._network(xt).cpu().numpy()

        super().__init__(
            domain=domain, codomain=codomain, fn=nn_fn,
            name=name, properties=set(),
        )

    @property
    def network(self) -> MNNNetwork:
        return self._network

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              n_epochs: int = 1000, lr: float = 1e-3,
              batch_size: int = 256, verbose: bool = True,
              print_every: int = 200) -> Dict[str, List[float]]:
        """Train the neural morphism on (input, target) pairs."""
        trainer = MNNTrainer(self._network, lr=lr, device=self._device)
        tracker = trainer.train_supervised(
            x_train, y_train, n_epochs=n_epochs,
            batch_size=batch_size, verbose=verbose, print_every=print_every,
        )
        self._trained = True
        self._history = tracker.to_numpy()
        return self._history

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)

    def compose(self, other: Morphism) -> Morphism:
        """Compose this neural morphism with another morphism.

        If *other* is also a NeuralMorphism, returns a CompositeMorphism
        that chains the two networks (no retraining needed).
        """
        if other.codomain != self.domain:
            from .core import CompositionError
            raise CompositionError(
                f"Cannot compose {self.name} ∘ {other.name}: "
                f"codomain mismatch"
            )
        return _CompositeMorphism(self, other)

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        p = self._network.count_parameters()
        return f"NeuralMorphism({self.name}: {self.domain.name}→{self.codomain.name}, {status}, {p:,} params)"


class _CompositeMorphism(Morphism):
    """Internal: composition of two morphisms (possibly neural)."""
    def __init__(self, outer: Morphism, inner: Morphism):
        self._outer = outer
        self._inner = inner

        def composed_fn(x):
            intermediate = inner.fn(x)
            return outer.fn(intermediate)

        super().__init__(
            domain=inner.domain, codomain=outer.codomain,
            fn=composed_fn,
            name=f"{outer.name}∘{inner.name}",
        )


# ---------------------------------------------------------------------------
# Neural Category
# ---------------------------------------------------------------------------

class NeuralCategory(Category):
    """A category where all morphisms are neural networks.

    Each morphism A→B is an MNNNetwork(dim_A, dim_B).
    Training the category means training all morphisms simultaneously.
    """
    def __init__(self, name: str = "NeuralCat", device: str = "cpu"):
        super().__init__(name=name, description="Category of neural morphisms")
        self.device = device

    def add_neural_morphism(self, domain: CatObject, codomain: CatObject,
                             width: int = 64, depth: int = 3,
                             name: str = "f") -> NeuralMorphism:
        m = NeuralMorphism(domain, codomain, width, depth, name, self.device)
        self.add_morphism(m)
        return m

    def train_morphism(self, name: str, x_train: np.ndarray, y_train: np.ndarray,
                       **kwargs) -> Dict:
        m = self._morphisms[name]
        if not isinstance(m, NeuralMorphism):
            raise TypeError(f"{name} is not a NeuralMorphism")
        return m.train(x_train, y_train, **kwargs)

    def train_composition(self, path: List[str],
                           x_start: np.ndarray, y_end: np.ndarray,
                           n_epochs: int = 1000, lr: float = 1e-3,
                           verbose: bool = True, print_every: int = 200) -> Dict:
        """Train a chain of neural morphisms end-to-end.

        path: list of morphism names to compose (applied left-to-right).
        """
        networks = []
        for name in path:
            m = self._morphisms[name]
            if isinstance(m, NeuralMorphism):
                networks.append(m.network)

        if not networks:
            raise ValueError("No neural morphisms in path")

        # Build composite forward
        all_params = []
        for net in networks:
            all_params.extend(net.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr)
        loss_fn = nn.MSELoss()

        X = torch.tensor(np.array(x_start), dtype=torch.float32).to(self.device)
        Y = torch.tensor(np.array(y_end), dtype=torch.float32).to(self.device)

        history = {"loss": []}
        it = tqdm(range(n_epochs), desc="Categorical Training") if verbose else range(n_epochs)

        for ep in it:
            optimizer.zero_grad()
            h = X
            for net in networks:
                h = net(h)
            loss = loss_fn(h, Y)
            loss.backward()
            optimizer.step()
            history["loss"].append(loss.item())
            if verbose and (ep + 1) % print_every == 0:
                tqdm.write(f"[{ep+1}] loss={loss.item():.6f}")

        # Mark all as trained
        for name in path:
            m = self._morphisms[name]
            if isinstance(m, NeuralMorphism):
                m._trained = True

        return history

    def evaluate_path(self, path: List[str], x: np.ndarray) -> np.ndarray:
        """Push data through a chain of morphisms."""
        result = x
        for name in path:
            m = self._morphisms[name]
            result = m.fn(result)
        return result

    def verify_composition_numerically(self, f_name: str, g_name: str,
                                        x_test: np.ndarray, atol: float = 0.1
                                        ) -> Dict:
        """Compare g∘f evaluated step-by-step vs as a single pass.

        Returns error metrics (useful for checking if separate training
        matches end-to-end training).
        """
        f = self._morphisms[f_name]
        g = self._morphisms[g_name]
        step_result = g.fn(f.fn(x_test))

        # If there's a registered composition, compare
        comp_name = f"{g_name}∘{f_name}"
        if comp_name in self._morphisms:
            comp = self._morphisms[comp_name]
            direct_result = comp.fn(x_test)
            error = float(np.mean((step_result - direct_result) ** 2))
            return {"mse": error, "close": error < atol}

        return {"mse": None, "close": True, "note": "No registered composition"}

    def summary(self) -> str:
        lines = [f"NeuralCategory: {self.name}"]
        lines.append(f"  Objects ({len(self._objects)}):")
        for o in self._objects.values():
            lines.append(f"    {o}")
        lines.append(f"  Neural Morphisms:")
        for m in self._morphisms.values():
            if isinstance(m, NeuralMorphism):
                lines.append(f"    {m}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Learnable Functor
# ---------------------------------------------------------------------------

class LearnableFunctor(Functor):
    """A functor whose object and morphism maps are neural networks.

    F: C → D where F(A) is computed by a network, and F(f) is computed
    by transforming the morphism representation through another network.

    This enables *learning* the relationship between two mathematical domains.
    """
    def __init__(self, source: Category, target: Category,
                 obj_dim_in: int, obj_dim_out: int,
                 width: int = 64, depth: int = 3,
                 name: str = "F_learn", device: str = "cpu"):
        self._device = device
        self._obj_network = MNNNetwork(obj_dim_in, obj_dim_out,
                                        width=width, depth=depth).to(device)
        self._trained = False
        self._history: Dict[str, List[float]] = {}

        def obj_map(obj: CatObject) -> CatObject:
            if obj.data is not None:
                self._obj_network.eval()
                with torch.no_grad():
                    inp = torch.tensor(
                        np.atleast_2d(self._encode_object(obj)),
                        dtype=torch.float32
                    ).to(device)
                    out = self._obj_network(inp).cpu().numpy().flatten()
                return CatObject(
                    f"{name}({obj.name})", "transformed",
                    data=out, dim=obj_dim_out,
                    metadata={"source": obj.name},
                )
            return CatObject(f"{name}({obj.name})", "generic", dim=obj_dim_out)

        def mor_map(m: Morphism) -> Morphism:
            return Morphism(
                obj_map(m.domain), obj_map(m.codomain),
                fn=lambda x: obj_map(CatObject("temp", data=x, dim=obj_dim_in)).data,
                name=f"{name}({m.name})",
            )

        super().__init__(source, target, obj_map, mor_map, name=name)

    def _encode_object(self, obj: CatObject) -> np.ndarray:
        """Encode an object's data into a fixed-size vector."""
        if isinstance(obj.data, np.ndarray):
            d = obj.data.flatten()
            target_dim = self._obj_network.input_dim
            if len(d) >= target_dim:
                return d[:target_dim].astype(np.float32)
            return np.pad(d, (0, target_dim - len(d))).astype(np.float32)
        return np.zeros(self._obj_network.input_dim, dtype=np.float32)

    def train(self, object_pairs: List[Tuple[CatObject, np.ndarray]],
              n_epochs: int = 1000, lr: float = 1e-3,
              verbose: bool = True, print_every: int = 200) -> Dict:
        """Train the learnable functor on (source_object, target_repr) pairs."""
        X = np.array([self._encode_object(o) for o, _ in object_pairs])
        Y = np.array([y for _, y in object_pairs])

        trainer = MNNTrainer(self._obj_network, lr=lr, device=self._device)
        tracker = trainer.train_supervised(
            X, Y, n_epochs=n_epochs, verbose=verbose, print_every=print_every,
        )
        self._trained = True
        self._history = tracker.to_numpy()
        return self._history

    def __repr__(self):
        status = "trained" if self._trained else "untrained"
        return (f"LearnableFunctor({self.name}: {self.source.name}→{self.target.name}, "
                f"{status})")


# ---------------------------------------------------------------------------
# Categorical Pipeline
# ---------------------------------------------------------------------------

class CategoricalPipeline:
    """Compose morphisms across multiple categories via functors.

    This is the 'glue layer' — chains:
      topology → neural representation
      algebra  → computation
      dynamics → learning

    Example:
        pipe = CategoricalPipeline()
        pipe.add_stage("embed", Vect, manifold_embed_morphism)
        pipe.add_stage("learn", NeuralCat, flow_map_morphism)
        pipe.add_functor_bridge("geom_to_alg", GeomToAlgFunctor)
        result = pipe.run(input_data)
    """
    def __init__(self, name: str = "Pipeline"):
        self.name = name
        self._stages: List[Tuple[str, Morphism]] = []
        self._functors: List[Tuple[str, Functor]] = []

    def add_stage(self, name: str, morphism: Morphism) -> "CategoricalPipeline":
        self._stages.append((name, morphism))
        return self

    def add_functor_bridge(self, name: str, functor: Functor) -> "CategoricalPipeline":
        self._functors.append((name, functor))
        return self

    def run(self, x: Any) -> Any:
        """Push data through the pipeline."""
        result = x
        for name, morphism in self._stages:
            result = morphism.fn(result)
        return result

    def run_with_intermediates(self, x: Any) -> Dict[str, Any]:
        """Push data and record each intermediate result."""
        results = {"input": x}
        current = x
        for name, morphism in self._stages:
            current = morphism.fn(current)
            results[name] = current
        results["output"] = current
        return results

    def summary(self) -> str:
        lines = [f"CategoricalPipeline: {self.name}"]
        for i, (name, m) in enumerate(self._stages):
            lines.append(f"  [{i}] {name}: {m}")
        if self._functors:
            lines.append("  Functor bridges:")
            for name, f in self._functors:
                lines.append(f"    {name}: {f}")
        return "\n".join(lines)

    def __repr__(self):
        return f"CategoricalPipeline({self.name}, stages={len(self._stages)})"


# ---------------------------------------------------------------------------
# Neural Natural Transformation
# ---------------------------------------------------------------------------

class NeuralNaturalTransformation(NaturalTransformation):
    """Natural transformation where components are neural morphisms.

    Compares two neural models (functors F, G) by learning the
    transformation α_A: F(A) → G(A) for each object A.
    """
    def __init__(self, F: Functor, G: Functor, name: str = "α_nn",
                 width: int = 32, depth: int = 2, device: str = "cpu"):
        components = {}
        for obj in F.source.objects:
            FA = F.obj_map(obj)
            GA = G.obj_map(obj)
            dim_in = FA.dim or 1
            dim_out = GA.dim or 1
            nm = NeuralMorphism(FA, GA, width=width, depth=depth,
                                 name=f"{name}_{obj.name}", device=device)
            components[obj.name] = nm

        super().__init__(F, G, components, name=name)
        self.device = device

    def train_component(self, obj_name: str,
                        x_train: np.ndarray, y_train: np.ndarray,
                        **kwargs) -> Dict:
        """Train the component α_A for object A."""
        comp = self.components[obj_name]
        if isinstance(comp, NeuralMorphism):
            return comp.train(x_train, y_train, **kwargs)
        raise TypeError(f"Component {obj_name} is not a NeuralMorphism")

    def train_all(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  **kwargs) -> Dict[str, Dict]:
        """Train all components. data maps obj_name → (x_train, y_train)."""
        results = {}
        for name, (x, y) in data.items():
            if name in self.components:
                results[name] = self.train_component(name, x, y, **kwargs)
        return results

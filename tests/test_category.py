"""Tests for the Category Theory Engine (mnn.category)."""
import numpy as np
import torch
import pytest


# ===== Core Tests =====

class TestCatObject:
    def test_creation(self):
        from mnn.category.core import CatObject
        o = CatObject("R3", "vector", dim=3, data=np.zeros(3))
        assert o.name == "R3"
        assert o.kind == "vector"
        assert o.dim == 3

    def test_equality(self):
        from mnn.category.core import CatObject
        a = CatObject("A", "generic")
        b = CatObject("A", "generic")
        # Different instances with different id → different uid
        # Same object should equal itself
        assert a == a

    def test_hash(self):
        from mnn.category.core import CatObject
        a = CatObject("X", "vector")
        s = {a, a}
        assert len(s) == 1


class TestMorphism:
    def test_basic(self):
        from mnn.category.core import CatObject, Morphism
        A = CatObject("A", "vector", dim=2)
        B = CatObject("B", "vector", dim=3)
        f = Morphism(A, B, fn=lambda x: np.append(x, 0), name="f")
        assert f.domain == A
        assert f.codomain == B
        result = f(np.array([1.0, 2.0]))
        assert np.allclose(result, [1, 2, 0])

    def test_composition(self):
        from mnn.category.core import CatObject, Morphism
        A = CatObject("A", "vector", dim=1)
        B = CatObject("B", "vector", dim=1)
        C = CatObject("C", "vector", dim=1)
        f = Morphism(A, B, fn=lambda x: x * 2, name="f")
        g = Morphism(B, C, fn=lambda x: x + 1, name="g")
        gf = g @ f
        assert gf.domain == A
        assert gf.codomain == C
        assert gf(np.array([3.0])) == 7.0  # 3*2 + 1

    def test_composition_mismatch(self):
        from mnn.category.core import CatObject, Morphism, CompositionError
        A = CatObject("A", "vector")
        B = CatObject("B", "vector")
        C = CatObject("C", "vector")
        f = Morphism(A, B, fn=lambda x: x, name="f")
        g = Morphism(A, C, fn=lambda x: x, name="g")  # domain A, not B
        with pytest.raises(CompositionError):
            g @ f  # g's domain is A, f's codomain is B → error

    def test_identity(self):
        from mnn.category.core import CatObject, IdentityMorphism
        A = CatObject("A", "vector", dim=2)
        idA = IdentityMorphism(A)
        x = np.array([3.0, 4.0])
        assert np.allclose(idA(x), x)
        assert idA.is_endomorphism()
        assert idA.is_isomorphism()


class TestCategory:
    def test_add_objects_morphisms(self):
        from mnn.category.core import CatObject, Morphism, Category
        C = Category("C")
        A = CatObject("A", dim=1)
        B = CatObject("B", dim=1)
        C.add_objects(A, B)
        f = Morphism(A, B, fn=lambda x: x, name="f")
        C.add_morphism(f)
        assert len(C.objects) == 2
        assert len(C.hom(A, B)) >= 1

    def test_identity_laws(self):
        from mnn.category.core import CatObject, Morphism, Category
        C = Category("C")
        A = CatObject("A", dim=1)
        B = CatObject("B", dim=1)
        f = Morphism(A, B, fn=lambda x: x * 2, name="f")
        C.add_morphism(f)
        assert C.verify_identity_laws()

    def test_commutative_diagram(self):
        from mnn.category.core import CatObject, Morphism, Category
        C = Category("C")
        A = CatObject("A", dim=2)
        B = CatObject("B", dim=2)
        D = CatObject("D", dim=1)
        f = Morphism(A, B, fn=lambda x: x * 2, name="f")
        g = Morphism(B, D, fn=lambda x: np.array([np.sum(x)]), name="g")
        h = Morphism(A, D, fn=lambda x: np.array([np.sum(x * 2)]), name="h")
        C.add_morphisms(f, g, h)
        assert C.verify_commutative_diagram(
            [["f", "g"], ["h"]], test_input=np.array([1.0, 2.0]))

    def test_product_object(self):
        from mnn.category.core import CatObject, Category
        C = Category("C")
        A = CatObject("A", dim=2, data=np.array([1, 2]))
        B = CatObject("B", dim=3, data=np.array([3, 4, 5]))
        C.add_objects(A, B)
        prod = C.product_object(A, B)
        assert prod.dim == 5
        assert "A×B" in prod.name

    def test_vect_category(self):
        from mnn.category.core import make_vect_category
        V = make_vect_category([1, 2, 3])
        assert len(V.objects) == 3
        assert V.verify_identity_laws()


class TestDerivedCategories:
    def test_opposite(self):
        from mnn.category.core import CatObject, Morphism, Category, OppositeCategory
        C = Category("C")
        A = CatObject("A"); B = CatObject("B")
        C.add_morphism(Morphism(A, B, lambda x: x, "f"))
        C_op = OppositeCategory(C)
        # In C_op, f_op goes from B to A
        non_id = [m for m in C_op.morphisms
                   if not m.name.startswith("id_")]
        assert any(m.domain.name == "B" for m in non_id)

    def test_product_category(self):
        from mnn.category.core import make_vect_category, make_grp_category, ProductCategory
        V = make_vect_category([1, 2])
        G = make_grp_category()
        P = ProductCategory(V, G)
        assert len(P.objects) >= 2 * 4  # 2 vects × 4 groups

    def test_slice_category(self):
        from mnn.category.core import CatObject, Morphism, Category, SliceCategory
        C = Category("C")
        A = CatObject("A"); B = CatObject("B"); X = CatObject("X")
        C.add_morphisms(
            Morphism(A, X, lambda x: x, "f"),
            Morphism(B, X, lambda x: x, "g"),
        )
        S = SliceCategory(C, X)
        assert len(S.objects) >= 2


# ===== Functor Tests =====

class TestFunctor:
    def test_basic_functor(self):
        from mnn.category.core import CatObject, Morphism, Category
        from mnn.category.functors import Functor
        C = Category("C"); D = Category("D")
        A = CatObject("A", dim=1); B = CatObject("B", dim=1)
        C.add_objects(A, B)
        C.add_morphism(Morphism(A, B, lambda x: x*2, "f"))

        F = Functor(C, D,
                    obj_map=lambda o: CatObject(f"F({o.name})", dim=o.dim),
                    mor_map=lambda m: Morphism(
                        CatObject(f"F({m.domain.name})", dim=m.domain.dim),
                        CatObject(f"F({m.codomain.name})", dim=m.codomain.dim),
                        fn=m.fn, name=f"F({m.name})"),
                    name="F")
        F.apply_all()
        assert len(D.objects) >= 2

    def test_functor_composition(self):
        from mnn.category.core import CatObject, Category
        from mnn.category.functors import Functor
        C = Category("C"); D = Category("D"); E = Category("E")
        F = Functor(C, D,
                    obj_map=lambda o: CatObject(f"F({o.name})"),
                    mor_map=lambda m: m, name="F")
        G = Functor(D, E,
                    obj_map=lambda o: CatObject(f"G({o.name})"),
                    mor_map=lambda m: m, name="G")
        GF = G.compose(F)
        assert GF.name == "G∘F"


class TestNaturalTransformation:
    def test_naturality(self):
        from mnn.category.core import CatObject, Morphism, Category
        from mnn.category.functors import Functor, NaturalTransformation
        C = Category("C")
        A = CatObject("A", dim=1); B = CatObject("B", dim=1)
        C.add_objects(A, B)
        f = Morphism(A, B, lambda x: x * 3, "f")
        C.add_morphism(f)

        F = Functor(C, C, lambda o: o, lambda m: m, "Id")
        G = Functor(C, C, lambda o: o,
                    lambda m: Morphism(m.domain, m.codomain,
                                        lambda x: m.fn(x), m.name),
                    "Id2")

        comps = {
            A.name: Morphism(A, A, lambda x: x, "α_A"),
            B.name: Morphism(B, B, lambda x: x, "α_B"),
        }
        alpha = NaturalTransformation(F, G, comps, "α")
        assert alpha.verify_naturality(f, test_input=np.array([2.0]))


class TestBridgeFunctors:
    def test_forgetful(self):
        from mnn.category.core import make_grp_category
        from mnn.category.functors import ForgetfulFunctor
        G = make_grp_category()
        U = ForgetfulFunctor(G)
        U.apply_all()
        assert len(U.target.objects) >= 4

    def test_algebra_to_computation(self):
        from mnn.category.core import CatObject, Category
        from mnn.category.functors import AlgebraToComputationFunctor
        from mnn.algebra.groups import Group
        Alg = Category("Alg")
        z3 = Group.cyclic(3)
        Alg.add_object(CatObject("Z_3", "group", data=z3, dim=3))
        Comp = Category("Comp")
        A2C = AlgebraToComputationFunctor(Alg, Comp)
        A2C.apply_all()
        comp_objs = [o for o in Comp.objects if o.data is not None and hasattr(o.data, 'shape')]
        assert len(comp_objs) >= 1
        assert comp_objs[0].data.shape == (3, 3)


# ===== Neural Category Tests =====

class TestNeuralMorphism:
    def test_creation(self):
        from mnn.category.core import CatObject
        from mnn.category.neural import NeuralMorphism
        A = CatObject("A", dim=2)
        B = CatObject("B", dim=3)
        f = NeuralMorphism(A, B, width=16, depth=1, name="f_nn")
        assert f.domain == A
        assert f.codomain == B
        result = f.evaluate(np.array([[1.0, 2.0]]))
        assert result.shape == (1, 3)

    def test_training(self):
        from mnn.category.core import CatObject
        from mnn.category.neural import NeuralMorphism
        A = CatObject("A", dim=1)
        B = CatObject("B", dim=1)
        f = NeuralMorphism(A, B, width=16, depth=1, name="f")
        x = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
        y = (2 * x).astype(np.float32)
        history = f.train(x, y, n_epochs=200, verbose=False)
        assert "data_loss" in history
        assert f._trained


class TestNeuralCategory:
    def test_chain_training(self):
        from mnn.category.core import CatObject
        from mnn.category.neural import NeuralCategory
        NC = NeuralCategory("Test")
        A = CatObject("A", dim=1)
        B = CatObject("B", dim=2)
        C = CatObject("C", dim=1)
        NC.add_objects(A, B, C)
        NC.add_neural_morphism(A, B, width=16, depth=1, name="f")
        NC.add_neural_morphism(B, C, width=16, depth=1, name="g")

        x = np.linspace(-1, 1, 200).reshape(-1, 1).astype(np.float32)
        y = (x ** 2).astype(np.float32)
        history = NC.train_composition(["f", "g"], x, y,
                                        n_epochs=200, verbose=False)
        assert len(history["loss"]) == 200

    def test_evaluate_path(self):
        from mnn.category.core import CatObject
        from mnn.category.neural import NeuralCategory
        NC = NeuralCategory("Test")
        A = CatObject("A", dim=2)
        B = CatObject("B", dim=1)
        NC.add_neural_morphism(A, B, width=16, depth=1, name="f")
        result = NC.evaluate_path(["f"], np.array([[1.0, 2.0]]))
        assert result.shape == (1, 1)


class TestCategoricalPipeline:
    def test_pipeline(self):
        from mnn.category.core import CatObject, Morphism
        from mnn.category.neural import CategoricalPipeline
        pipe = CategoricalPipeline("test")
        A = CatObject("A", dim=2)
        B = CatObject("B", dim=2)
        pipe.add_stage("scale", Morphism(A, B, lambda x: x*3, "scale"))
        result = pipe.run(np.array([1.0, 2.0]))
        assert np.allclose(result, [3.0, 6.0])

    def test_intermediates(self):
        from mnn.category.core import CatObject, Morphism
        from mnn.category.neural import CategoricalPipeline
        pipe = CategoricalPipeline("test")
        A = CatObject("A", dim=1)
        B = CatObject("B", dim=1)
        C = CatObject("C", dim=1)
        pipe.add_stage("double", Morphism(A, B, lambda x: x*2, "d"))
        pipe.add_stage("inc", Morphism(B, C, lambda x: x+1, "i"))
        res = pipe.run_with_intermediates(np.array([5.0]))
        assert np.allclose(res["double"], [10.0])
        assert np.allclose(res["inc"], [11.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

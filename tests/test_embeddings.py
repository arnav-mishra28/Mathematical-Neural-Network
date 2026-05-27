"""Tests for Quantum-Inspired Theorem Embeddings (mnn.embeddings)."""
import numpy as np
import torch
import pytest


# ===== Part 1: Theorem States =====

class TestTheoremState:
    def test_from_concepts(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t = TheoremState.from_concepts({"commutativity": 1.0, "symmetry": 0.5}, basis)
        assert abs(np.linalg.norm(t.amplitudes) - 1.0) < 1e-10

    def test_similarity_self(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t = TheoremState.from_concepts({"commutativity": 1.0}, basis)
        assert abs(t.similarity(t) - 1.0) < 1e-6

    def test_orthogonal_similarity(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t1 = TheoremState.from_concepts({"commutativity": 1.0}, basis)
        t2 = TheoremState.from_concepts({"continuity": 1.0}, basis)
        assert t1.similarity(t2) < 1e-6

    def test_top_concepts(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t = TheoremState.from_concepts({"commutativity": 1.0, "symmetry": 0.5}, basis)
        top = t.top_concepts(2)
        assert top[0][0] == "commutativity"

    def test_entropy(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t = TheoremState.from_concepts({"commutativity": 1.0}, basis, name="pure")
        assert t.concept_entropy() < 1e-6  # pure state = 0 entropy

    def test_category_projection(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t = TheoremState.from_concepts({
            "commutativity": 1.0, "continuity": 0.5
        }, basis)
        proj = t.category_projection("algebra")
        probs = proj.probabilities()
        assert probs[basis.index("continuity")] < 1e-10

    def test_superposition(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t1 = TheoremState.from_concepts({"commutativity": 1.0}, basis, name="A")
        t2 = TheoremState.from_concepts({"continuity": 1.0}, basis, name="B")
        s = t1.superpose(t2)
        assert abs(np.linalg.norm(s.amplitudes) - 1.0) < 1e-10

    def test_density_matrix(self):
        from mnn.embeddings.theorem_states import ConceptBasis, TheoremState
        basis = ConceptBasis()
        t = TheoremState.from_concepts({"commutativity": 1.0}, basis)
        rho = t.density_matrix()
        assert abs(np.trace(rho) - 1.0) < 1e-10


class TestConceptBasis:
    def test_standard_dim(self):
        from mnn.embeddings.theorem_states import ConceptBasis
        basis = ConceptBasis()
        assert basis.dim == 32

    def test_add_concept(self):
        from mnn.embeddings.theorem_states import ConceptBasis, MathConcept
        basis = ConceptBasis()
        old_dim = basis.dim
        basis.add_concept(MathConcept("new_concept", "test"))
        assert basis.dim == old_dim + 1

    def test_category_mask(self):
        from mnn.embeddings.theorem_states import ConceptBasis
        basis = ConceptBasis()
        mask = basis.get_category_mask("algebra")
        assert mask.sum() > 0


# ===== Part 2: Tokenizer =====

class TestMathTokenizer:
    def test_tokenize(self):
        from mnn.embeddings.tokenizer import MathTokenizer
        tok = MathTokenizer()
        tokens = tok.tokenize("a + b = c")
        assert len(tokens) == 5

    def test_encode(self):
        from mnn.embeddings.tokenizer import MathTokenizer
        tok = MathTokenizer()
        enc = tok.encode("a + b", max_len=16)
        assert enc.shape == (16,)
        assert enc[0] == 2  # BOS

    def test_structural_encoding(self):
        from mnn.embeddings.tokenizer import MathTokenizer
        tok = MathTokenizer()
        info = tok.structural_encoding("sin(x) + cos(y)")
        assert info["n_functions"] >= 2
        assert info["n_operators"] >= 1


# ===== Parts 3-4: Complex Embeddings + Similarity =====

class TestComplexEmbeddings:
    def test_encoder_output_shape(self):
        from mnn.embeddings.complex_embed import TheoremEncoder
        enc = TheoremEncoder(100, embed_dim=16, n_heads=4, n_layers=1)
        ids = torch.randint(0, 100, (3, 8))
        r, i = enc(ids)
        assert r.shape == (3, 16)
        assert i.shape == (3, 16)

    def test_encoder_normalized(self):
        from mnn.embeddings.complex_embed import TheoremEncoder
        enc = TheoremEncoder(100, embed_dim=16, n_heads=4, n_layers=1)
        ids = torch.randint(1, 100, (5, 8))
        r, i = enc(ids)
        norms = torch.sqrt((r**2 + i**2).sum(dim=-1))
        assert torch.allclose(norms, torch.ones(5), atol=0.01)


class TestTheoremSimilarity:
    def test_self_similarity(self):
        from mnn.embeddings.complex_embed import TheoremSimilarity
        r = np.random.randn(8)
        i = np.random.randn(8)
        norm = np.sqrt(np.sum(r**2 + i**2)) + 1e-15
        r, i = r / norm, i / norm
        s = TheoremSimilarity.overlap(r, i, r, i)
        assert abs(s - 1.0) < 1e-6

    def test_similarity_matrix(self):
        from mnn.embeddings.complex_embed import TheoremSimilarity
        r = np.random.randn(5, 8)
        i = np.random.randn(5, 8)
        # Normalize
        norm = np.sqrt((r**2 + i**2).sum(axis=-1, keepdims=True)) + 1e-8
        r, i = r / norm, i / norm
        mat = TheoremSimilarity.similarity_matrix(r, i)
        assert mat.shape == (5, 5)
        # Diagonal should be ~1
        for j in range(5):
            assert abs(mat[j, j] - 1.0) < 1e-4

    def test_nearest(self):
        from mnn.embeddings.complex_embed import TheoremSimilarity
        r = np.eye(4, dtype=float)
        i = np.zeros((4, 4), dtype=float)
        names = ["A", "B", "C", "D"]
        nearest = TheoremSimilarity.nearest_theorems(r[0], i[0], r, i, names, 2)
        assert nearest[0][0] == "A"  # self is nearest


# ===== Part 5: Proof Trajectories =====

class TestProofTrajectories:
    def test_trajectory_distance(self):
        from mnn.embeddings.proof_trajectories import ProofTrajectory
        traj = ProofTrajectory(name="test")
        traj.add_step(np.array([1, 0, 0.0]), np.zeros(3))
        traj.add_step(np.array([0, 1, 0.0]), np.zeros(3))
        assert traj.total_distance() > 0

    def test_interpolation(self):
        from mnn.embeddings.proof_trajectories import ProofNavigator
        r1, i1 = np.array([1.0, 0]), np.array([0.0, 0])
        r2, i2 = np.array([0.0, 1]), np.array([0.0, 0])
        interp_r, interp_i = ProofNavigator.interpolate(r1, i1, r2, i2, 5)
        assert interp_r.shape == (5, 2)

    def test_analogy(self):
        from mnn.embeddings.proof_trajectories import ProofNavigator
        a_r, a_i = np.array([1, 0.0]), np.zeros(2)
        b_r, b_i = np.array([0, 1.0]), np.zeros(2)
        c_r, c_i = np.array([1, 1.0]) / np.sqrt(2), np.zeros(2)
        d_r, d_i = ProofNavigator.analogy(a_r, a_i, b_r, b_i, c_r, c_i)
        assert abs(np.linalg.norm(d_r + 1j * d_i) - 1.0) < 1e-6


# ===== Part 6: Categorical Embeddings =====

class TestCategoricalEmbed:
    def test_add_theorems(self):
        from mnn.embeddings.categorical_embed import CategoricalTheoremSpace
        cat = CategoricalTheoremSpace(4)
        cat.add_theorem("T1", np.array([1, 0, 0, 0.0]), np.zeros(4))
        cat.add_theorem("T2", np.array([0, 1, 0, 0.0]), np.zeros(4))
        assert len(cat.theorems) == 2

    def test_morphism(self):
        from mnn.embeddings.categorical_embed import CategoricalTheoremSpace
        cat = CategoricalTheoremSpace(4)
        cat.add_theorem("T1", np.array([1, 0, 0, 0.0]), np.zeros(4))
        cat.add_theorem("T2", np.array([0, 1, 0, 0.0]), np.zeros(4))
        cat.add_proof_morphism("T1", "T2", name="proof_1")
        assert len(cat.morphisms) == 1

    def test_composition(self):
        from mnn.embeddings.categorical_embed import CategoricalTheoremSpace
        cat = CategoricalTheoremSpace(4)
        cat.add_theorem("A", np.array([1, 0, 0, 0.0]), np.zeros(4))
        cat.add_theorem("B", np.array([0, 1, 0, 0.0]), np.zeros(4))
        cat.add_theorem("C", np.array([0, 0, 1, 0.0]), np.zeros(4))
        cat.add_proof_morphism("A", "B", name="f")
        cat.add_proof_morphism("B", "C", name="g")
        composed = cat.compose_morphisms("f", "g")
        assert composed is not None
        assert composed["source"] == "A"
        assert composed["target"] == "C"

    def test_connected_components(self):
        from mnn.embeddings.categorical_embed import CategoricalTheoremSpace
        cat = CategoricalTheoremSpace(4)
        cat.add_theorem("A", np.array([1, 0, 0, 0.0]), np.zeros(4))
        cat.add_theorem("B", np.array([0, 1, 0, 0.0]), np.zeros(4))
        cat.add_theorem("C", np.array([0, 0, 1, 0.0]), np.zeros(4))
        cat.add_proof_morphism("A", "B", name="f")
        comps = cat.connected_components()
        assert len(comps) == 2  # {A,B} and {C}

    def test_distance(self):
        from mnn.embeddings.categorical_embed import CategoricalTheoremSpace
        cat = CategoricalTheoremSpace(4)
        cat.add_theorem("T1", np.array([1, 0, 0, 0.0]), np.zeros(4))
        cat.add_theorem("T2", np.array([0, 1, 0, 0.0]), np.zeros(4))
        d = cat.theorem_distance("T1", "T2")
        assert abs(d - np.pi / 2) < 1e-6


# ===== Part 7: Training =====

class TestTraining:
    def test_structural_loss(self):
        from mnn.embeddings.training import StructuralSimilarityLoss
        loss_fn = StructuralSimilarityLoss()
        r1, i1 = torch.randn(3, 8), torch.randn(3, 8)
        r2, i2 = torch.randn(3, 8), torch.randn(3, 8)
        labels = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(r1, i1, r2, i2, labels)
        assert loss.item() >= 0

    def test_continuity_loss(self):
        from mnn.embeddings.training import ProofContinuityLoss
        loss_fn = ProofContinuityLoss()
        traj_r = torch.randn(5, 8)
        traj_i = torch.randn(5, 8)
        loss = loss_fn(traj_r, traj_i)
        assert loss.item() >= 0

    def test_algebraic_loss(self):
        from mnn.embeddings.training import AlgebraicConsistencyLoss
        loss_fn = AlgebraicConsistencyLoss()
        r = torch.randn(3, 8)
        i = torch.randn(3, 8)
        loss = loss_fn(r[0:1], i[0:1], r[1:2], i[1:2], r[2:3], i[2:3])
        assert loss.item() == loss.item()  # not NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

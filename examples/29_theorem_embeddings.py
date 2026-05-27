"""
Example 29 — Quantum-Inspired Theorem Embeddings: Full Pipeline
Demonstrates all 7 parts: theorem states, tokenization, complex embeddings,
similarity, proof trajectories, categorical structure, and training.
"""
import numpy as np


def main():
    print("=" * 70)
    print("  QUANTUM-INSPIRED THEOREM EMBEDDINGS — Full Pipeline")
    print("=" * 70)

    # ---- Part 1: Theorem State Representation ----
    print("\n[Part 1] Theorem State Representation")
    from mnn.embeddings.theorem_states import ConceptBasis, TheoremState

    basis = ConceptBasis()
    print(f"  {basis}")

    # Represent key theorems
    comm = TheoremState.from_concepts({
        "commutativity": 1.0, "associativity": 0.5,
        "symmetry": 0.8, "equivalence": 0.3,
    }, basis, name="commutativity_thm", statement="a+b = b+a")

    pythag = TheoremState.from_concepts({
        "metric": 1.0, "orthogonality": 0.8,
        "linearity": 0.4, "eigenvalue": 0.2,
    }, basis, name="pythagorean_thm", statement="a^2 + b^2 = c^2")

    ftc = TheoremState.from_concepts({
        "continuity": 1.0, "differentiability": 0.9,
        "integrability": 1.0, "convergence": 0.3,
    }, basis, name="fund_thm_calculus", statement="integral(f') = f(b) - f(a)")

    print(f"  {comm}")
    print(f"  {pythag}")
    print(f"  {ftc}")
    print(f"  Top concepts of commutativity: {comm.top_concepts(3)}")

    # Theorem similarity
    print(f"\n  Similarities:")
    print(f"    |<comm|pythag>|^2 = {comm.similarity(pythag):.4f}")
    print(f"    |<comm|ftc>|^2   = {comm.similarity(ftc):.4f}")
    print(f"    |<pythag|ftc>|^2 = {pythag.similarity(ftc):.4f}")
    print(f"  Entropy: comm={comm.concept_entropy():.2f}, "
          f"pythag={pythag.concept_entropy():.2f}")

    # Superposition of theorems
    hybrid = comm.superpose(pythag, 0.7+0j, 0.3+0j)
    print(f"  Hybrid theorem: {hybrid.top_concepts(3)}")

    # Category projection
    alg_comm = comm.category_projection("algebra")
    print(f"  Algebraic part of commutativity: {alg_comm.top_concepts(3)}")

    # ---- Part 2: Mathematical Tokenization ----
    print("\n[Part 2] Mathematical Tokenization")
    from mnn.embeddings.tokenizer import MathTokenizer

    tokenizer = MathTokenizer()

    expressions = [
        "a + b = b + a",
        "forall x : sin(x)**2 + cos(x)**2 = 1",
        "det(A * B) = det(A) * det(B)",
        "f(g(x)) = (f . g)(x)",
    ]
    for expr in expressions:
        tokens = tokenizer.tokenize(expr)
        encoded = tokenizer.encode(expr)
        print(f"  '{expr}'")
        print(f"    Tokens: {tokens}")
        print(f"    Encoded: {encoded[:10]}...")
        info = tokenizer.structural_encoding(expr)
        print(f"    Vars={info['n_variables']}, Ops={info['n_operators']}, "
              f"Fns={info['n_functions']}")

    # ---- Parts 3-4: Complex Embeddings + Similarity ----
    print("\n[Parts 3-4] Complex Embeddings & Theorem Similarity")
    import torch
    from mnn.embeddings.complex_embed import TheoremEncoder, TheoremSimilarity

    vocab = tokenizer.vocab
    encoder = TheoremEncoder(vocab.size, embed_dim=32, n_heads=4, n_layers=2)
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # Encode theorems
    batch = tokenizer.batch_encode(expressions, max_len=32)
    batch_t = torch.tensor(batch, dtype=torch.long)
    r, i = encoder.encode_numpy(batch)
    print(f"  Embeddings shape: real={r.shape}, imag={i.shape}")

    # Similarity matrix
    sim_mat = TheoremSimilarity.similarity_matrix(r, i)
    print(f"  Similarity matrix:")
    for idx, expr in enumerate(expressions):
        row = "    " + "  ".join(f"{sim_mat[idx, j]:.3f}" for j in range(len(expressions)))
        print(f"    {expr[:20]:>20} {row}")

    # Nearest theorem
    query_r, query_i = r[0], i[0]
    names = [e[:20] for e in expressions]
    nearest = TheoremSimilarity.nearest_theorems(query_r, query_i, r, i, names)
    print(f"  Nearest to '{expressions[0][:20]}': {nearest}")

    # Interference
    interf = TheoremSimilarity.interference_term(r[0], i[0], r[1], i[1])
    print(f"  Interference(thm0, thm1) = {interf:.4f}")

    # ---- Part 5: Proof Trajectories ----
    print("\n[Part 5] Proof Trajectories")
    from mnn.embeddings.proof_trajectories import (
        ProofTrajectory, ProofNavigator,
    )

    # Create a synthetic proof trajectory
    traj = ProofTrajectory(name="commutativity_proof")
    n_steps = 6
    for s in range(n_steps):
        alpha = s / (n_steps - 1)
        step_r = (1 - alpha) * r[0] + alpha * r[1]
        step_i = (1 - alpha) * i[0] + alpha * i[1]
        norm = np.sqrt(np.sum(step_r**2 + step_i**2) + 1e-8)
        traj.add_step(step_r / norm, step_i / norm, f"step_{s}")

    print(f"  {traj}")
    print(f"  Step distances: {[f'{d:.4f}' for d in traj.step_distances()]}")
    print(f"  Smoothness: {traj.smoothness():.6f}")
    print(f"  Curvature: {[f'{c:.4f}' for c in traj.curvature()]}")

    # Geodesic interpolation
    interp_r, interp_i = ProofNavigator.interpolate(r[0], i[0], r[2], i[2], 8)
    print(f"  Geodesic interpolation: {interp_r.shape[0]} points")

    # Neighborhood search
    nbrs = ProofNavigator.neighborhood(r[0], i[0], r, i, radius=2.0, names=names)
    print(f"  Neighborhood (radius=2.0): {nbrs}")

    # Analogy: A:B :: C:?
    d_r, d_i = ProofNavigator.analogy(r[0], i[0], r[1], i[1], r[2], i[2])
    print(f"  Analogy result shape: ({d_r.shape},)")

    # ---- Part 6: Category-Theoretic Embeddings ----
    print("\n[Part 6] Category-Theoretic Embeddings")
    from mnn.embeddings.categorical_embed import CategoricalTheoremSpace

    cat = CategoricalTheoremSpace(32, "MathTheorems")
    for idx, expr in enumerate(expressions):
        cat.add_theorem(f"T{idx}", r[idx], i[idx])

    cat.add_proof_morphism("T0", "T1", name="generalize")
    cat.add_proof_morphism("T1", "T2", name="apply_det")
    cat.add_proof_morphism("T0", "T2", name="direct")

    # Compose morphisms
    composed = cat.compose_morphisms("generalize", "apply_det")
    print(f"  Composed morphism: {composed['name'] if composed else 'None'}")

    # Connected components
    print(f"\n  {cat.summary()}")

    # Distances
    for t1 in ["T0", "T1", "T2"]:
        for t2 in ["T0", "T1", "T2"]:
            if t1 < t2:
                d = cat.theorem_distance(t1, t2)
                print(f"  dist({t1}, {t2}) = {d:.4f}")

    # ---- Part 7: Training Objectives ----
    print("\n[Part 7] Training Objectives")
    from mnn.embeddings.training import TheoremEmbeddingTrainer

    trainer = TheoremEmbeddingTrainer(encoder, lr=1e-3)

    # Create training pairs: T0 related to T1, T2 unrelated to T3
    pairs_1 = torch.tensor(batch[[0, 2]], dtype=torch.long)
    pairs_2 = torch.tensor(batch[[1, 3]], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0])

    losses = trainer.train_structural(pairs_1, pairs_2, labels,
                                        n_epochs=50, verbose=True)
    print(f"  Final loss: {losses[-1]:.6f}")

    # Final similarity after training
    r2, i2 = encoder.encode_numpy(batch)
    sim_final = TheoremSimilarity.similarity_matrix(r2, i2)
    print(f"\n  Trained similarity matrix:")
    for idx, expr in enumerate(expressions):
        row = "  ".join(f"{sim_final[idx, j]:.3f}" for j in range(len(expressions)))
        print(f"    {expr[:20]:>20}  {row}")

    print("\n" + "=" * 70)
    print("  QUANTUM-INSPIRED THEOREM EMBEDDINGS — All 7 parts complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

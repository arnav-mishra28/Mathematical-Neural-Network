"""
Example 35 — Interactive Theorem-Discovery Environment: Full Pipeline
Demonstrates all 10 modules: workspace, theorem canvas, conjecture playground,
proof assistant, knowledge graph, research notebook, AI co-researcher,
multi-agent system, and discovery dashboard.
"""
import numpy as np


def main():
    print("=" * 70)
    print("  INTERACTIVE THEOREM-DISCOVERY ENVIRONMENT")
    print("=" * 70)

    from mnn.itde.environment import TheoremDiscoveryEnvironment

    itde = TheoremDiscoveryEnvironment("MathLab")

    # ---- Module 1: Workspace ----
    print("\n[Module 1] Mathematical Workspace")
    itde.create_object("S2", "manifold", "Unit 2-sphere", dimension=2, radius=1.0, curvature=1.0)
    itde.create_object("Z12", "group", "Cyclic group", order=12, abelian=True)
    itde.create_object("heat_eq", "pde", "Heat equation", type="parabolic", order=2)
    print(itde.inspect("S2"))
    print(itde.inspect("Z12"))
    print(f"  {itde.workspace.summary()}")

    # ---- Module 2: Live Theorem Canvas ----
    print("\n[Module 2] Theorem Canvas")
    itde.add_definition("group_def", "A set G with associative operation, identity, inverses")
    itde.add_definition("subgroup_def", "A subset H ≤ G that is itself a group")
    itde.canvas.add_lemma("coset_lemma", "Cosets partition the group", ["group_def", "subgroup_def"])
    itde.add_theorem("lagrange", "|H| divides |G| for H ≤ G", ["coset_lemma"])
    itde.canvas.add_conjecture("conj_sylow", "Sylow subgroups always exist", ["lagrange"])
    itde.canvas.add_proof_idea("lagrange", "Count cosets — they partition G")
    itde.canvas.set_status("lagrange", "proved")
    print(itde.view_canvas())
    info = itde.canvas.inspect_node("lagrange")
    print(f"  Lagrange dependencies: {info['dependencies']}")
    print(f"  Proof ideas: {info['proof_ideas']}")

    # ---- Module 3: Conjecture Playground ----
    print("\n[Module 3] Conjecture Playground")
    conjs = itde.hypothesize("symmetry detected in spectral operator")
    for c in conjs:
        print(f"  {c.card()}")
    conjs2 = itde.hypothesize("periodic behavior in dynamical system")
    for c in conjs2:
        print(f"  {c.card()}")
    print(f"  {itde.conjectures.summary()}")

    # ---- Module 4: Proof Assistant ----
    print("\n[Module 4] Proof Assistant")
    suggestions = itde.suggest_proof("Prove Lagrange's theorem by counting cosets",
                                      ["algebra", "number_theory"])
    for s in suggestions:
        print(f"  ▸ {s.strategy} (score={s.applicability:.2f}): {s.description[:60]}")

    # Start a proof attempt
    itde.proofs.start_proof("Lagrange's theorem", "Direct Construction")
    itde.proofs.add_step(0, "Define cosets aH = {ah : h ∈ H}", "Well-defined")
    itde.proofs.add_step(0, "Show cosets partition G", "Equivalence relation")
    itde.proofs.add_step(0, "Each coset has |H| elements", "Bijection a ↦ ah")
    itde.proofs.add_step(0, "|G| = [G:H] × |H|", "QED")
    itde.proofs.complete_proof(0, True)
    print(f"  {itde.proofs.summary()}")

    # ---- Module 6: Knowledge Graph Explorer ----
    print("\n[Module 6] Knowledge Graph Explorer")
    itde.explorer.add_node("group_theory", "definition", "algebra",
                            "Study of groups", ["algebra"])
    itde.explorer.add_node("fundamental_group", "theorem", "topology",
                            "π₁(X) is a group", ["topology", "algebra"])
    itde.explorer.add_edge("group_theory", "fundamental_group", "generalizes")
    info = itde.explorer.explore("group_theory")
    print(f"  Explore 'group_theory': cross_domain={info['cross_domain']}")
    print(f"  Cross-domain links: {itde.explorer.cross_domain_connections()}")
    print(f"  {itde.explorer.summary()}")

    # ---- Module 7: Research Notebook ----
    print("\n[Module 7] Research Notebook")
    itde.notebook.add_markdown("# Lagrange's Theorem Investigation")
    itde.notebook.add_experiment("Verify for Z12", "All subgroup orders divide 12 ✓")
    itde.notebook.add_conjecture("Converse of Lagrange holds for all groups", "open")
    itde.notebook.add_result("Converse fails for A4 (no subgroup of order 6)")
    print(f"  {itde.notebook.summary()}")

    # ---- Module 8: AI Co-Researcher ----
    print("\n[Module 8] AI Co-Researcher")
    report = itde.investigate("Nonlinear heat equation with symmetry",
                               {"observations": [{"x": 1, "T": 0.5}]})
    # Print abbreviated
    for line in report.render().split("\n")[:8]:
        print(f"  {line}")

    # ---- Module 9: Multi-Agent Debate ----
    print("\n[Module 9] Multi-Agent Mathematics")
    contributions = itde.discuss("Spectral gap and group expansion")
    for c in contributions:
        print(f"  {c}")
    result = itde.debate("Every manifold admits a spectral decomposition")
    print(f"  Consensus: {result['consensus']}")

    # ---- Module 10: Discovery Dashboard ----
    print("\n[Module 10] Discovery Dashboard")
    itde.dashboard.add_entry("Conjecture #1: Conservation law", "conjecture", 0.87, 5, 0, "high")
    itde.dashboard.add_entry("Conjecture #2: Periodic orbit", "conjecture", 0.70, 3, 0, "medium")
    itde.dashboard.add_entry("Open Problem: Spectral characterization", "open_problem", 0.30, 1, 0, "low")
    print(itde.dashboard.render())
    print(f"  Needs proof: {len(itde.dashboard.needs_proof())} conjectures")

    # ---- Full Status ----
    print("\n[Full Status]")
    status = itde.status()
    for key, val in status.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("  ITDE — All 10 modules complete! Mathematics is now interactive.")
    print("=" * 70)


if __name__ == "__main__":
    main()

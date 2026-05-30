"""
Example 33 — Mathematical AGI Assistant: Full Pipeline
Demonstrates all 10 modules: knowledge, memory, reasoning, conjectures,
proof strategies, planning, explanation, research, and dialogue.
"""


def main():
    print("=" * 70)
    print("  MATHEMATICAL AGI ASSISTANT — Full Pipeline")
    print("=" * 70)

    from mnn.agi.assistant import MathAGIAssistant

    agi = MathAGIAssistant("MathResearchAgent").initialize()

    # ---- Module 1: Knowledge ----
    print("\n[Module 1] Knowledge Layer")
    print(f"  {agi.knowledge.summary()}")
    agi.learn("sylow_thm", "theorem", "algebra",
              "Every group of order p^k*m has a subgroup of order p^k.",
              ["group_def", "lagrange_thm"], ["algebra", "group_theory"])
    results = agi.knowledge.search("group")
    print(f"  Search 'group': {len(results)} results")
    chain = agi.knowledge.dependency_chain("field_def")
    print(f"  Dependency chain of field_def: {chain}")

    # ---- Module 2: Memory ----
    print("\n[Module 2] Mathematical Memory")
    agi.memory.concepts.store("manifold", "A topological space locally like R^n", ["topology"])
    agi.memory.research.add_conjecture("Every manifold admits a Riemannian metric")
    agi.memory.research.add_pattern("Low-dim manifolds are classifiable")
    print(f"  Memory: {agi.memory.summary()}")

    # ---- Module 3: Reasoning ----
    print("\n[Module 3] Reasoning Engine")
    chain = agi.reason("Prove that every finite abelian group is a direct product of cyclic groups")
    print(f"  {chain.summary()}")

    # Symbolic property check
    result = agi.reasoner.symbolic.check_property(
        lambda a, b: (a + b) % 7, list(range(7)), "commutativity")
    print(f"  Z/7Z commutativity: {result['holds']}")

    # Geometric reasoning
    import numpy as np
    agi.reasoner.geometric.embed("algebra", np.random.randn(16))
    agi.reasoner.geometric.embed("topology", np.random.randn(16))
    agi.reasoner.geometric.embed("geometry", np.random.randn(16))
    neighbors = agi.reasoner.geometric.nearest("algebra", 2)
    print(f"  Nearest to 'algebra': {neighbors}")

    # ---- Module 4: Conjecture ----
    print("\n[Module 4] Conjecture Engine")
    observations = [{"input": x, "output": x**2} for x in range(-3, 4)]
    conjectures = agi.conjecture(observations)
    for c in conjectures:
        print(f"  Conjecture: {c.statement} (conf={c.confidence:.2f})")
    print(f"  {agi.conjectures.summary()}")

    # ---- Module 5: Proof Strategy ----
    print("\n[Module 5] Proof Strategy Engine")
    strategies = agi.suggest_proof(["algebra", "number_theory"])
    for s in strategies:
        print(f"  Strategy: {s.name} ({s.strategy_type.value}, rate={s.success_rate:.2f})")

    # ---- Module 6: Mathematical Planner ----
    print("\n[Module 6] Mathematical Planner")
    plan = agi.plan_proof("fundamental_thm_algebra", [
        {"name": "field_extension", "description": "Construct splitting field",
         "strategy": "construction"},
        {"name": "degree_argument", "description": "Degree bounds",
         "dependencies": ["field_extension"], "strategy": "algebraic"},
        {"name": "root_existence", "description": "Show root exists",
         "dependencies": ["degree_argument"], "strategy": "contradiction"},
    ])
    print(plan)
    print(f"  Progress: {agi.planner.progress()}")

    # ---- Module 7: Explanation ----
    print("\n[Module 7] Explanation Engine")
    for level in ["beginner", "undergraduate", "research"]:
        text = agi.explain("group", level)
        # Print just the summary line
        lines = text.split("\n")
        summary = [l for l in lines if l.strip() and "===" not in l][:2]
        print(f"  [{level}] {summary[0].strip()}")

    # ---- Module 9: Research Assistant ----
    print("\n[Module 9] Research Assistant")
    inv = agi.investigate("Spectral Gap", "Study relationship between spectral gap and connectivity")
    agi.research.literature_review(["Chung spectral graph theory", "Fiedler 1973"])
    agi.research.formulate_hypothesis("Spectral gap predicts graph expansion")
    agi.research.record_finding("Cheeger inequality confirmed numerically")
    report = agi.research.complete_investigation()
    # Print abbreviated report
    for line in report.split("\n")[:8]:
        print(f"  {line}")

    # ---- Module 10: Dialogue ----
    print("\n[Module 10] Mathematical Dialogue")
    print(f"  {agi.chat('define G as a finite group of order 24')}")
    print(f"  {agi.chat('assume G is simple')}")
    print(f"  {agi.chat('what is G')}")
    print(f"  Consistency: {agi.dialogue.consistency_check()}")

    # ---- Full Status ----
    print("\n[Status] Full System Report")
    status = agi.status()
    for key, val in status.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("  MATHEMATICAL AGI ASSISTANT — All 10 modules complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

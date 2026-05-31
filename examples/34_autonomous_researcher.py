"""
Example 34 — Autonomous Scientific Researcher: Full Pipeline
Demonstrates all 10 modules: literature, knowledge graph, hypothesis generation,
experiment planning, simulation, evidence scoring, self-critique, discovery,
publication, and research roadmap. Runs a full autonomous research cycle.
"""
import numpy as np


def main():
    print("=" * 70)
    print("  AUTONOMOUS SCIENTIFIC RESEARCHER — Full Pipeline")
    print("=" * 70)

    from mnn.researcher.autonomous import AutonomousResearcher

    amsr = AutonomousResearcher("MathResearchBot")

    # ---- Module 1: Literature ----
    print("\n[Module 1] Literature Engine")
    amsr.ingest_source("Spectral Graph Theory", "textbook",
                        abstract="Comprehensive treatment of spectral methods",
                        concepts=["laplacian", "eigenvalues", "spectral_gap"],
                        theorems=["Cheeger_inequality", "Fiedler_theorem"],
                        open_problems=["Spectral characterization of graphs"])
    amsr.ingest_source("PDE Solutions via Neural Operators", "paper",
                        abstract="Neural operators for PDE families",
                        concepts=["neural_operator", "PDE", "spectral"],
                        theorems=["Universal_approximation_operators"])
    print(f"  {amsr.literature.summary()}")
    print(f"  Open problems: {amsr.literature.open_problems()}")

    # ---- Module 2: Knowledge Graph ----
    print("\n[Module 2] Research Knowledge Graph")
    from mnn.researcher.literature import ResearchEdge, RelationType
    amsr.knowledge.add_edge(ResearchEdge("Cheeger_inequality", "spectral_gap",
                                          RelationType.USES))
    amsr.knowledge.add_edge(ResearchEdge("laplacian", "eigenvalues",
                                          RelationType.IMPLIES))
    print(f"  {amsr.knowledge.summary()}")
    cross = amsr.knowledge.cross_domain_links()
    print(f"  Cross-domain links: {len(cross)}")

    # ---- Module 3: Hypothesis Generation ----
    print("\n[Module 3] Hypothesis Generator")
    observations = [{"input": x, "output": x**2 + 0.01*np.random.randn()}
                    for x in np.linspace(-3, 3, 20)]
    hypotheses = amsr.observe(observations)
    for h in hypotheses:
        print(f"  [{h.level.value}] {h.statement} (conf={h.confidence:.2f})")

    # ---- Module 4: Experiment Planning ----
    print("\n[Module 4] Experiment Planner")
    if hypotheses:
        exps = amsr.plan_experiments(0)
        for e in exps:
            print(f"  Experiment: {e.name} ({e.experiment_type.value})")
    print(f"  {amsr.experiments.summary()}")

    # ---- Module 5: Simulations ----
    print("\n[Module 5] Simulation Engine")
    heat = amsr.run_simulation("heat_1d", "pde",
                                {"n_points": 30, "n_steps": 100, "alpha": 0.1})
    print(f"  Heat 1D: success={heat.success}, max={heat.data.get('max_val', 'N/A')}")

    logistic = amsr.run_simulation("logistic_map", "dynamical",
                                    {"r": 3.9, "n_steps": 200})
    print(f"  Logistic: mean={logistic.data.get('mean', 'N/A'):.4f}, "
          f"std={logistic.data.get('std', 'N/A'):.4f}")

    harmonic = amsr.run_simulation("harmonic_oscillator", "dynamical",
                                    {"omega": 2.0, "n_steps": 1000})
    print(f"  Harmonic: energy_drift={harmonic.data.get('energy_drift', 'N/A'):.6f}")

    print(f"  {amsr.simulation.summary()}")

    # ---- Module 6: Evidence Scoring ----
    print("\n[Module 6] Evidence Scoring")
    from mnn.researcher.simulation import EvidenceItem, EvidenceType
    amsr.evidence.add_evidence(0, EvidenceItem(
        EvidenceType.THEORETICAL, "Polynomial regression R² > 0.99", True, 0.9))
    amsr.evidence.add_evidence(0, EvidenceItem(
        EvidenceType.NUMERICAL, "100 random tests passed", True, 0.85))
    amsr.evidence.add_evidence(0, EvidenceItem(
        EvidenceType.EXPERIMENTAL, "Visual inspection confirms quadratic", True, 0.7))
    score = amsr.score_evidence(0)
    print(f"  Hypothesis 0: confidence={score['confidence']}, verdict={score['verdict']}")
    print(f"  Breakdown: {score['breakdown']}")

    # ---- Module 7: Self-Critique ----
    print("\n[Module 7] Self-Critique Engine")
    if hypotheses:
        critiques = amsr.self_critique(
            hypotheses[0].statement,
            [{"supports": True, "scope": "specific"}],
            ["Data is noise-free", "Function is smooth"])
        for c in critiques:
            print(f"  [{c.severity}] {c.issue}")
    print(f"  {amsr.critique.summary()}")

    # ---- Module 8: Discovery ----
    print("\n[Module 8] Discovery Engine")
    patterns = [
        {"type": "symmetry", "description": "f(x)=f(-x) for all tested inputs",
         "domains": ["analysis"], "evidence": ["Even function tests"]},
        {"type": "cross_domain", "description": "Spectral gap ↔ graph expansion",
         "domains": ["spectral_theory", "graph_theory"],
         "evidence": ["Cheeger inequality", "Numerical verification"]},
    ]
    discoveries = amsr.detect_discoveries(patterns)
    for d in discoveries:
        print(f"  {d.discovery_type.value}: {d.title} (conf={d.confidence:.1%})")
    print(f"  {amsr.discovery.summary()}")

    # ---- Module 9: Publication ----
    print("\n[Module 9] Publication Engine")
    pub = amsr.publish("Quadratic Structure in Observed Data", {
        "objective": "identify functional relationships",
        "findings": ["Quadratic relationship y ≈ x²", "Symmetry f(x)=f(-x)"],
        "conjectures": ["Underlying conservation law exists"],
        "steps": [{"action": "regression", "description": "Polynomial fit"},
                  {"action": "simulation", "description": "Heat equation test"}],
        "references": ["Spectral Graph Theory", "PDE Solutions via Neural Operators"],
    })
    # Print abbreviated
    rendered = pub.render()
    for line in rendered.split("\n")[:15]:
        print(f"  {line}")
    print(f"  ... ({len(rendered)} chars total)")

    # ---- Module 10: Research Roadmap ----
    print("\n[Module 10] Research Roadmap")
    amsr.roadmap.generate_questions(
        findings=["Quadratic structure detected"],
        conjectures=["Conservation law conjecture"],
        discoveries=[{"type": "new_connection", "title": "Spectral-graph bridge"}])
    for q in amsr.roadmap.questions[:5]:
        print(f"  [{q.priority.value}] {q.question}")
    cycle = amsr.roadmap.research_cycle()
    print(f"  Next: {cycle}")

    # ---- Full Autonomous Cycle ----
    print("\n[Autonomous Cycle]")
    new_obs = [{"input": x, "output": np.sin(x)} for x in np.linspace(0, 2*np.pi, 15)]
    result = amsr.research_cycle(new_obs)
    print(f"  Cycle #{result['cycle']}: {len(result['steps'])} steps")
    for step in result["steps"]:
        print(f"    {step}")
    if "next_investigation" in result:
        print(f"  Next investigation: {result['next_investigation']}")

    # ---- Full Status ----
    print("\n[Full Status]")
    status = amsr.status()
    for key, val in status.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 70)
    print("  AUTONOMOUS SCIENTIFIC RESEARCHER — All 10 modules complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

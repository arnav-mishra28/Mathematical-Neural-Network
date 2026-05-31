"""
Example 36 — MathOS: AI-Powered Mathematical Operating System
Full pipeline across all 10 layers: kernel, memory, agents, knowledge graph,
simulation, discovery, proof, visualization, apps, and cloud.
"""
import numpy as np
import os


def main():
    print("=" * 60)
    print("  MathOS — AI-Powered Mathematical Operating System")
    print("=" * 60)

    from mnn.mathos.mathos import MathOS

    mos = MathOS("ResearchLab")
    print(mos.boot())

    # ---- Layer 1: Kernel ----
    print("\n[Layer 1] Mathematical Kernel")
    mos.create("S2", "manifold", domain="geometry", dimension=2, radius=1.0, curvature=1.0)
    mos.create("Z12", "group", domain="algebra", order=12)
    mos.create("I3", "matrix", data=np.eye(3), domain="linear_algebra")
    mos.create("e1", "vector", data=np.array([1.0, 0.0, 0.0]), domain="linear_algebra")
    mos.create("heat_eq", "pde", domain="analysis", type="parabolic", order=2)
    print(f"  {mos.kernel.summary()}")
    print(f"  S2: {mos.kernel.get('S2').info()}")
    print(f"  Z12 factors: {mos.kernel.get('Z12').computed.get('prime_factorization')}")
    print(f"  I3 det: {mos.kernel.get('I3').computed.get('determinant')}")

    # ---- Layer 2: Memory ----
    print("\n[Layer 2] Mathematical Memory")
    mos.memory.store("proof", "lagrange", "Coset counting proof", 0.9, ["algebra"])
    mos.memory.store("discovery", "spectral_gap", "Gap predicts expansion", 0.8, ["spectral"])
    mos.memory.store("concept", "manifold", "Locally Euclidean space")
    print(f"  {mos.memory.summary()}")
    print(f"  Recall 'lagrange': {mos.memory.recall('lagrange')}")

    # ---- Layer 3: Agents ----
    print("\n[Layer 3] Agent Runtime")
    results = mos.dispatch("Investigate spectral gap in group expansion")
    for name, result in results.items():
        print(f"  {name}: {result[:70]}...")
    mos.agents.broadcast("DiscoveryAgent", "New pattern: spectral-algebra link")
    print(f"  {mos.agents.summary()}")

    # ---- Layer 4: Knowledge Graph ----
    print("\n[Layer 4] Knowledge Graph OS")
    mos.knowledge.add_edge("S2", "Z12", "related")
    mos.knowledge.add_node("Lagrange", "theorem", "algebra",
                            "|H| divides |G|", ["group_theory"])
    mos.knowledge.add_edge("Z12", "Lagrange", "exemplifies")
    path = mos.knowledge.find_path("S2", "Lagrange")
    print(f"  Path S2→Lagrange: {path}")
    print(f"  Domains: {mos.knowledge.domains()}")
    print(f"  Cross-domain: {len(mos.knowledge.cross_domain())} links")
    print(f"  {mos.knowledge.summary()}")

    # ---- Layer 5: Simulation ----
    print("\n[Layer 5] Simulation Subsystem")
    heat = mos.simulate("heat_1d", "pde", {"n": 40, "steps": 100})
    print(f"  Heat 1D: max={heat.data.get('max', 'N/A'):.6f}")

    lorenz = mos.simulate("lorenz", "chaos", {"steps": 3000})
    print(f"  Lorenz: x_range={lorenz.data.get('x_range', 'N/A')}")

    logistic = mos.simulate("logistic_map", "chaos", {"r": 3.9, "steps": 300})
    print(f"  Logistic: mean={logistic.data.get('mean', 'N/A'):.4f}")

    harmonic = mos.simulate("harmonic", "dynamics", {"omega": 2.0, "steps": 1000})
    print(f"  Harmonic: drift={harmonic.data.get('energy_drift', 'N/A'):.6f}")

    diff2d = mos.simulate("diffusion_2d", "pde", {"n": 20, "steps": 50})
    print(f"  Diffusion 2D: max={diff2d.data.get('max', 'N/A'):.6f}")
    print(f"  {mos.simulation.summary()}")

    # ---- Layer 6: Discovery ----
    print("\n[Layer 6] Discovery Subsystem")
    cycle = mos.research_cycle({"values": [2, 4, 6, 8, 10], "symmetry": True})
    print(f"  Cycle #{cycle['cycle']}: patterns={cycle['patterns']}")
    print(f"  New questions: {cycle['questions'][:3]}")
    print(f"  {mos.discovery.summary()}")

    # ---- Layer 7: Proof ----
    print("\n[Layer 7] Proof Subsystem")
    attempt = mos.prove("Every finite group has a Sylow subgroup", "construction")
    attempt.add_step("Construct orbit of group action on cosets", "Group action")
    attempt.add_step("Count fixed points via class equation", "Burnside")
    attempt.add_step("Apply induction on group order", "Strong induction")
    attempt.verify_step(0)
    attempt.verify_step(1)
    print(attempt.render())
    suggestions = mos.proof.suggest_strategies("Manifold curvature bound")
    print(f"  Strategies for curvature: {suggestions[:3]}")
    ce = mos.proof.search_counterexample(lambda n: n**2 < 100, list(range(15)))
    print(f"  Counterexample for n²<100: n={ce}")
    print(f"  {mos.proof.summary()}")

    # ---- Layer 8: Visualization ----
    print("\n[Layer 8] Visualization Engine")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    try:
        r = mos.visualize_manifold("torus", save_path=os.path.join(out_dir, "torus.png"))
        print(f"  Torus: {r}")
        r = mos.visualize_manifold("sphere", save_path=os.path.join(out_dir, "sphere.png"))
        print(f"  Sphere: {r}")
        r = mos.visualize_manifold("klein_bottle", save_path=os.path.join(out_dir, "klein.png"))
        print(f"  Klein: {r}")
        # Lorenz attractor
        traj = lorenz.data.get("trajectory")
        if traj is not None:
            r = mos.visualize_dynamics(traj, "lorenz_attractor",
                                        os.path.join(out_dir, "lorenz.png"))
            print(f"  Lorenz dynamics: {r}")
        # Diffusion heatmap
        final = diff2d.data.get("final")
        if final is not None:
            r = mos.viz.plot_heatmap(final, "diffusion_2d",
                                      os.path.join(out_dir, "diffusion.png"))
            print(f"  Diffusion heatmap: {r}")
        # Knowledge network
        r = mos.visualize_network(os.path.join(out_dir, "network.png"))
        print(f"  Network: {r}")
    except Exception as e:
        print(f"  Viz skipped ({e})")
    print(f"  {mos.viz.summary()}")

    # ---- Layer 9: Apps ----
    print("\n[Layer 9] Application Ecosystem")
    for app in mos.apps.list_apps():
        print(f"  [{app['category']}] {app['name']}: {app['description'][:50]}")
    from mnn.mathos.apps import MathApp, AppCategory
    mos.install_app(MathApp("SpectrumAnalyzer", AppCategory.RESEARCH,
                             "Analyze spectral properties",
                             entrypoint=lambda: {"eigenvalues": [1, 2, 3]}))
    r = mos.run_app("SpectrumAnalyzer")
    print(f"  Custom app result: {r}")
    print(f"  {mos.apps.summary()}")

    # ---- Layer 10: Cloud ----
    print("\n[Layer 10] Mathematical Cloud")
    task = mos.cloud_submit("Prove Goldbach conjecture for n<10000", "proof")
    print(f"  Task: {task.task_id} → {task.status}")
    mos.cloud_share("lagrange_theorem", {"statement": "|H| divides |G|",
                                          "proof": "coset_counting"})
    print(f"  Shared knowledge: {list(mos.cloud.shared_knowledge.keys())}")
    print(f"  {mos.cloud.summary()}")

    # ---- Final Status ----
    print("\n" + mos.boot())
    print("\n" + "=" * 60)
    print("  MathOS — All 10 layers operational!")
    print("=" * 60)


if __name__ == "__main__":
    main()

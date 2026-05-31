"""Tests for MathOS — AI-Powered Mathematical Operating System."""
import numpy as np
import pytest


# ===== Layer 1: Kernel =====

class TestKernel:
    def test_create_scalar(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        e = k.create("pi", "scalar", data=3.14159)
        assert e.computed["is_positive"]

    def test_create_vector(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        v = np.array([1.0, 0.0, 0.0])
        e = k.create("e1", "vector", data=v)
        assert e.computed["dimension"] == 3
        assert e.computed["is_unit"]

    def test_create_matrix(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        m = np.eye(3)
        e = k.create("I3", "matrix", data=m)
        assert abs(e.computed["determinant"] - 1.0) < 1e-10
        assert e.computed["is_symmetric"]

    def test_create_group(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        e = k.create("Z12", "group", order=12)
        assert e.computed["prime_factorization"] == [2, 2, 3]

    def test_create_manifold(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        e = k.create("S2", "manifold", dimension=2, radius=1.0, curvature=1.0)
        assert abs(e.computed["area"] - 4 * np.pi) < 0.01

    def test_search(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        k.create("A", "group", order=5)
        k.create("B", "manifold", dimension=3)
        assert len(k.search("group")) == 1

    def test_relate(self):
        from mnn.mathos.kernel import MathKernel
        k = MathKernel()
        k.create("A", "scalar", data=1)
        k.create("B", "scalar", data=2)
        k.relate("A", "B", "less_than")
        assert len(k.get("A").relations) >= 1


# ===== Layer 2: Memory =====

class TestMemory:
    def test_store_recall(self):
        from mnn.mathos.kernel import MathMemorySystem
        m = MathMemorySystem()
        m.store("concept", "group_def", "A set with operation")
        assert m.recall("group_def") == "A set with operation"

    def test_search(self):
        from mnn.mathos.kernel import MathMemorySystem
        m = MathMemorySystem()
        m.store("proof", "lagrange_proof", "Coset counting", tags=["algebra"])
        results = m.search("algebra")
        assert len(results) == 1

    def test_by_kind(self):
        from mnn.mathos.kernel import MathMemorySystem
        m = MathMemorySystem()
        m.store("concept", "a", "x")
        m.store("proof", "b", "y")
        assert len(m.by_kind("concept")) == 1


# ===== Layer 3: Agent Runtime =====

class TestAgentRuntime:
    def test_dispatch(self):
        from mnn.mathos.agents import AgentRuntime
        rt = AgentRuntime()
        results = rt.dispatch("group symmetry analysis")
        assert len(results) == 5  # 5 default agents

    def test_message(self):
        from mnn.mathos.agents import AgentRuntime
        rt = AgentRuntime()
        rt.send_message("AlgebraAgent", "GeometryAgent", "Check manifold", "query")
        assert len(rt.agents["GeometryAgent"].inbox) == 1

    def test_broadcast(self):
        from mnn.mathos.agents import AgentRuntime
        rt = AgentRuntime()
        rt.broadcast("AlgebraAgent", "New theorem found")
        assert all(len(a.inbox) >= 1 for n, a in rt.agents.items() if n != "AlgebraAgent")


# ===== Layer 4: Knowledge Graph OS =====

class TestKGOS:
    def test_path(self):
        from mnn.mathos.agents import KnowledgeGraphOS
        kg = KnowledgeGraphOS()
        kg.add_node("A", "theorem", "algebra", "")
        kg.add_node("B", "theorem", "algebra", "")
        kg.add_node("C", "theorem", "topology", "")
        kg.add_edge("A", "B", "implies")
        kg.add_edge("B", "C", "generalizes")
        path = kg.find_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_cross_domain(self):
        from mnn.mathos.agents import KnowledgeGraphOS
        kg = KnowledgeGraphOS()
        kg.add_node("X", "theorem", "algebra", "")
        kg.add_node("Y", "theorem", "topology", "")
        kg.add_edge("X", "Y", "related")
        links = kg.cross_domain()
        assert len(links) == 1


# ===== Layer 5: Simulation =====

class TestSimulation:
    def test_heat(self):
        from mnn.mathos.simulation import SimulationSubsystem
        ss = SimulationSubsystem()
        r = ss.run("heat_1d", "pde", {"n": 30, "steps": 50})
        assert r.success

    def test_lorenz(self):
        from mnn.mathos.simulation import SimulationSubsystem
        ss = SimulationSubsystem()
        r = ss.run("lorenz", "chaos", {"steps": 500})
        assert r.success
        assert "trajectory" in r.data

    def test_logistic(self):
        from mnn.mathos.simulation import SimulationSubsystem
        ss = SimulationSubsystem()
        r = ss.run("logistic_map", "chaos", {"r": 3.9})
        assert r.success

    def test_harmonic(self):
        from mnn.mathos.simulation import SimulationSubsystem
        ss = SimulationSubsystem()
        r = ss.run("harmonic", "dynamics", {"omega": 2.0, "steps": 500})
        assert r.success
        assert r.data["energy_drift"] < 0.1

    def test_diffusion_2d(self):
        from mnn.mathos.simulation import SimulationSubsystem
        ss = SimulationSubsystem()
        r = ss.run("diffusion_2d", "pde", {"n": 15, "steps": 30})
        assert r.success


# ===== Layer 6: Discovery =====

class TestDiscovery:
    def test_cycle(self):
        from mnn.mathos.simulation import DiscoverySubsystem
        ds = DiscoverySubsystem()
        result = ds.research_cycle({"values": [1, 2, 3, 4, 5]})
        assert result["cycle"] == 1
        assert "arithmetic_progression" in result["patterns"]

    def test_validate(self):
        from mnn.mathos.simulation import DiscoverySubsystem
        ds = DiscoverySubsystem()
        ds.hypothesize(["symmetry"])
        ok = ds.validate(0, lambda x: x > 0, [1, 2, 3, 4])
        assert ok is True


# ===== Layer 7: Proof =====

class TestProof:
    def test_plan(self):
        from mnn.mathos.proof import ProofSubsystem
        ps = ProofSubsystem()
        attempt = ps.plan_proof("Every natural number > 1 has prime factorization",
                                 "induction")
        attempt.add_step("Base case: n=2", "2 is prime")
        attempt.verify_step(0)
        assert attempt.steps[0].status == "verified"

    def test_suggest(self):
        from mnn.mathos.proof import ProofSubsystem
        ps = ProofSubsystem()
        strats = ps.suggest_strategies("Manifold curvature bound")
        assert strats[0] == "geometric"

    def test_counterexample(self):
        from mnn.mathos.proof import ProofSubsystem
        ps = ProofSubsystem()
        ce = ps.search_counterexample(lambda x: x > 0, [-3, -2, -1, 0, 1, 2])
        assert ce == -3


# ===== Layer 8: Visualization =====

class TestVisualization:
    def test_engine_exists(self):
        from mnn.mathos.proof import VisualizationEngine
        ve = VisualizationEngine()
        assert ve.summary() == "VisualizationEngine: 0 figures, types={}"


# ===== Layer 9: Apps =====

class TestApps:
    def test_builtin(self):
        from mnn.mathos.apps import AppEcosystem
        ae = AppEcosystem()
        assert len(ae.apps) == 6

    def test_by_category(self):
        from mnn.mathos.apps import AppEcosystem
        ae = AppEcosystem()
        research = ae.by_category("research")
        assert len(research) == 1

    def test_install_run(self):
        from mnn.mathos.apps import AppEcosystem, MathApp, AppCategory
        ae = AppEcosystem()
        ae.install(MathApp("TestApp", AppCategory.CUSTOM, "test",
                            entrypoint=lambda: {"result": 42}))
        r = ae.run_app("TestApp")
        assert r["result"] == 42


# ===== Layer 10: Cloud =====

class TestCloud:
    def test_submit(self):
        from mnn.mathos.apps import MathCloud
        mc = MathCloud()
        task = mc.submit_task("Prove theorem X", "proof")
        assert task.status == "running"

    def test_share(self):
        from mnn.mathos.apps import MathCloud
        mc = MathCloud()
        mc.share_knowledge("thm1", {"statement": "X"})
        assert mc.get_shared("thm1") is not None


# ===== Unified MathOS =====

class TestMathOS:
    def test_boot(self):
        from mnn.mathos.mathos import MathOS
        os = MathOS("TestOS")
        boot = os.boot()
        assert "MathOS" in boot
        assert "Layer 1" in boot

    def test_full_pipeline(self):
        from mnn.mathos.mathos import MathOS
        os = MathOS("Lab")
        # Create objects
        os.create("S2", "manifold", domain="geometry", dimension=2, radius=1.0, curvature=1.0)
        os.create("Z6", "group", domain="algebra", order=6)
        assert len(os.kernel.entities) == 2
        # Dispatch
        results = os.dispatch("Analyze group structure")
        assert len(results) == 5
        # Simulate
        r = os.simulate("heat_1d", "pde", {"n": 20, "steps": 30})
        assert r.success
        # Research
        cycle = os.research_cycle({"values": [1, 3, 5, 7, 9]})
        assert cycle["cycle"] == 1
        # Prove
        attempt = os.prove("Lagrange's theorem", "induction")
        assert attempt.strategy == "induction"
        # Cloud
        task = os.cloud_submit("Test task")
        assert task.status == "running"
        # Status
        status = os.status()
        assert status["name"] == "Lab"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

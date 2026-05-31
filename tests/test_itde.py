"""Tests for Interactive Theorem-Discovery Environment (mnn.itde)."""
import numpy as np
import pytest


# ===== Module 1: Workspace =====

class TestWorkspace:
    def test_create_group(self):
        from mnn.itde.workspace import MathWorkspace
        ws = MathWorkspace()
        obj = ws.create("Z6", "group", "Cyclic group", order=6, abelian=True)
        assert obj.computed["order"] == 6
        assert obj.computed["prime_factorization"] == [2, 3]

    def test_create_manifold(self):
        from mnn.itde.workspace import MathWorkspace
        ws = MathWorkspace()
        obj = ws.create("S2", "manifold", "2-sphere", dimension=2,
                         radius=1.0, curvature=1.0)
        assert abs(obj.computed["area"] - 4 * np.pi) < 0.01
        assert obj.computed["curvature_type"] == "positive"

    def test_search(self):
        from mnn.itde.workspace import MathWorkspace
        ws = MathWorkspace()
        ws.create("S2", "manifold", "sphere")
        ws.create("T2", "manifold", "torus")
        results = ws.search("manifold")
        assert len(results) == 2

    def test_connect(self):
        from mnn.itde.workspace import MathWorkspace
        ws = MathWorkspace()
        ws.create("A", "group", "")
        ws.create("B", "group", "")
        ws.connect("A", "B", "subgroup")
        assert len(ws.get("A").relations) >= 1


# ===== Module 2: Theorem Canvas =====

class TestCanvas:
    def test_build_tree(self):
        from mnn.itde.workspace import TheoremCanvas
        tc = TheoremCanvas()
        tc.add_definition("def_group", "A set with operation...")
        tc.add_lemma("lem_order", "Order divides |G|", ["def_group"])
        tc.add_theorem("lagrange", "Lagrange's theorem", ["lem_order"])
        tree = tc.render_tree()
        assert "def_group" in tree
        assert "lagrange" in tree

    def test_inspect(self):
        from mnn.itde.workspace import TheoremCanvas
        tc = TheoremCanvas()
        tc.add_theorem("thm_A", "Statement A")
        tc.add_proof_idea("thm_A", "Try induction")
        info = tc.inspect_node("thm_A")
        assert len(info["proof_ideas"]) == 1

    def test_conjecture(self):
        from mnn.itde.workspace import TheoremCanvas
        tc = TheoremCanvas()
        tc.add_conjecture("conj_1", "All primes > 2 are odd")
        assert tc.nodes["conj_1"].status.value == "conjectured"


# ===== Module 3: Conjecture Playground =====

class TestConjecturePlayground:
    def test_generate(self):
        from mnn.itde.conjecture_proof import ConjecturePlayground
        cp = ConjecturePlayground()
        conjs = cp.generate_conjectures("symmetry detected in operator")
        assert len(conjs) >= 2
        assert any("conserved" in c.statement.lower() for c in conjs)

    def test_test_conjecture(self):
        from mnn.itde.conjecture_proof import ConjecturePlayground
        cp = ConjecturePlayground()
        cp.generate_conjectures("linear pattern")
        result = cp.test_conjecture(0, lambda x: x > 0, [1, 2, 3])
        assert result["passed"] == 3


# ===== Module 4: Proof Assistant =====

class TestProofAssistant:
    def test_suggest(self):
        from mnn.itde.conjecture_proof import ProofAssistant
        pa = ProofAssistant()
        suggestions = pa.suggest("Prove by induction on n", ["number_theory"])
        assert suggestions[0].strategy == "Induction"

    def test_proof_attempt(self):
        from mnn.itde.conjecture_proof import ProofAssistant
        pa = ProofAssistant()
        attempt = pa.start_proof("Prove X", "Induction")
        pa.add_step(0, "Base case: n=0", "Holds")
        pa.add_step(0, "Inductive step", "Assume n, prove n+1")
        pa.complete_proof(0, True)
        assert pa.proof_attempts[0]["status"] == "proved"


# ===== Module 6: Knowledge Graph Explorer =====

class TestExplorer:
    def test_explore(self):
        from mnn.itde.explorer import KnowledgeGraphExplorer
        kge = KnowledgeGraphExplorer()
        kge.add_node("group_def", "definition", "algebra", "Set with operation")
        kge.add_node("topology", "definition", "topology", "Open sets")
        kge.add_edge("group_def", "topology", "related")
        info = kge.explore("group_def")
        assert "topology" in info["cross_domain"]

    def test_neighborhood(self):
        from mnn.itde.explorer import KnowledgeGraphExplorer
        kge = KnowledgeGraphExplorer()
        kge.add_node("A", "theorem", "algebra", "")
        kge.add_node("B", "theorem", "algebra", "")
        kge.add_node("C", "theorem", "algebra", "")
        kge.add_edge("A", "B", "implies")
        kge.add_edge("B", "C", "implies")
        nbr = kge.neighborhood("A", depth=2)
        assert "C" in nbr


# ===== Module 7: Research Notebook =====

class TestNotebook:
    def test_notebook(self):
        from mnn.itde.explorer import ResearchNotebook
        nb = ResearchNotebook("Test Notebook")
        nb.add_markdown("# Introduction")
        nb.add_experiment("Heat equation test", "Converged in 100 steps")
        nb.add_conjecture("Exponential decay", "open")
        rendered = nb.render()
        assert "Introduction" in rendered
        assert "Heat equation" in rendered
        assert len(nb.cells) == 3


# ===== Module 8: AI Co-Researcher =====

class TestCoResearcher:
    def test_investigate(self):
        from mnn.itde.collaboration import AICoResearcher
        cr = AICoResearcher()
        report = cr.investigate("nonlinear PDE family",
                                 {"observations": [1, 2, 3]})
        assert report.status == "completed"
        assert len(report.simulations) >= 1
        assert "PDE" in report.render()


# ===== Module 9: Multi-Agent =====

class TestMultiAgent:
    def test_discuss(self):
        from mnn.itde.collaboration import MultiAgentSystem
        mas = MultiAgentSystem()
        contributions = mas.discuss("group symmetry in manifolds")
        assert len(contributions) == 4  # 4 agents

    def test_debate(self):
        from mnn.itde.collaboration import MultiAgentSystem
        mas = MultiAgentSystem()
        result = mas.debate("PDE has a spectral structure")
        assert "consensus" in result


# ===== Module 10: Discovery Dashboard =====

class TestDashboard:
    def test_track(self):
        from mnn.itde.collaboration import DiscoveryDashboard
        dd = DiscoveryDashboard()
        dd.add_entry("Conjecture A", "conjecture", 0.87, 5, 0)
        dd.add_entry("Problem B", "open_problem", 0.3, 1, 2)
        assert len(dd.active_entries()) == 2
        assert len(dd.high_confidence(0.8)) == 1
        assert len(dd.needs_proof()) == 1

    def test_render(self):
        from mnn.itde.collaboration import DiscoveryDashboard
        dd = DiscoveryDashboard()
        dd.add_entry("Test", "conjecture", 0.5)
        rendered = dd.render()
        assert "DASHBOARD" in rendered


# ===== Unified ITDE =====

class TestITDE:
    def test_full_pipeline(self):
        from mnn.itde.environment import TheoremDiscoveryEnvironment
        itde = TheoremDiscoveryEnvironment("TestLab")
        # Create objects
        itde.create_object("S2", "manifold", "2-sphere", dimension=2, radius=1.0, curvature=1.0)
        assert "S2" in itde.inspect("S2")
        # Canvas
        itde.add_definition("group_def", "Set with associative op, id, inverses")
        itde.add_theorem("lagrange", "Order of subgroup divides order of group",
                          ["group_def"])
        tree = itde.view_canvas()
        assert "lagrange" in tree
        # Conjectures
        conjs = itde.hypothesize("symmetry in the spectrum")
        assert len(conjs) >= 1
        # Proof
        suggestions = itde.suggest_proof("Prove by spectral decomposition")
        assert any(s.strategy == "Spectral Decomposition" for s in suggestions)
        # Investigate
        report = itde.investigate("Heat equation regularity")
        assert report.status == "completed"
        # Discuss
        contributions = itde.discuss("Group theory and topology connection")
        assert len(contributions) >= 3
        # Status
        status = itde.status()
        assert status["name"] == "TestLab"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

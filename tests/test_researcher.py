"""Tests for Autonomous Scientific Researcher (mnn.researcher)."""
import numpy as np
import pytest


# ===== Module 1: Literature =====

class TestLiterature:
    def test_add_and_search(self):
        from mnn.researcher.literature import LiteratureEngine, LiteratureSource, SourceType
        le = LiteratureEngine()
        le.add_source(LiteratureSource("Spectral Graph Theory", SourceType.TEXTBOOK,
                                        key_concepts=["spectral", "graph", "laplacian"]))
        results = le.search("spectral")
        assert len(results) == 1

    def test_open_problems(self):
        from mnn.researcher.literature import LiteratureEngine, LiteratureSource, SourceType
        le = LiteratureEngine()
        le.add_source(LiteratureSource("Paper A", SourceType.PAPER,
                                        open_problems=["Prove conjecture X"]))
        problems = le.open_problems()
        assert len(problems) == 1


# ===== Module 2: Knowledge Graph =====

class TestKnowledgeGraph:
    def test_add_find_path(self):
        from mnn.researcher.literature import ResearchKnowledgeGraph, ResearchNode, ResearchEdge, RelationType
        kg = ResearchKnowledgeGraph()
        kg.add_node(ResearchNode("A", "theorem", "algebra", "Thm A"))
        kg.add_node(ResearchNode("B", "theorem", "algebra", "Thm B"))
        kg.add_node(ResearchNode("C", "theorem", "topology", "Thm C"))
        kg.add_edge(ResearchEdge("A", "B", RelationType.IMPLIES))
        kg.add_edge(ResearchEdge("B", "C", RelationType.GENERALIZES))
        path = kg.find_path("A", "C")
        assert path == ["A", "B", "C"]

    def test_cross_domain(self):
        from mnn.researcher.literature import ResearchKnowledgeGraph, ResearchNode, ResearchEdge, RelationType
        kg = ResearchKnowledgeGraph()
        kg.add_node(ResearchNode("X", "theorem", "algebra", ""))
        kg.add_node(ResearchNode("Y", "theorem", "topology", ""))
        kg.add_edge(ResearchEdge("X", "Y", RelationType.RELATED))
        links = kg.cross_domain_links()
        assert len(links) == 1


# ===== Module 3: Hypothesis Generator =====

class TestHypothesis:
    def test_linear(self):
        from mnn.researcher.hypothesis import HypothesisGenerator
        hg = HypothesisGenerator()
        obs = [{"input": x, "output": 3*x + 1} for x in range(5)]
        hyps = hg.generate(obs)
        assert any("linear" in h.statement.lower() for h in hyps)

    def test_quadratic(self):
        from mnn.researcher.hypothesis import HypothesisGenerator
        hg = HypothesisGenerator()
        obs = [{"input": x, "output": x**2} for x in range(5)]
        hyps = hg.generate(obs)
        assert any("quadratic" in h.statement.lower() for h in hyps)


# ===== Module 4: Experiment Planner =====

class TestExperimentPlanner:
    def test_plan(self):
        from mnn.researcher.hypothesis import HypothesisGenerator, ExperimentPlanner
        hg = HypothesisGenerator()
        obs = [{"input": x, "output": x**2} for x in range(5)]
        hyps = hg.generate(obs)
        ep = ExperimentPlanner()
        exps = ep.plan_experiments(hyps[0], 0)
        assert len(exps) >= 2  # numerical + counterexample


# ===== Module 5: Simulation =====

class TestSimulation:
    def test_heat(self):
        from mnn.researcher.simulation import SimulationEngine, SimulationType
        se = SimulationEngine()
        result = se.run("heat_1d", SimulationType.PDE, {"n_points": 20, "n_steps": 50})
        assert result.success
        assert "trajectory" in result.data

    def test_logistic(self):
        from mnn.researcher.simulation import SimulationEngine, SimulationType
        se = SimulationEngine()
        result = se.run("logistic_map", SimulationType.DYNAMICAL, {"r": 3.9, "n_steps": 100})
        assert result.success

    def test_harmonic(self):
        from mnn.researcher.simulation import SimulationEngine, SimulationType
        se = SimulationEngine()
        result = se.run("harmonic_oscillator", SimulationType.DYNAMICAL,
                         {"omega": 2.0, "n_steps": 500})
        assert result.success
        assert result.data["energy_drift"] < 0.1


# ===== Module 6: Evidence =====

class TestEvidence:
    def test_scoring(self):
        from mnn.researcher.simulation import EvidenceScorer, EvidenceItem, EvidenceType
        es = EvidenceScorer()
        es.add_evidence(0, EvidenceItem(EvidenceType.THEORETICAL, "Proof sketch", True, 0.8))
        es.add_evidence(0, EvidenceItem(EvidenceType.NUMERICAL, "100 tests passed", True, 0.9))
        score = es.score(0)
        assert score["confidence"] > 0.5
        assert score["verdict"] in ("strongly_supported", "moderately_supported")

    def test_counterexample_override(self):
        from mnn.researcher.simulation import EvidenceScorer, EvidenceItem, EvidenceType
        es = EvidenceScorer()
        es.add_evidence(0, EvidenceItem(EvidenceType.THEORETICAL, "Proof", True, 0.9))
        es.add_evidence(0, EvidenceItem(EvidenceType.COUNTEREXAMPLE, "Found x=5", False, 1.0))
        score = es.score(0)
        assert score["confidence"] <= 0.2


# ===== Module 7: Self-Critique =====

class TestCritique:
    def test_hypothesis_critique(self):
        from mnn.researcher.critique import SelfCritiqueEngine
        sc = SelfCritiqueEngine()
        critiques = sc.critique_hypothesis("X is always true", [{"supports": True}],
                                            ["X is positive"])
        assert len(critiques) >= 1

    def test_numerical_critique(self):
        from mnn.researcher.critique import SelfCritiqueEngine
        sc = SelfCritiqueEngine()
        values = np.array([1.0, np.nan, 3.0])
        critiques = sc.critique_numerical(values, "test")
        assert any(c.severity == "critical" for c in critiques)


# ===== Module 8: Discovery =====

class TestDiscovery:
    def test_patterns(self):
        from mnn.researcher.critique import DiscoveryEngine
        de = DiscoveryEngine()
        patterns = [
            {"type": "symmetry", "description": "Even function detected",
             "domains": ["analysis"], "evidence": ["f(x)=f(-x)"]},
            {"type": "cross_domain", "description": "Algebra-topology link",
             "domains": ["algebra", "topology"], "evidence": ["functor"]},
        ]
        discoveries = de.analyze_patterns(patterns)
        assert len(discoveries) == 2


# ===== Module 9: Publication =====

class TestPublication:
    def test_generate(self):
        from mnn.researcher.publication import PublicationEngine
        pe = PublicationEngine()
        inv = {"findings": ["Result A", "Result B"],
               "conjectures": ["Conjecture X"],
               "objective": "study spectral properties"}
        pub = pe.generate("Spectral Analysis", inv)
        text = pub.render()
        assert "ABSTRACT" in text
        assert "Spectral Analysis" in text


# ===== Module 10: Roadmap =====

class TestRoadmap:
    def test_generate_and_select(self):
        from mnn.researcher.publication import ResearchRoadmap
        rr = ResearchRoadmap()
        rr.generate_questions(findings=["Linear decay observed"],
                              conjectures=["Exponential growth conjecture"])
        assert len(rr.questions) >= 2
        next_q = rr.next_investigation()
        assert next_q is not None

    def test_cycle(self):
        from mnn.researcher.publication import ResearchRoadmap
        rr = ResearchRoadmap()
        rr.generate_questions(conjectures=["Test conjecture"])
        result = rr.research_cycle()
        assert result["status"] == "investigating"


# ===== Unified Autonomous Researcher =====

class TestAutonomous:
    def test_full_cycle(self):
        from mnn.researcher.autonomous import AutonomousResearcher
        ar = AutonomousResearcher("TestAMSR")
        # Ingest
        ar.ingest_source("Paper A", "paper", abstract="Study of X",
                          concepts=["spectral"], theorems=["Thm1"])
        # Research cycle
        obs = [{"input": x, "output": x**2 + 1} for x in range(-3, 4)]
        result = ar.research_cycle(obs)
        assert result["cycle"] == 1
        assert len(result["steps"]) >= 3
        # Status
        status = ar.status()
        assert status["name"] == "TestAMSR"

    def test_simulation(self):
        from mnn.researcher.autonomous import AutonomousResearcher
        ar = AutonomousResearcher()
        result = ar.run_simulation("heat_1d", "pde", {"n_points": 20, "n_steps": 50})
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Tests for Mathematical AGI Assistant (mnn.agi)."""
import numpy as np
import pytest


# ===== Module 1: Knowledge Layer =====

class TestKnowledge:
    def test_add_and_search(self):
        from mnn.agi.knowledge import MathKnowledgeGraph, KnowledgeNode, KnowledgeType, Domain
        kg = MathKnowledgeGraph()
        kg.add_node(KnowledgeNode("group", KnowledgeType.DEFINITION, Domain.ALGEBRA,
                                   "A set with associative operation, identity, inverses."))
        results = kg.search("group")
        assert len(results) == 1

    def test_standard_knowledge(self):
        from mnn.agi.knowledge import MathKnowledgeGraph
        kg = MathKnowledgeGraph()
        kg.add_standard_knowledge()
        assert len(kg.nodes) >= 10
        assert len(kg.edges) > 0

    def test_dependency_chain(self):
        from mnn.agi.knowledge import MathKnowledgeGraph
        kg = MathKnowledgeGraph()
        kg.add_standard_knowledge()
        chain = kg.dependency_chain("field_def")
        assert "field_def" in chain

    def test_topological_sort(self):
        from mnn.agi.knowledge import MathKnowledgeGraph
        kg = MathKnowledgeGraph()
        kg.add_standard_knowledge()
        order = kg.topological_sort()
        assert len(order) > 0

    def test_adjacency_matrix(self):
        from mnn.agi.knowledge import MathKnowledgeGraph
        kg = MathKnowledgeGraph()
        kg.add_standard_knowledge()
        A, names = kg.adjacency_matrix()
        assert A.shape[0] == len(names)


# ===== Module 2: Memory =====

class TestMemory:
    def test_concept_memory(self):
        from mnn.agi.memory import MathematicalMemory
        mem = MathematicalMemory()
        mem.concepts.store("group", "A set with operation...", ["algebra"])
        assert mem.concepts.recall("group") is not None
        assert mem.concepts.recall("nonexistent") is None

    def test_proof_memory(self):
        from mnn.agi.memory import MathematicalMemory
        mem = MathematicalMemory()
        mem.proofs.store_strategy("induction", "Base + step", ["number_theory"])
        strategies = mem.proofs.recall_strategies("number_theory")
        assert len(strategies) >= 1

    def test_research_memory(self):
        from mnn.agi.memory import MathematicalMemory
        mem = MathematicalMemory()
        mem.research.add_conjecture("All primes > 2 are odd")
        mem.research.add_pattern("Primes thin out", evidence="PNT")
        assert len(mem.research.open_conjectures()) == 1
        assert len(mem.research.patterns) == 1


# ===== Module 3: Reasoning =====

class TestReasoning:
    def test_symbolic(self):
        from mnn.agi.reasoning import SymbolicReasoner
        sr = SymbolicReasoner()
        result = sr.apply_rule("commutativity", "a*b")
        assert result is not None

    def test_geometric(self):
        from mnn.agi.reasoning import GeometricReasoner
        gr = GeometricReasoner()
        gr.embed("A", np.array([1, 0, 0.0]))
        gr.embed("B", np.array([0, 1, 0.0]))
        sim = gr.similarity("A", "B")
        assert abs(sim) < 0.1

    def test_hybrid_chain(self):
        from mnn.agi.reasoning import HybridReasoner
        hr = HybridReasoner()
        chain = hr.reason("Prove commutativity of addition")
        assert len(chain.steps) >= 3
        assert chain.confidence > 0

    def test_check_property(self):
        from mnn.agi.reasoning import SymbolicReasoner
        sr = SymbolicReasoner()
        result = sr.check_property(lambda a, b: a + b, [1, 2, 3], "commutativity")
        assert result["holds"] is True


# ===== Module 4: Conjecture =====

class TestConjecture:
    def test_constant_detection(self):
        from mnn.agi.reasoning import ConjectureEngine
        ce = ConjectureEngine()
        obs = [{"input": i, "output": 5} for i in range(5)]
        conjs = ce.observe_and_conjecture(obs)
        assert len(conjs) >= 1
        assert "constant" in conjs[0].statement.lower()

    def test_linearity_detection(self):
        from mnn.agi.reasoning import ConjectureEngine
        ce = ConjectureEngine()
        obs = [{"input": i, "output": 2*i + 1} for i in range(5)]
        conjs = ce.observe_and_conjecture(obs)
        assert any("linear" in c.statement.lower() for c in conjs)


# ===== Module 5: Proof Strategy =====

class TestStrategy:
    def test_suggest(self):
        from mnn.agi.planner import ProofStrategyEngine
        pse = ProofStrategyEngine()
        strategies = pse.suggest_strategies(["algebra"])
        assert len(strategies) >= 1

    def test_update_rate(self):
        from mnn.agi.planner import ProofStrategyEngine
        pse = ProofStrategyEngine()
        old_rate = pse.strategies[0].success_rate
        pse.update_success_rate(pse.strategies[0].name, True)
        assert pse.strategies[0].success_rate >= old_rate


# ===== Module 6: Planner =====

class TestPlanner:
    def test_plan(self):
        from mnn.agi.planner import MathematicalPlanner
        mp = MathematicalPlanner()
        mp.set_goal("main_thm", "Prove X")
        mp.add_subgoal("lemma_a", "Prove A", "main_thm")
        mp.add_subgoal("lemma_b", "Prove B", "main_thm", ["lemma_a"])
        actions = mp.next_actions()
        names = [a.name for a in actions]
        assert "lemma_a" in names
        assert "lemma_b" not in names  # blocked by lemma_a

    def test_progress(self):
        from mnn.agi.planner import MathematicalPlanner
        mp = MathematicalPlanner()
        mp.set_goal("thm", "Prove")
        mp.add_subgoal("lem", "Lemma", "thm")
        mp.mark_complete("lem")
        prog = mp.progress()
        assert prog["completed"] == 1


# ===== Module 7: Explanation =====

class TestExplanation:
    def test_explain_group(self):
        from mnn.agi.explanation import ExplanationEngine, AudienceLevel
        ee = ExplanationEngine()
        expl = ee.explain("group", AudienceLevel.BEGINNER)
        assert "symmetry" in expl.summary.lower() or "rules" in expl.summary.lower()

    def test_format(self):
        from mnn.agi.explanation import ExplanationEngine, AudienceLevel
        ee = ExplanationEngine()
        expl = ee.explain("group", AudienceLevel.UNDERGRADUATE)
        text = ee.format_explanation(expl)
        assert "GROUP" in text


# ===== Module 9: Research =====

class TestResearch:
    def test_investigation(self):
        from mnn.agi.research import ResearchAssistant
        ra = ResearchAssistant()
        inv = ra.start_investigation("Heat Equation", "Study diffusion")
        ra.formulate_hypothesis("Solutions decay exponentially")
        ra.record_finding("Spectral analysis confirms decay")
        report = ra.complete_investigation()
        assert "Heat Equation" in report
        assert "decay" in report.lower()


# ===== Module 10: Dialogue =====

class TestDialogue:
    def test_define_and_recall(self):
        from mnn.agi.research import MathDialogueSystem
        ds = MathDialogueSystem()
        ds.user_input("define G as a finite group of order 12")
        response = ds.user_input("what is G")
        assert "finite group" in response

    def test_assumptions(self):
        from mnn.agi.research import MathDialogueSystem
        ds = MathDialogueSystem()
        ds.user_input("assume G is abelian")
        check = ds.consistency_check()
        assert check["consistent"]

    def test_transcript(self):
        from mnn.agi.research import MathDialogueSystem
        ds = MathDialogueSystem()
        ds.user_input("define X as a topological space")
        ds.user_input("what is X")
        t = ds.transcript()
        assert "USER" in t


# ===== Unified Assistant =====

class TestMathAGI:
    def test_initialize(self):
        from mnn.agi.assistant import MathAGIAssistant
        agi = MathAGIAssistant().initialize()
        status = agi.status()
        assert status["knowledge"] is not None
        assert status["memory"]["strategies"] >= 5

    def test_full_pipeline(self):
        from mnn.agi.assistant import MathAGIAssistant
        agi = MathAGIAssistant().initialize()
        # Learn
        agi.learn("my_thm", "theorem", "algebra", "All X are Y")
        # Reason
        chain = agi.reason("Prove my_thm")
        assert len(chain.steps) >= 3
        # Conjecture
        obs = [{"input": i, "output": i**2} for i in range(5)]
        conjs = agi.conjecture(obs)
        # Explain
        text = agi.explain("group", "beginner")
        assert len(text) > 0
        # Chat
        resp = agi.chat("define f as a homomorphism")
        assert "Defined" in resp


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

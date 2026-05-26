import unittest

from ts_benchmarks.baselines.graph_baselines import degree_baseline, pagerank_like_baseline
from ts_benchmarks.tasks.scaling import RelaxationConfig, generate_graph, run_relaxation


class ScalingTests(unittest.TestCase):
    def test_generate_graph_is_deterministic(self):
        left = generate_graph("scale_free", nodes=100, seed=42)
        right = generate_graph("scale_free", nodes=100, seed=42)
        self.assertEqual(left.values, right.values)
        self.assertEqual(left.edges, right.edges)
        self.assertEqual(left.spec.contradiction_pairs, right.spec.contradiction_pairs)

    def test_relaxation_emits_core_metrics(self):
        graph = generate_graph("small_world", nodes=100, seed=7)
        result = run_relaxation(graph, RelaxationConfig(steps=8))
        self.assertGreater(result.metrics["iterations"], 0)
        self.assertGreater(result.metrics["edges_relaxed"], 0)
        self.assertIn("final_global_tension", result.metrics)
        self.assertGreaterEqual(result.metrics["contradiction_localization_f1"], 0.0)
        self.assertLessEqual(result.metrics["contradiction_localization_f1"], 1.0)
        self.assertTrue(result.top_tension_nodes)

    def test_baselines_emit_localization_metrics(self):
        graph = generate_graph("random", nodes=100, seed=9)
        for baseline in [degree_baseline, pagerank_like_baseline]:
            metrics = baseline(graph)
            self.assertEqual(set(metrics), {"precision", "recall", "f1"})
            self.assertGreaterEqual(metrics["f1"], 0.0)
            self.assertLessEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

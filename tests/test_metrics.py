import unittest

from ml.evaluate import edit_distance, normalize, percentile, score


class MetricsTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Hello, WORLD!"), "hello world")

    def test_edit_distance(self):
        self.assertEqual(edit_distance(["i", "want", "coffee"], ["i", "need", "coffee"]), 1)

    def test_percentile(self):
        self.assertEqual(percentile([100, 200, 300], .5), 200)

    def test_score_and_gap(self):
        result = score([
            {"reference": "thank you", "prediction": "thank you", "condition": "clean", "latency_ms": 100},
            {"reference": "thank you", "prediction": "thank", "condition": "dim", "latency_ms": 200},
        ])
        self.assertEqual(result["wer"], .25)
        self.assertEqual(result["exact_match"], .5)
        self.assertEqual(result["conditions"]["clean"]["wer"], 0)
        self.assertEqual(result["conditions"]["dim"]["wer_gap_vs_clean"], .5)


if __name__ == "__main__":
    unittest.main()

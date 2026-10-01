import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from ml.prepare_manifest import build_manifest
from ml.evaluate import score

class DataTest(unittest.TestCase):
    def test_how2sign_tsv_preserves_general_topics_and_signer(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/"data.tsv", Path(folder)/"out.jsonl"
            source.write_text("SENTENCE_NAME\tSENTENCE\tSIGNER_ID\tSTART\tEND\nclip1\tThe train is late\tperson1\t1.0\t4.0\n")
            self.assertEqual(build_manifest(source,target,None,None),1)
            row = json.loads(target.read_text())
            self.assertEqual(row["text"],"The train is late")
            self.assertEqual(row["signer_id"],"person1")
            self.assertEqual(row["start"],"1.0")

    def test_in_place_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            build_manifest(Path("same.csv"),Path("same.csv"),None,None)

    def test_empty_reference_and_invalid_latency(self):
        for rows in [[],[{"reference":"","prediction":"hello"}], [{"reference":"hello","latency_ms":-1}], [{"reference":"hello","latency_ms":float('nan')}]]:
            with self.assertRaises(ValueError):
                score(rows)

    def test_evaluation_cannot_overwrite_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "predictions.jsonl"
            original = '{"reference":"hello","prediction":"hi"}\n'
            source.write_text(original)
            script = Path(__file__).resolve().parents[1] / "ml" / "evaluate.py"
            result = subprocess.run([sys.executable, str(script), "--input", str(source), "--output", str(source)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_text(), original)

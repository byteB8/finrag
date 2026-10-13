import json
import tempfile
import unittest
from pathlib import Path

from core import report


class Baselines(unittest.TestCase):
    def test_paper_baselines(self):
        with tempfile.TemporaryDirectory() as d:
            labels = ["Correct Answer", "Refusal", "Refusal", "Incorrect Answer"]
            Path(d, "gpt-4_sharedStore.jsonl").write_text("\n".join(json.dumps({"label": x}) for x in labels))
            Path(d, "gpt-4_closedBook.jsonl").write_text(json.dumps({"label": "Refusal"}))   # not a reported setting
            base = report.paper_baselines(Path(d))
        self.assertEqual(len(base), 1)
        self.assertEqual((base[0]["model"], base[0]["setting"], base[0]["n"]), ("gpt-4", "sharedStore", 4))
        self.assertEqual((base[0]["correct"], base[0]["refusal"]), (0.25, 0.5))

    def test_render_without_results(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(report.render(Path(d)).startswith("# Results"))


if __name__ == "__main__":
    unittest.main()

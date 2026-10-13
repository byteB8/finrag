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



class ExperimentSection(unittest.TestCase):
    def test_test_half_hidden_until_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "raw").mkdir()
            (r / "split.json").write_text(json.dumps({"dev": ["a", "b"], "test": ["c", "d"]}))
            exp = {"name": "reasoning", "baseline": {"gold": "gold"}}
            (r / "grade_gold_calc.json").write_text(json.dumps({"ctx": "gold", "label": "calc", "experiment": exp}))
            for name, verdicts in (("gold", ["incorrect", "correct", "incorrect", "incorrect"]),
                                   ("gold_calc", ["correct", "correct", "correct", "incorrect"])):
                (r / "raw" / f"grade_{name}.jsonl").write_text("\n".join(
                    json.dumps({"id": i, "verdict": v}) for i, v in zip("abcd", verdicts)))
                (r / "raw" / f"gen_{name}.jsonl").write_text("\n".join(
                    json.dumps({"id": i, "output_tokens": 100}) for i in "abcd"))
            rows, frozen = report.experiment(r)
            self.assertFalse(frozen)
            self.assertEqual(len(rows), 1)
            self.assertIn("dev (2)", rows[0])
            self.assertNotIn("test", "".join(rows))
            (r / "experiment_frozen.json").write_text("{}")
            rows, frozen = report.experiment(r)
            self.assertTrue(frozen)
            self.assertEqual([("dev" in x, "test" in x) for x in rows], [(True, False), (False, True)])


if __name__ == "__main__":
    unittest.main()

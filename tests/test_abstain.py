import unittest

from core import abstain


class Abstention(unittest.TestCase):
    def test_summary(self):
        answerable = [{"id": "a", "answerable": True, "answer": "USD 5 million"},
                      {"id": "b", "answerable": False, "answer": "The excerpts do not contain it."},
                      {"id": "c", "answerable": True, "answer": "USD 7 million"}]
        verdicts = {"a": "correct", "b": "incorrect", "c": "incorrect"}
        unans = [{"id": "u1", "answerable": False, "answer": "The excerpts do not contain FY2033 data."},
                 {"id": "u2", "answerable": True, "answer": "USD 9 million"}]
        s = abstain.summarize(answerable, verdicts, unans)["flag"]
        self.assertEqual(s["answerable"]["declined"], 1)
        self.assertAlmostEqual(s["answerable"]["coverage"], 2 / 3)
        self.assertAlmostEqual(s["answerable"]["correct"], 1 / 3)
        self.assertAlmostEqual(s["answerable"]["selective_accuracy"], 1 / 2)
        self.assertEqual((s["unanswerable"]["declined"], s["unanswerable"]["declined_rate"]), (1, 0.5))

    def test_text_rule_ignores_the_flag(self):
        rows = [{"id": "x", "answerable": False, "answer": "USD 3 million"}]
        s = abstain.summarize(rows, {"x": "correct"}, rows)
        self.assertEqual((s["flag"]["answerable"]["declined"], s["text"]["answerable"]["declined"]), (1, 0))


if __name__ == "__main__":
    unittest.main()

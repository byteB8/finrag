import json
import unittest

from core import review


class Sampling(unittest.TestCase):
    def test_sample_is_fixed_and_unique(self):
        ids = [f"q{i}" for i in range(150)]
        a, b = review.sample_ids(ids, 30), review.sample_ids(ids, 30)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 30)
        self.assertEqual(len(review.sample_ids(ids[:5], 30)), 5)


class Agreement(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(review.cohen_kappa(["correct", "incorrect"], ["correct", "incorrect"]), 1.0)
        # observed 0.5, chance 0.5 -> no agreement beyond chance
        self.assertAlmostEqual(review.cohen_kappa(["correct", "correct", "incorrect", "incorrect"],
                                                  ["correct", "incorrect", "correct", "incorrect"]), 0.0)

    def test_agreement_counts(self):
        human = {"a": "correct", "b": "incorrect", "c": "refusal", "d": "skip"}
        judge = {"a": "correct", "b": "correct", "c": "refusal"}
        r = review.agreement(human, judge)
        self.assertEqual((r["n"], r["agree"]), (3, 2))
        self.assertEqual(r["confusion"]["incorrect"]["correct"], 1)
        self.assertEqual(r["disagreements"], [{"id": "b", "human": "incorrect", "judge": "correct"}])
        self.assertAlmostEqual(r["correct_rate"]["judge"], 2 / 3)
        # b: human incorrect vs judge correct is a binary disagreement; c: refusal vs refusal agrees
        self.assertEqual(r["binary"]["agree"], 2)

    def test_binary_agreement_ignores_refusal_vs_incorrect(self):
        r = review.agreement({"a": "incorrect", "b": "correct"}, {"a": "refusal", "b": "correct"})
        self.assertEqual((r["agree"], r["binary"]["agree"], r["binary"]["kappa"]), (1, 2, 1.0))


class Sheet(unittest.TestCase):
    def test_embedded_data_cannot_close_the_script(self):
        item = {"id": "x", "question": "q</script><b>", "reference": "r", "justification": None,
                "answer": "a", "reasoning": "", "passages": [["h", "t"]]}
        page = review.render_sheet([item], "retrieved")
        self.assertEqual(page.count("</script>"), 2)        # the two real closing tags only
        start = page.index('type="application/json">') + len('type="application/json">')
        data = json.loads(page[start:page.index("</script>", start)].replace("<\\/", "</"))
        self.assertEqual(data["items"][0]["question"], "q</script><b>")


if __name__ == "__main__":
    unittest.main()

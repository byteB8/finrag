import unittest

from core import answer, grade, llm


class Prompt(unittest.TestCase):
    def test_passages_numbered_from_one(self):
        msgs = answer.build_messages("Q?", [("3M 10K 2018, page 60", "capex 1,577"), ("h2", "t2")])
        self.assertIn("[1] 3M 10K 2018, page 60\ncapex 1,577", msgs[1]["content"])
        self.assertIn("[2] h2\nt2", msgs[1]["content"])


class Checks(unittest.TestCase):
    passages = [("p1", "Capital expenditures were $1,577 million in 2018."), ("p2", "Revenue 32,765")]

    def test_numbers_normalised(self):
        self.assertEqual(answer.numbers("$1,577.00 million and 4.80% in 2018."), {"1577", "4.8", "2018"})

    def test_unparsed(self):
        self.assertFalse(answer.check(None, 2, self.passages)["parsed"])

    def test_grounded_answer(self):
        out = {"reasoning": "Capex was 1,577.", "answer": "USD 1,577 million in FY2018", "citations": [1], "answerable": True}
        c = answer.check(out, 2, self.passages)
        self.assertTrue(c["valid_citations"])
        self.assertEqual(c["ungrounded_numbers"], [])

    def test_out_of_range_citation_and_ungrounded_number(self):
        out = {"reasoning": "", "answer": "USD 1,600 million", "citations": [3], "answerable": True}
        c = answer.check(out, 2, self.passages)
        self.assertFalse(c["valid_citations"])
        self.assertEqual(c["ungrounded_numbers"], ["1600"])

    def test_no_citations_is_invalid(self):
        out = {"reasoning": "", "answer": "unknown", "citations": [], "answerable": False}
        self.assertFalse(answer.check(out, 2, self.passages)["valid_citations"])


class Judge(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(grade.verdict_of({"verdict": "correct"}), "correct")
        self.assertEqual(grade.verdict_of({"verdict": "maybe"}), "incorrect")
        self.assertEqual(grade.verdict_of(None), "incorrect")

    def test_parse_json(self):
        self.assertEqual(llm.parse_json('{"a": 1}'), {"a": 1})
        self.assertIsNone(llm.parse_json('{"a": '))
        self.assertIsNone(llm.parse_json("[1, 2]"))


if __name__ == "__main__":
    unittest.main()

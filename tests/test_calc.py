import unittest

from core import calc


class SafeEval(unittest.TestCase):
    def test_arithmetic_and_names(self):
        names = {"ap_2016": 25309, "ap_2017": 34616, "cogs": 111934, "inv_2016": 11461, "inv_2017": 16047}
        dpo = calc.safe_eval("365 * ((ap_2016 + ap_2017) / 2) / (cogs + (inv_2017 - inv_2016))", names)
        self.assertAlmostEqual(dpo, 365 * 29962.5 / 116520)
        self.assertAlmostEqual(calc.safe_eval("(65984 / 65398) ** (1 / 2) - 1", {}), (65984 / 65398) ** 0.5 - 1)
        self.assertEqual(calc.safe_eval("-abs(-2) + max(1, 3) + round(2.6)", {}), 4.0)
        # the model writes round(x, 2) for "round to two decimal places"; a float digit count used to raise
        self.assertEqual(calc.safe_eval("round(cfo / cl, 2)", {"cfo": 1283, "cl": 2000}), 0.64)

    def test_rejects_code_and_unknown_names(self):
        for bad in ("__import__('os')", "open('x')", "x.y", "[1, 2]", "2 ** 1000", "lambda: 1", "1 if 1 else 2"):
            with self.assertRaises((ValueError, SyntaxError)):
                calc.safe_eval(bad, {})
        with self.assertRaises(ValueError):
            calc.safe_eval("missing + 1", {})

    def test_run_collects_errors_per_calculation(self):
        plan = {"figures": [{"name": "a", "value": 10, "excerpt": 1}, {"name": "b", "value": 0, "excerpt": 2}],
                "calculations": [{"label": "ratio", "expression": "a / 4"}, {"label": "bad", "expression": "a / b"}]}
        out = calc.run(plan)
        self.assertEqual(out[0]["value"], 2.5)
        self.assertIn("error", out[1])
        block = calc.results_block(plan, out)
        self.assertIn("ratio: a / 4 = 2.5", block)
        self.assertIn("could not be computed", block)

    def test_later_calculations_use_earlier_labels(self):
        plan = {"figures": [{"name": "oi_22", "value": 6098, "excerpt": 1}, {"name": "rev_22", "value": 17606, "excerpt": 1},
                            {"name": "oi_21", "value": 5802, "excerpt": 1}, {"name": "rev_21", "value": 15785, "excerpt": 1}],
                "calculations": [{"label": "margin_22", "expression": "oi_22 / rev_22"},
                                 {"label": "margin_21", "expression": "oi_21 / rev_21"},
                                 {"label": "change", "expression": "margin_22 - margin_21"}]}
        out = calc.run(plan)
        self.assertAlmostEqual(out[2]["value"], 6098 / 17606 - 5802 / 15785)

    def test_no_calculation_means_no_block(self):
        self.assertEqual(calc.results_block({"figures": [], "calculations": []}, calc.run({"figures": [], "calculations": []})), "")
        self.assertEqual(calc.run(None), [])


if __name__ == "__main__":
    unittest.main()

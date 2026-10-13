import unittest

import run
from core import report


class Config(unittest.TestCase):
    def test_merge_overrides_only_what_differs(self):
        base = {"gen": {"model": "m", "prompt": "v1"}, "search": {"k": [1, 5]}}
        out = run.merge(base, {"tag": "_v2", "gen": {"prompt": "v2"}})
        self.assertEqual(out, {"gen": {"model": "m", "prompt": "v2"}, "search": {"k": [1, 5]}, "tag": "_v2"})
        self.assertEqual(base["gen"]["prompt"], "v1")      # base left untouched

    def test_v2_config_inherits_mvp(self):
        cfg = run.load_cfg(run.env.BASE / "configs/v2.toml")
        self.assertEqual((cfg["tag"], cfg["gen"]["prompt"], cfg["gen"]["model"]), ("_v2", "v2", "Qwen/Qwen3.5-4B"))

    def test_run_labels(self):
        self.assertEqual(report.run_label("retrieved"), "retrieved chunks, prompt v1")
        self.assertEqual(report.run_label("gold_v2"), "labelled evidence pages (oracle), prompt v2")
        self.assertEqual(report.run_label("retrieved_prefix"), "retrieved chunks, prompt v1, before the decoding-loop fixes")


if __name__ == "__main__":
    unittest.main()

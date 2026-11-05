"""Regression checks for the public CLI after separating the pipeline stages."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from grounded_qa import __main__ as cli
from grounded_qa.data.io import read_rows, write_rows
from grounded_qa.inference.boundaries import complete_boundary
from grounded_qa.inference.generate import DECODING_CONFIG

ROOT = Path(__file__).resolve().parents[1]


class PipelineLayoutTest(unittest.TestCase):
    def test_all_commands_load_without_the_model(self):
        for name in ("prompt", "trace", "pairs", "sft", "simpo", "intervene", "predict", "judge", "metrics"):
            with self.subTest(command=name):
                output = subprocess.check_output(
                    [sys.executable, "-B", "-m", "grounded_qa", name, "--help"], cwd=ROOT, text=True
                )
                self.assertIn("--input", output)
                self.assertIn("--out", output)

    def test_training_traces_and_generation_dispatch_to_their_own_modules(self):
        stages = {"sft": "training.trainer.train", "simpo": "training.trainer.train",
                  "trace": "feedback.traces.trace", "predict": "inference.generate.predict"}
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "in.jsonl", Path(temp) / "out.jsonl"
            write_rows(source, [{"id": "example"}])
            for name, handler in stages.items():
                args = ["grounded_qa", name, "--input", str(source), "--out", str(target), "--model", "local"]
                if name in ("simpo", "sft"):
                    args += ["--lr", "0.000003"]
                with self.subTest(command=name), patch.object(sys, "argv", args), \
                        patch("grounded_qa." + handler, return_value=[{"id": "prediction"}]) as stub:
                    cli.main()
                    parsed, rows = stub.call_args.args
                    self.assertEqual(parsed.command, name)
                    self.assertEqual(rows, [{"id": "example"}])
                    if name in ("trace", "predict"):
                        self.assertEqual(read_rows(target), [{"id": "prediction"}])

    def test_decoding_policy_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = Path.cwd()
            try:
                os.chdir(temp)
                cfg = json.loads(DECODING_CONFIG.read_text(encoding="utf-8"))
            finally:
                os.chdir(previous)
        block = "<statement>Result<cite>[0-0]</cite></statement>"
        self.assertTrue(complete_boundary(block, 12, False, cfg))
        self.assertFalse(complete_boundary(block, 12, True, cfg))


if __name__ == "__main__":
    unittest.main()

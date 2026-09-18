"""CPU-only regression checks for inference lifetime and allocator policy.

Execute the real loop with lightweight dependencies; no model weights or CUDA
are needed. These checks do not claim numerical or GPU performance validation.
"""
import ast
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[1]


def source_tree(relative):
    return ast.parse((ROOT / relative).read_text())


def compile_nodes(nodes, namespace):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<production-source>", "exec"), namespace)


class PredictionToken:
    pass


class MemoryLifetimeTests(unittest.TestCase):
    def run_loop(self, fail_dump=False):
        function = next(n for n in source_tree("runner/inference.py").body
                        if isinstance(n, ast.FunctionDef) and n.name == "infer_predict")
        refs, saved, seeded = [], [], []
        calls = 0

        def predict(data):
            nonlocal calls
            self.assertTrue(all(ref() is None for ref in refs), "previous prediction survives into next forward")
            calls += 1
            token = PredictionToken()
            refs.append(weakref.ref(token))
            return {"coordinate": token, "summary_confidence": [{"ranking_score": calls}]}

        def dump(**kwargs):
            prediction = kwargs["pred_dict"]
            self.assertIs(prediction["coordinate"], refs[-1]())
            if fail_dump and calls == 1:
                raise OSError("simulated output failure")
            saved.append((kwargs["seed"], kwargs["pdb_id"], prediction["summary_confidence"][0]["ranking_score"]))

        def empty_cache():
            if not fail_dump:
                self.assertTrue(all(ref() is None for ref in refs), "outputs must be released before cache flush")

        class Loader:
            dataset = [0, 1]

            def __iter__(self):
                for index in self.dataset:
                    scalar = SimpleNamespace(item=lambda: 4)
                    data = dict(sample_name=f"item{index}", sample_index=index,
                                N_asym=scalar, N_token=scalar, N_atom=scalar,
                                N_msa=scalar, entity_poly_type={"A": "protein"})
                    yield [(data, object(), "")]

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(json.dumps([{"name": "test"}]))
            errors = Path(directory) / "ERR"
            errors.mkdir()
            config = SimpleNamespace(input_json_path=str(input_path), use_seeds_in_json=False,
                                     seeds=[101, 102], deterministic=True, dump_dir=directory)
            logger = logging.Logger("memory-test")
            logger.addHandler(logging.NullHandler())
            ns = dict(logger=logger, json=json, os=os, time=time, traceback=traceback,
                      opjoin=os.path.join, opexists=os.path.exists,
                      DIST_WRAPPER=SimpleNamespace(rank=0),
                      get_inference_dataloader=lambda configs: Loader(),
                      seed_everything=lambda seed, deterministic: seeded.append(seed),
                      update_inference_configs=lambda configs, n: configs,
                      torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=empty_cache)))
            compile_nodes([function], ns)
            runner = SimpleNamespace(error_dir=str(errors), predict=predict,
                                     update_model_configs=lambda configs: None,
                                     dumper=SimpleNamespace(dump=dump))
            ns["infer_predict"](runner, config)
            self.assertEqual(seeded, [101, 102])
            expected = [(101, "item0", 1), (101, "item1", 2), (102, "item0", 3), (102, "item1", 4)]
            self.assertEqual(saved, expected[1:] if fail_dump else expected)
            self.assertTrue(all(ref() is None for ref in refs))
            if fail_dump:
                self.assertIn("simulated output failure", (errors / "item0.txt").read_text())

    def test_outputs_saved_then_released_across_items_and_seeds(self):
        self.run_loop()

    def test_dump_failure_releases_prediction_before_next_item(self):
        self.run_loop(fail_dump=True)

    def test_confidence_entry_allocator_policy(self):
        # Execute the actual cleanup block at the z_init lifetime boundary.
        tree = source_tree("protenix/model/modules/confidence.py")
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and any(isinstance(child, ast.Delete) and any(
                         isinstance(t, ast.Name) and t.id == "z_init" for t in child.targets)
                         for child in n.body))
        for training, tokens, expected in [(False, 400, 0), (False, 2000, 0),
                                           (False, 2001, 1), (True, 2001, 0)]:
            with self.subTest(training=training, tokens=tokens):
                flushed = []
                ns = dict(self=SimpleNamespace(training=training), z_init=object(),
                          z_trunk=SimpleNamespace(shape=(tokens, tokens, 128)),
                          torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: flushed.append(1))))
                compile_nodes([block], ns)
                self.assertEqual(len(flushed), expected)
                self.assertEqual("z_init" in ns, training)


if __name__ == "__main__":
    unittest.main()

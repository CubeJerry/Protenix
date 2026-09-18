"""CPU regressions for opt-in inference work reuse; no model weights required."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

os.environ.setdefault("LAYERNORM_TYPE", "torch")

import torch
from torch import nn

from protenix.model.inference_optimization import (
    checkpoint_initialization,
    enabled,
    skipping_random_init,
)
from protenix.model.modules.pairformer import MSAStack, TemplateEmbedder
from protenix.model.modules.primitives import Linear
from protenix.model.triangular.layers import OpenfoldLinear


def randomize(module):
    # Avoid equality tests passing trivially because output projections are zero.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.2)
    return module


class WorkReuseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_msa_reuse_is_exact_and_call_local(self):
        module = randomize(MSAStack(c_m=4, c_z=8, c=2, dropout=0)).eval()
        m = torch.randn(7, 3, 4)  # Partial final chunk.
        for z in (torch.randn(3, 3, 8), torch.randn(3, 3, 8)):
            results = []
            for flag, expected_calls in (("0", 3), ("1", 1)):
                projection = module.msa_pair_weighted_averaging.linear_no_bias_z
                with patch.dict(
                    os.environ, PROTENIX_REUSE_MSA_PAIR_WEIGHTS=flag
                ), patch.object(
                    projection, "forward", wraps=projection.forward
                ) as calls, torch.no_grad():
                    results.append(module.inference_forward(m.clone(), z, chunk_size=3))
                    self.assertEqual(calls.call_count, expected_calls)
            self.assertTrue(torch.equal(*results))
            self.assertFalse(torch.equal(results[0], m))

    def test_msa_single_chunk_and_training_are_not_reused(self):
        module = MSAStack(c_m=4, c_z=8, c=2, dropout=0)
        projection = module.msa_pair_weighted_averaging.linear_no_bias_z
        with patch.dict(os.environ, PROTENIX_REUSE_MSA_PAIR_WEIGHTS="1"):
            for training, depth, expected in ((False, 2, 1), (True, 7, 3)):
                module.train(training)
                with torch.no_grad(), patch.object(
                    projection, "forward", wraps=projection.forward
                ) as calls:
                    module.inference_forward(
                        torch.randn(depth, 3, 4), torch.randn(3, 3, 8), 3
                    )
                self.assertEqual(calls.call_count, expected)

    def template_features(self):
        features = {"asym_id": torch.tensor([1, 1, 2])}
        for key, shape in {
            "template_aatype": (3,),
            "template_distogram": (3, 3, 39),
            "template_pseudo_beta_mask": (3, 3),
            "template_unit_vector": (3, 3, 3),
            "template_backbone_frame_mask": (3, 3),
            "template_pair_geometry_mask": (3, 3),
        }.items():
            value = (
                torch.randint(0, 20, shape)
                if key == "template_aatype"
                else torch.rand(shape)
            )
            features[key] = torch.stack([value.clone() for _ in range(3)])
        return features

    def test_templates_preserve_order_masks_and_call_scope(self):
        module = randomize(TemplateEmbedder(n_blocks=1, c=16, c_z=8, dropout=0)).eval()
        for differing_key in (None, *self.template_features().keys()):
            if differing_key == "asym_id":
                continue
            features = self.template_features()
            if differing_key:
                features[differing_key][1] += 1
            # Two separate pair representations must each get new calculations.
            for z in (torch.randn(3, 3, 8), torch.randn(3, 3, 8)):
                results = []
                for flag in ("0", "1"):
                    with patch.dict(
                        os.environ, PROTENIX_REUSE_TEMPLATES=flag
                    ), patch.object(
                        module,
                        "single_template_forward",
                        wraps=module.single_template_forward,
                    ) as calls, torch.no_grad():
                        results.append(module(features, z.clone()))
                        self.assertEqual(
                            calls.call_count,
                            3 if flag == "0" else (2 if differing_key else 1),
                        )
                self.assertTrue(torch.equal(*results), differing_key)
                self.assertGreater(results[0].abs().sum().item(), 0)

    def test_template_training_and_grad_paths_do_not_reuse(self):
        module = TemplateEmbedder(n_blocks=1, c=16, c_z=8, dropout=0)
        features = self.template_features()
        with patch.dict(os.environ, PROTENIX_REUSE_TEMPLATES="1"):
            for training, gradients in ((True, False), (False, True)):
                module.train(training)
                with torch.set_grad_enabled(gradients), patch.object(
                    module,
                    "single_template_forward",
                    wraps=module.single_template_forward,
                ) as calls:
                    module(features, torch.randn(3, 3, 8))
                self.assertEqual(calls.call_count, 3)

    def test_checkpoint_restores_linear_parameters_and_constants(self):
        def model():
            result = nn.Sequential(
                Linear(4, 5), OpenfoldLinear(5, 3), Linear(3, 2, initializer="zeros")
            )
            result.register_buffer("nonpersistent", torch.ones(3), persistent=False)
            return result

        original = model()
        checkpoint = original.state_dict()
        with patch(
            "torch.nn.init.kaiming_uniform_", side_effect=AssertionError("random init")
        ), patch(
            "protenix.model.triangular.layers.truncnorm.rvs",
            side_effect=AssertionError("random init"),
        ), checkpoint_initialization(
            skip=True, strict=True
        ):
            restored = model()
        self.assertTrue(
            torch.equal(restored[2].weight, torch.zeros_like(restored[2].weight))
        )
        restored.load_state_dict(checkpoint, strict=True)
        for key, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, checkpoint[key]), key)
        self.assertTrue(torch.equal(original.nonpersistent, restored.nonpersistent))
        x = torch.randn(2, 4)
        self.assertTrue(torch.equal(original(x), restored(x)))
        with self.assertRaises(RuntimeError):
            restored.load_state_dict({}, strict=True)

    def test_initialization_scope_and_strict_guard(self):
        with self.assertRaises(ValueError), checkpoint_initialization(True, False):
            self.fail("Non-strict load accepted")
        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            with checkpoint_initialization(True, True):
                self.assertTrue(skipping_random_init())
                with ThreadPoolExecutor(1) as pool:
                    self.assertFalse(pool.submit(skipping_random_init).result())
                raise RuntimeError("construction failed")
        self.assertFalse(skipping_random_init())
        with patch.dict(os.environ, PROTENIX_REUSE_TEMPLATES="yes"), self.assertRaises(
            ValueError
        ):
            enabled("PROTENIX_REUSE_TEMPLATES")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the Wang-DNN baseline and the physics_baseline predictor.

These tests focus on the new pieces added in `src/ofc_ml/models/wang_dnn.py`
and `src/ofc_ml/models/physics_baseline.py`, including their integration with
`build_model` and the YAML configs under `experiments/main/`.

They intentionally avoid disk / training side effects: the fitting loop is
exercised via separate smoke runs of `scripts/run_experiment.py`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ofc_ml.configs import load_experiment_config  # noqa: E402
from ofc_ml.configs.schema import ModelConfig  # noqa: E402
from ofc_ml.models import (  # noqa: E402
    PhysicsBaselinePredictor,
    WangDNNPredictor,
    build_model,
    count_parameters,
)


class TestWangDNNPredictor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.input_dim = 204
        self.output_dim = 95
        self.batch = 8

    def test_forward_shape_and_mask(self):
        model = WangDNNPredictor(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[256, 128, 128, 128],
            dropout=0.0,
        )
        x = torch.randn(self.batch, self.input_dim)
        mask = torch.zeros(self.batch, self.output_dim)
        mask[:, :40] = 1.0  # only first 40 channels active

        out = model(x, mask)
        self.assertEqual(out.shape, (self.batch, self.output_dim))
        self.assertTrue(torch.all(out[:, 40:] == 0.0))

    def test_kaiming_init_zero_bias(self):
        model = WangDNNPredictor(input_dim=10, output_dim=5, hidden_dims=[8, 4])
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                self.assertTrue(torch.all(module.bias == 0.0))

    def test_layer_pattern(self):
        model = WangDNNPredictor(input_dim=10, output_dim=5, hidden_dims=[8, 4])
        types = [type(m).__name__ for m in model.body]
        # Linear -> BatchNorm1d -> ELU repeated for each hidden layer.
        self.assertEqual(types, ["Linear", "BatchNorm1d", "ELU"] * 2)
        self.assertIsInstance(model.head, torch.nn.Linear)

    def test_dropout_is_inserted_when_requested(self):
        model = WangDNNPredictor(input_dim=10, output_dim=5, hidden_dims=[8], dropout=0.3)
        types = [type(m).__name__ for m in model.body]
        self.assertEqual(types, ["Linear", "BatchNorm1d", "ELU", "Dropout"])

    def test_build_model_dispatch(self):
        cfg = ModelConfig(name="wang_dnn", hidden_dims=[256, 128, 128, 128], dropout=0.0)
        model = build_model(cfg, input_dim=self.input_dim, output_dim=self.output_dim)
        self.assertIsInstance(model, WangDNNPredictor)
        n = count_parameters(model)
        self.assertGreater(n, 0)


class TestPhysicsBaselinePredictor(unittest.TestCase):
    def test_forward_returns_zeros(self):
        model = PhysicsBaselinePredictor(output_dim=95)
        x = torch.randn(4, 200)
        out = model(x)
        self.assertEqual(out.shape, (4, 95))
        self.assertTrue(torch.all(out == 0.0))

    def test_forward_with_mask(self):
        model = PhysicsBaselinePredictor(output_dim=95)
        x = torch.randn(4, 200)
        mask = torch.zeros(4, 95)
        mask[:, :10] = 1.0
        out = model(x, mask)
        self.assertEqual(out.shape, (4, 95))
        self.assertTrue(torch.all(out == 0.0))

    def test_no_trainable_params(self):
        model = PhysicsBaselinePredictor(output_dim=95)
        self.assertEqual(count_parameters(model), 0)

    def test_build_model_dispatch(self):
        cfg = ModelConfig(name="physics_baseline")
        model = build_model(cfg, input_dim=204, output_dim=95)
        self.assertIsInstance(model, PhysicsBaselinePredictor)


class TestPredictTestPathPhysicsBaseline(unittest.TestCase):
    """Exercise the same call sequence used by `scripts/run_experiment.py` for
    `physics_baseline`, without touching disk or running any training."""

    def test_baseline_recovered_from_zero_pred(self):
        from ofc_ml.network import compute_baseline_gain

        device = torch.device("cpu")
        model = PhysicsBaselinePredictor(output_dim=95).to(device)
        model.eval()

        target_gain = np.array([15.0, 18.0, 21.0])
        target_gain_tilt = np.array([0.0, -1.0, 1.0])
        mask = np.ones((3, 95), dtype=np.float32)
        mask[0, 50:] = 0.0  # row 0 has only first 50 channels active.

        baseline = compute_baseline_gain(target_gain, target_gain_tilt)
        x = torch.zeros(3, 10, device=device)
        with torch.no_grad():
            pred = model(x, torch.as_tensor(mask, device=device)).cpu().numpy()

        out = (baseline + pred) * mask
        # First row: only channels 0..49 are non-zero and equal the baseline.
        np.testing.assert_allclose(out[0, :50], baseline[0, :50])
        np.testing.assert_array_equal(out[0, 50:], np.zeros(45))
        # Other rows: full baseline because mask is all ones.
        np.testing.assert_allclose(out[1], baseline[1])
        np.testing.assert_allclose(out[2], baseline[2])


class TestExperimentYamls(unittest.TestCase):
    def test_wang_dnn_tl_loads(self):
        cfg = load_experiment_config("experiments/main/wang_dnn_tl.yaml")
        self.assertEqual(cfg.name, "wang_dnn_tl")
        self.assertEqual(cfg.model.name, "wang_dnn")
        self.assertEqual(list(cfg.model.hidden_dims), [256, 128, 128, 128])
        self.assertTrue(cfg.model.predict_absolute)
        self.assertEqual(list(cfg.stages), ["pretrain", "finetune"])
        self.assertEqual(cfg.pretrain.grad_clip, 3.0)
        self.assertEqual(cfg.finetune.grad_clip, 3.0)

    def test_physics_baseline_loads_with_empty_stages(self):
        cfg = load_experiment_config("experiments/main/physics_baseline.yaml")
        self.assertEqual(cfg.name, "physics_baseline")
        self.assertEqual(cfg.model.name, "physics_baseline")
        self.assertFalse(cfg.model.predict_absolute)
        self.assertEqual(list(cfg.stages), [])

    def test_existing_main_configs_unaffected(self):
        cfg = load_experiment_config("experiments/main/m1_ours.yaml")
        self.assertEqual(cfg.name, "m1_ours")
        self.assertEqual(cfg.model.name, "hybrid_fno_kan")
        self.assertFalse(cfg.model.predict_absolute)


class TestRunMatrixOrdering(unittest.TestCase):
    def test_ordered_experiments_includes_new_yamls(self):
        # Import here so that test runs even if scripts/ isn't already on sys.path.
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import run_matrix  # type: ignore
        finally:
            sys.path.pop(0)
        names = [Path(p).stem for p in run_matrix.ORDERED_EXPERIMENTS]
        self.assertIn("physics_baseline", names)
        self.assertIn("wang_dnn_tl", names)
        self.assertIn("m1_ours", names)
        # physics_baseline comes first because it has no training stage and
        # m1_ours runs before any architecture baseline that re-uses the cache.
        self.assertEqual(names[0], "physics_baseline")
        self.assertLess(names.index("m1_ours"), names.index("m2_mlp"))
        self.assertLess(names.index("m1_ours"), names.index("wang_dnn_tl"))


class TestAggregateGroups(unittest.TestCase):
    def test_main_arch_transfer_physics_groups(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import aggregate_results  # type: ignore
        finally:
            sys.path.pop(0)
        self.assertEqual(
            aggregate_results.MAIN_EXPS,
            ["physics_baseline", "wang_dnn_tl", "m1_ours"],
        )
        self.assertEqual(
            aggregate_results.ARCH_EXPS,
            ["m1_ours", "m2_mlp", "m3_cnn1d", "m4_transformer"],
        )
        self.assertEqual(
            aggregate_results.TRANSFER_EXPS,
            ["m1_ours", "a_t1_no_finetune", "a_t2_no_pretrain", "a_t3_joint"],
        )
        self.assertEqual(
            aggregate_results.PHYSICS_EXPS,
            ["m1_ours", "a_p1_predict_absolute"],
        )


if __name__ == "__main__":
    unittest.main()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from tabfm.src.mlx import lora
from tabfm.src.mlx import model as mlx_model


def _small_model(is_classifier=True):
  return mlx_model.TabFM(
      embed_dim=8,
      max_classes=3,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=4,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=2,
      icl_num_blocks=2,
      icl_nhead=2,
      ff_factor=2,
      feature_group_size=2,
      is_classifier=is_classifier,
  )


def _toy_data(n=96, h=4, seed=0):
  rng = np.random.default_rng(seed)
  X = rng.normal(size=(n, h)).astype(np.float32)
  y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
  return X, y


class LoRATest(unittest.TestCase):

  def test_apply_lora_freezes_base(self):
    """Only adapter matrices are trainable; base weights never change."""
    model = _small_model()
    replaced = lora.apply_lora(model, rank=4, scope="icl")
    self.assertGreater(replaced, 0)

    trainable = dict(tree_flatten(model.trainable_parameters()))
    self.assertTrue(trainable)
    for key in trainable:
      self.assertTrue(
          key.endswith("lora_a") or key.endswith("lora_b"),
          f"unexpected trainable parameter: {key}",
      )

    base_before = {
        k: np.array(v)
        for k, v in tree_flatten(model.parameters())
        if not (k.endswith("lora_a") or k.endswith("lora_b"))
    }
    X, y = _toy_data()
    lora.train_lora(model, X, y, steps=5, batch_rows=32, seed=0)
    for k, v in tree_flatten(model.parameters()):
      if k in base_before:
        np.testing.assert_array_equal(
            np.array(v), base_before[k], err_msg=f"base weight changed: {k}"
        )

  def test_train_lora_reduces_loss(self):
    """Adapter training reduces the loss on a trivially learnable task.

    A random-init tiny base has no in-context ability, so a *hard* task would
    make this test flaky; constant labels are learnable through the adapters
    alone and make loss descent a pure optimization check.
    """
    model = _small_model()
    lora.apply_lora(model, rank=8, scope="icl")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 4)).astype(np.float32)
    y = np.zeros(96, dtype=np.int64)
    losses = lora.train_lora(
        model, X, y, steps=80, batch_rows=48, learning_rate=1e-2,
        num_classes=2, seed=0,
    )
    head = float(np.mean(losses[:10]))
    tail = float(np.mean(losses[-10:]))
    self.assertLess(tail, head, f"loss did not decrease: {head} -> {tail}")

  def test_merge_lora_matches_unmerged(self):
    """Folding adapters into the base weights preserves outputs."""
    model = _small_model()
    lora.apply_lora(model, rank=4, scope="icl")
    X, y = _toy_data()
    lora.train_lora(model, X, y, steps=10, batch_rows=32, seed=0)

    x = mx.array(np.random.default_rng(1).normal(size=(2, 6, 4)).astype(np.float32))
    y_in = mx.array(np.random.default_rng(2).integers(0, 3, size=(2, 6)))
    ts = mx.array(np.array([4, 4], dtype=np.int32))
    before = np.array(model(x, y_in, ts))

    merged = lora.merge_lora(model)
    self.assertGreater(merged, 0)
    for k, _ in tree_flatten(model.parameters()):
      self.assertNotIn("lora_", k)
    after = np.array(model(x, y_in, ts))
    np.testing.assert_allclose(before, after, atol=1e-4)

  def test_save_load_adapters_roundtrip(self):
    """Adapters saved from one model reproduce outputs in a fresh model."""
    model = _small_model()
    # Snapshot the base tree BEFORE adapting: same structure as a fresh model,
    # and the base weights stay frozen (unchanged) through training.
    base_params = model.parameters()
    lora.apply_lora(model, rank=4, scope="icl")
    X, y = _toy_data()
    lora.train_lora(model, X, y, steps=5, batch_rows=32, seed=0)

    x = mx.array(
        np.random.default_rng(1).normal(size=(1, 6, 4)).astype(np.float32)
    )
    y_in = mx.array(np.random.default_rng(2).integers(0, 3, size=(1, 6)))
    ts = mx.array(np.array([4], dtype=np.int32))
    want = np.array(model(x, y_in, ts))

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "adapters.safetensors")
      lora.save_adapters(model, path)

      fresh = _small_model()
      fresh.update(base_params)  # same base weights as `model`
      lora.apply_lora(fresh, rank=4, scope="icl")
      lora.load_adapters(fresh, path)
      got = np.array(fresh(x, y_in, ts))

    np.testing.assert_allclose(want, got, atol=1e-5)

  def test_fit_lora_uses_wrapper_preprocessing(self):
    """fit_lora trains on the wrapper's preprocessed matrix, then predicts."""
    from tabfm.src.classifier_and_regressor import TabFMClassifier

    model = _small_model()
    lora.apply_lora(model, rank=4, scope="icl")
    clf = TabFMClassifier(
        model=model, n_estimators=2, batch_size=2, random_state=42
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 3))
    y = rng.integers(0, 3, size=24)

    losses = lora.fit_lora(clf, X, y, steps=4, batch_rows=16, seed=0)
    self.assertEqual(len(losses), 4)
    self.assertTrue(np.isfinite(losses).all())

    # The adapted model still flows through the sklearn wrapper end to end.
    preds = clf.predict(X)
    self.assertEqual(preds.shape, (24,))

  def test_regression_train_smoke(self):
    """Regression models train through the MSE path."""
    model = _small_model(is_classifier=False)
    lora.apply_lora(model, rank=2, scope="icl")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1]).astype(np.float32)
    losses = lora.train_lora(model, X, y, steps=5, batch_rows=32, seed=0)
    self.assertEqual(len(losses), 5)
    self.assertTrue(np.isfinite(losses).all())


if __name__ == "__main__":
  unittest.main()

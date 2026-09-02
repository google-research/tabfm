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

import pickle
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch

from tabfm.src.pytorch import model as pytorch_model
from tabfm.src.classifier_and_regressor import TabFMClassifier, TabFMRegressor


class PyTorchClassifierRegressorTest(unittest.TestCase):

  def test_classifier_fit_predict(self):
    np.random.seed(42)
    # Instantiate small PyTorch model
    model = pytorch_model.TabFM(
        embed_dim=8,
        max_classes=3,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=True
    )
    
    clf = TabFMClassifier(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42
    )

    X = np.random.rand(10, 3)
    y = np.random.randint(0, 3, size=10)

    clf.fit(X, y)
    
    # Predict
    preds = clf.predict(X)
    self.assertEqual(preds.shape, (10,))
    self.assertTrue(np.all(preds >= 0) and np.all(preds < 3))

    # Predict proba
    probs = clf.predict_proba(X)
    self.assertEqual(probs.shape, (10, 3))
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)

    # cache_context=True should be a pure speedup: predict_proba should match
    # the uncached path exactly. maybe_quantize_kv_cache=False isolates this
    # from the (lossy by design) int8 KV-cache quantization, which is tested
    # separately in model_test.py.
    cached_clf = TabFMClassifier(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )
    cached_clf.fit(X, y)
    probs_cached = cached_clf.predict_proba(X)
    self.assertEqual(probs_cached.shape, (10, 3))
    np.testing.assert_allclose(np.sum(probs_cached, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(probs_cached, probs, rtol=1e-5, atol=1e-6)

  def test_regressor_fit_predict(self):
    np.random.seed(42)
    # Instantiate small PyTorch model
    model = pytorch_model.TabFM(
        embed_dim=8,
        max_classes=1, # Regressor might ignore this or use 1
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=False
    )
    
    reg = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    reg.fit(X, y)

    # Predict
    preds = reg.predict(X)
    self.assertEqual(preds.shape, (10,))

    # cache_context=True should be a pure speedup: predict should match the
    # uncached path exactly. maybe_quantize_kv_cache=False isolates this from
    # the (lossy by design) int8 KV-cache quantization, which is tested
    # separately in model_test.py.
    cached_reg = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )
    cached_reg.fit(X, y)
    preds_cached = cached_reg.predict(X)
    self.assertEqual(preds_cached.shape, (10,))
    np.testing.assert_allclose(preds_cached, preds, rtol=1e-5, atol=1e-6)

  def test_regressor_batch_forward_float64_targets_on_mps(self):
    if not torch.backends.mps.is_available():
      self.skipTest("MPS is required for this test.")

    class _Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), device="mps"))

      def forward(self, X, y, train_size, cat_mask=None, d=None):
        del X, train_size, cat_mask, d
        return y.unsqueeze(-1) + self.anchor

    reg = TabFMRegressor(model=_Model(), batch_size=1)
    X = np.zeros((1, 2, 1), dtype=np.float32)
    y = np.zeros((1, 1), dtype=np.float64)

    preds = reg._batch_forward(X, y)

    self.assertEqual(preds.dtype, np.float32)
    np.testing.assert_array_equal(preds, [[[-100.0]]])

  def test_regressor_context_cache_float64_targets_on_mps(self):
    if not torch.backends.mps.is_available():
      self.skipTest("MPS is required for this test.")

    np.random.seed(42)
    model = pytorch_model.TabFM(
        embed_dim=8,
        max_classes=1,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=False,
    ).to("mps")

    reg = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )

    X = np.random.rand(10, 3)
    # numpy's default float dtype is float64, which MPS cannot hold.
    y = np.random.rand(10)

    # The context prefill runs inside fit(), so the dtype crash lands here.
    reg.fit(X, y)

    preds = reg.predict(X)
    self.assertEqual(preds.shape, (10,))
    self.assertTrue(np.all(np.isfinite(preds)))


class PyTorchModelPickleTest(unittest.TestCase):
  """The PyTorch model must be picklable.

  AutoGluon / TabArena save the fitted estimator (which holds the model) with
  stdlib pickle. The encode/decode heads are gelu ``MLP``s whose activation was
  previously a lambda from ``get_activation`` -- unpicklable -- so this guards
  against that regression. The regression path builds three such heads
  (y_embedder, y_encoder, decoder), so it exercises the activation most.
  """

  def _tiny_model(self, is_classifier):
    return pytorch_model.TabFM(
        embed_dim=8,
        max_classes=3,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=2,
        is_classifier=is_classifier,
    )

  def test_classifier_model_pickle_round_trip(self):
    restored = pickle.loads(pickle.dumps(self._tiny_model(is_classifier=True)))
    self.assertIsInstance(restored, pytorch_model.TabFM)

  def test_regressor_model_pickle_round_trip(self):
    restored = pickle.loads(pickle.dumps(self._tiny_model(is_classifier=False)))
    self.assertIsInstance(restored, pytorch_model.TabFM)

  def test_pickle_preserves_forward_output(self):
    # The unpickled model must still run and produce identical outputs: the
    # promoted _gelu_tanh is numerically identical to the previous lambda.
    torch.manual_seed(0)
    model = self._tiny_model(is_classifier=True).eval()
    x = torch.randn(2, 4, 6)  # [B, T, H]
    y = torch.randint(0, 3, (2, 4))  # [B, T]
    train_size = torch.tensor([2, 3])  # [B]
    with torch.no_grad():
      before = model(x, y, train_size)
      restored = pickle.loads(pickle.dumps(model)).eval()
      after = restored(x, y, train_size)
    torch.testing.assert_close(before, after)


if __name__ == "__main__":
  unittest.main()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLX-backend estimator tests (mirrors classifier_and_regressor_pytorch_test).

Covers the MLX paths junior added with zero prior coverage: cached vs
uncached agreement, mlx_batch_size chunking equivalence, and the regressor
cached path — all on small random MLX models (fast, no checkpoint needed).
"""

import unittest

import numpy as np

try:
  from tabfm.src.mlx import model as mlx_model_mod

  HAS_MLX = True
except ImportError:  # MLX ships macOS/arm64 wheels only.
  HAS_MLX = False
from tabfm.src.classifier_and_regressor import TabFMClassifier, TabFMRegressor


def _small_mlx_model(is_classifier, max_classes=3):
  return mlx_model_mod.TabFM(
      embed_dim=8,
      max_classes=max_classes,
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


@unittest.skipUnless(HAS_MLX, "mlx is required (Apple silicon only)")
class MlxClassifierRegressorTest(unittest.TestCase):

  def test_classifier_fit_predict(self):
    np.random.seed(42)
    model = _small_mlx_model(is_classifier=True)
    clf = TabFMClassifier(
        model=model, n_estimators=2, batch_size=2, random_state=42
    )
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 3, size=10)
    clf.fit(X, y)
    preds = clf.predict(X)
    self.assertEqual(preds.shape, (10,))
    self.assertTrue(np.all(preds >= 0) and np.all(preds < 3))
    probs = clf.predict_proba(X)
    self.assertEqual(probs.shape, (10, 3))
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)

  def test_classifier_cached_matches_uncached(self):
    np.random.seed(42)
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 3, size=10)
    model = _small_mlx_model(is_classifier=True)
    ref = TabFMClassifier(
        model=model, n_estimators=4, batch_size=2, random_state=42
    )
    ref.fit(X, y)
    probs = ref.predict_proba(X)
    cached = TabFMClassifier(
        model=model,
        n_estimators=4,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )
    cached.fit(X, y)
    probs_cached = cached.predict_proba(X)
    self.assertEqual(probs_cached.shape, probs.shape)
    np.testing.assert_allclose(probs_cached, probs, rtol=1e-4, atol=1e-5)

  def test_mlx_batch_size_equivalence(self):
    np.random.seed(7)
    X = np.random.rand(12, 3)
    y = np.random.randint(0, 3, size=12)
    model = _small_mlx_model(is_classifier=True)
    one = TabFMClassifier(
        model=model,
        n_estimators=4,
        batch_size=1,
        random_state=7,
        cache_context=True,
        maybe_quantize_kv_cache=False,
        mlx_batch_size=1,
    )
    one.fit(X, y)
    all_at_once = TabFMClassifier(
        model=model,
        n_estimators=4,
        batch_size=1,
        random_state=7,
        cache_context=True,
        maybe_quantize_kv_cache=False,
        mlx_batch_size=None,
    )
    all_at_once.fit(X, y)
    np.testing.assert_allclose(
        one.predict_proba(X), all_at_once.predict_proba(X), rtol=1e-5, atol=1e-6
    )
    # Uncached forward chunking must also match.
    one_nc = TabFMClassifier(
        model=model,
        n_estimators=4,
        batch_size=1,
        random_state=7,
        mlx_batch_size=1,
    )
    one_nc.fit(X, y)
    all_nc = TabFMClassifier(
        model=model,
        n_estimators=4,
        batch_size=1,
        random_state=7,
        mlx_batch_size=None,
    )
    all_nc.fit(X, y)
    np.testing.assert_allclose(
        one_nc.predict_proba(X), all_nc.predict_proba(X), rtol=1e-5, atol=1e-6
    )

  def test_regressor_fit_predict_cached(self):
    np.random.seed(42)
    X = np.random.rand(10, 3)
    y = np.random.rand(10) * 10
    model = _small_mlx_model(is_classifier=False)
    reg = TabFMRegressor(
        model=model, n_estimators=2, batch_size=2, random_state=42
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    self.assertEqual(preds.shape, (10,))
    self.assertTrue(np.all(np.isfinite(preds)))
    cached = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        random_state=42,
        cache_context=True,
        maybe_quantize_kv_cache=False,
    )
    cached.fit(X, y)
    np.testing.assert_allclose(cached.predict(X), preds, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  unittest.main()

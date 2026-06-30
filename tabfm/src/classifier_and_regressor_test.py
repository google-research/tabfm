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

import unittest
from unittest import mock
from absl.testing import absltest
import numpy as np
import pandas as pd

try:
  from flax import nnx
  from tabfm.src.jax import model as tabfm_model
  HAS_JAX = True
except ImportError:
  HAS_JAX = False
from tabfm.src.classifier_and_regressor import EnsembleGenerator
from tabfm.src.classifier_and_regressor import TabFMClassifier
from tabfm.src.classifier_and_regressor import TabFMRegressor
from tabfm.src.classifier_and_regressor import TransformToNumerical

# pylint: disable=invalid-name

class EnsembleGeneratorTest(absltest.TestCase):

  def test_permute_categorical_structure(self):
    # Create a simple dataset with one categorical column and one numerical
    X = pd.DataFrame({
        "cat": ["A", "B", "A", "B", "C"],
        "num": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    y = np.array([0, 1, 0, 1, 0])

    # Pre-encode using TransformToNumerical as the Classifier does
    encoder = TransformToNumerical(min_cat_frequency=1)
    X_enc = encoder.fit_transform(X)
    # "cat" should be column 0 (value 0, 1, 2), "num" column 1.

    cat_features = [0]
    n_estimators = 10

    generator = EnsembleGenerator(
        n_estimators=n_estimators,
        norm_methods=["none"],  # Use none to minimize transformation noise
        cat_features=cat_features,
        permute_categorical=True,
        random_state=42
    )

    generator.fit(X_enc, y)

    # Check if permutations are generated
    self.assertLen(generator.cat_permutations_, 1) # One group for "none" norm
    perms = generator.cat_permutations_["none"]
    self.assertLen(perms, n_estimators)

    # Verify structure of permutations
    for perm in perms:
      self.assertIn(0, perm)  # Column 0 should be in the permutation dict
      mapping = perm[0]
      # Original values are 0, 1, 2
      original_values = set(range(3))
      self.assertEqual(set(mapping.keys()), original_values)
      self.assertEqual(set(mapping.values()), original_values)

    # Verify that we actually have different permutations
    # It's statistically improbable that 10 estimators all have identity
    # permutation
    identity_count = 0
    for perm in perms:
      mapping = perm[0]
      is_identity = all(k == v for k, v in mapping.items())
      if is_identity:
        identity_count += 1

    self.assertLess(
        identity_count, n_estimators, "Categorical permutations should vary."
    )

  def test_permute_categorical_application(self):
    # Test that transform actually changes the data
    # X has 2 samples, 1 cat feature with values 0 and 1.
    X_enc = np.array([[0, 10.0], [1, 20.0]])
    y = np.array([0, 1])
    cat_features = [0]

    # Force a specific permutation logic by mocking or just checking output
    # We rely on random_state for reproducibility
    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        cat_features=cat_features,
        permute_categorical=True,
        # Disable feature shuffling to isolate value permutation
        feat_shuffle_method="none",
        random_state=42,
    )

    generator.fit(X_enc, y)
    data = generator.transform(X_enc)

    # Extract the transformed batch for "none" normalization
    X_out, _ = data["none"]
    # X_out shape: (n_estimators, n_samples, n_features)

    # We expect some estimators to swap 0 and 1 in the first column
    # Since PreprocessingPipeline with "none" still does StandardScaler,
    # the values won't be exactly 0 and 1, but they will be distinct.
    # However, if 0->1 and 1->0 (swap), the resulting standardized values
    # should effectively flip signs (if mean centered) or at least change.

    # Let's inspect the raw permutations to know what to expect
    perms = generator.cat_permutations_["none"]

    # Find an estimator that swaps 0 and 1
    swap_idx = -1
    for i, perm in enumerate(perms):
      mapping = perm[0]
      if mapping.get(0) == 1 and mapping.get(1) == 0:
        swap_idx = i
        break

    if swap_idx != -1:
      # Compare with an estimator that (hopefully) didn't swap or at least is
      # different
      # Actually, let's just compare X_out[swap_idx] vs input logic.
      # Input col 0: [0, 1]
      # Swapped col 0: [1, 0]
      # Standard Scaler on [0, 1] -> [-1, 1] (roughly)
      # Standard Scaler on [1, 0] -> [1, -1]
      # So the output values should be inverted relative to each other?
      # WAIT: The StandardScaler is fitted on the permuted TRAINING data.
      # If we permute train and test consistently, the distribution stats might
      # remain similar (since it's just relabeling), but the *instances* change
      # values.
      # 0 becomes 1.

      # Let's verify that X_out[swap_idx, 0, 0] (Sample 0, Feat 0)
      # is different from X_out[identity_idx, 0, 0] if we find an identity one.

      # Easier check: In the swapped estimator, Sample 0 (was 0->1) should look
      # like Sample 1 (was 1)
      # from a non-swapped estimator? No, Sample 1 in non-swapped is 1. Sample 0
      # in swapped is 1.
      # So yes, X_out[swap_idx, 0, 0] should be close to
      # X_out[non_swapped, 1, 0].
      pass

    # Assert outputs are not all identical across estimators for col 0.
    col0_values = X_out[:, 0, 0] # (n_estimators,)
    self.assertGreater(
        np.std(col0_values),
        1e-6,
        "Categorical values should vary across estimators due to permutation",
    )

    # Verify col 1 (numerical) does NOT vary (feat_shuffle="none")
    # Actually PreprocessingPipeline adds noise? No, only RTDLQuantile does.
    # CustomStandardScaler is deterministic.
    # So col 1 should be identical across estimators.
    col1_values = X_out[:, 0, 1]
    self.assertLess(
        np.std(col1_values),
        1e-6,
        "Numerical values should not vary if shuffling is off",
    )

  def test_permute_categorical_false(self):
    X_enc = np.array([[0, 10.0], [1, 20.0], [2, 30.0]])
    y = np.array([0, 1, 0])
    cat_features = [0]

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        cat_features=cat_features,
        permute_categorical=False, # DISABLED
        feat_shuffle_method="none",
        random_state=42
    )

    generator.fit(X_enc, y)

    # Check permutations are None
    perms = generator.cat_permutations_["none"]
    for perm in perms:
      self.assertIsNone(perm)

    data = generator.transform(X_enc)
    X_out, _ = data["none"]

    # Check that outputs are identical across estimators for col 0
    col0_values = X_out[:, 0, 0]
    self.assertLess(
        np.std(col0_values),
        1e-6,
        "Categorical values should be identical if permutation is disabled",
    )

  def test_row_and_col_subsampling(self):
    X_enc = np.array([[0, 10.0], [1, 20.0], [2, 30.0], [3, 40.0]])
    y = np.array([0, 1, 0, 1])

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        max_num_rows=2,
        max_num_features=1,
        random_state=42,
    )

    generator.fit(X_enc, y)

    feat_patterns = generator.feature_shuffle_patterns_["none"]
    for p in feat_patterns:
      self.assertLen(p, 1)

    row_patterns = generator.row_subsample_patterns_["none"]
    for p in row_patterns:
      self.assertLen(p, 2)

    X_test = np.array([[4, 50.0], [5, 60.0]])
    data = generator.transform(X_test)
    X_out, y_out = data["none"]

    self.assertEqual(X_out.shape, (5, 4, 1))
    self.assertEqual(y_out.shape, (5, 2))

  def test_max_num_features(self):
    X_enc = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        max_num_features=2,
        random_state=42,
    )

    generator.fit(X_enc, y)

    feat_patterns = generator.feature_shuffle_patterns_["none"]
    for p in feat_patterns:
      self.assertLen(p, 2)

  def test_variable_crosses_shapes(self):
    X_enc = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=10)

    # "split" allocation with sqrt feature crosses: sqrt(5) -> 2, so
    # even-indexed members get no crosses and odd-indexed members get 2.
    generator = EnsembleGenerator(
        n_estimators=5,
        norm_methods=["none"],
        n_feature_crosses="sqrt",
        random_state=42,
    )
    generator.fit(X_enc, y)

    self.assertEqual(generator.k_crosses_list_, [0, 2, 0, 2, 0])

    data = generator.transform(X_enc)
    X_out, _ = data["none"]
    # max features = 5 + 2 = 7
    self.assertEqual(X_out.shape[2], 7)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class OOFPredictionTest(absltest.TestCase):

  def test_classifier_predict_oof_proba(self):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    classifier = TabFMClassifier(
        model=model, n_estimators=2, batch_size=2
    )

    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)

    classifier.fit(X, y)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 2))

    with mock.patch.object(
        classifier, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      oof_preds = classifier.predict_oof_proba(cv=2)

      self.assertEqual(oof_preds.shape, (2, 10, 2))
      self.assertTrue(mock_batch_forward.called)

  def test_regressor_feature_crosses(self):
    regressor = TabFMRegressor(
        n_feature_crosses="sqrt",
        n_estimators=2,
        model=mock.Mock(),
    )

    X = np.random.rand(10, 4)
    y = np.random.rand(10)

    regressor.fit(X, y)

    # Split allocation with sqrt(4)=2 crosses on the odd member: pooled data
    # has N + 2 = 4 + 2 = 6 columns.
    self.assertEqual(regressor.ensemble_generator_.X_.shape[1], 6)

    X_test = np.random.rand(5, 4)
    data = regressor.ensemble_generator_.transform(X_test)
    for _, (Xs, _) in data.items():
      # Members are padded to N + K = 4 + 2 = 6 columns.
      self.assertEqual(Xs.shape[2], 6)

  def test_regressor_svd_features(self):
    regressor = TabFMRegressor(
        n_svd_features="sqrt",
        n_estimators=2,
        model=mock.Mock(),
    )

    X = pd.DataFrame({
        "num1": np.random.rand(10),
        "num2": np.random.rand(10),
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    })
    y = np.random.rand(10)

    regressor.fit(X, y)

    # 3 original (2 num + 1 cat) + 1 SVD (sqrt(3)=1) = 4 columns.
    self.assertEqual(regressor.ensemble_generator_.X_.shape[1], 4)

    X_test = pd.DataFrame({
        "num1": np.random.rand(5),
        "num2": np.random.rand(5),
        "cat1": ["a", "b", "a", "b", "a"],
    })
    X_test_encoded = regressor.X_encoder_.transform(X_test)
    data = regressor.ensemble_generator_.transform(X_test_encoded)
    for _, (Xs, _) in data.items():
      self.assertEqual(Xs.shape[2], 4)

  def test_regressor_nnls_blending(self):
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    regressor = TabFMRegressor(
        model=model,
        n_estimators=2,
        batch_size=2,
        enable_nnls=True,
        nnls_beta=0.5,
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    def mock_forward(Xs_batch, ys_batch, _cat_mask_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 1))

    with mock.patch.object(
        regressor, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      regressor.fit(X, y)

      self.assertTrue(mock_batch_forward.called)
      self.assertTrue(hasattr(regressor, "ensemble_weights_"))
      self.assertEqual(regressor.ensemble_weights_.shape, (2,))
      self.assertAlmostEqual(np.sum(regressor.ensemble_weights_), 1.0)

      test_preds = regressor.predict(X[:3])
      self.assertEqual(test_preds.shape, (3,))

  def test_regressor_sqrt_schedule(self):
    regressor = TabFMRegressor(
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        max_num_features=4,
        n_estimators=2,
        model=mock.Mock(),
    )

    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    regressor.fit(X, y)

    # Split allocation: sqrt(min(5, 4))=2. 5 original + 2 crosses pool +
    # 2 SVD pool = 9 columns.
    self.assertEqual(regressor.ensemble_generator_.X_.shape[1], 9)

    X_test = np.random.rand(5, 5)
    X_test_encoded = regressor.X_encoder_.transform(X_test)
    data = regressor.ensemble_generator_.transform(X_test_encoded)
    for _, (Xs, _) in data.items():
      # Members padded to max_num_features (4) + K_crosses (2) + K_svd (2) = 8.
      self.assertEqual(Xs.shape[2], 8)

  def test_regressor_predict_oof(self):
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    regressor = TabFMRegressor(
        model=model, n_estimators=2, batch_size=2
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    regressor.fit(X, y)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 1))

    with mock.patch.object(
        regressor, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      oof_preds = regressor.predict_oof(cv=2)

      self.assertEqual(oof_preds.shape, (2, 10))
      self.assertTrue(mock_batch_forward.called)

  def test_classifier_predict_oof_proba_with_subsampling(self):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    classifier = TabFMClassifier(
        model=model,
        n_estimators=2,
        batch_size=2,
        max_num_rows=8,
    )

    X = np.random.rand(10, 3)
    y = np.array([0, 1] * 5)

    classifier.fit(X, y)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 2))

    with mock.patch.object(
        classifier, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      oof_preds = classifier.predict_oof_proba(cv=2)

      self.assertEqual(oof_preds.shape, (2, 10, 2))
      self.assertTrue(mock_batch_forward.called)

  def test_regressor_variable_crosses_predict_flow(self):
    regressor = TabFMRegressor(
        n_estimators=2,
        batch_size=2,
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        model=mock.Mock(),
    )

    X = np.random.rand(10, 3)
    y = np.random.rand(10)
    X_test = np.random.rand(5, 3)

    def mock_forward(Xs_batch, ys_batch, _cat_masks_batch=None, **_kwargs):
      test_size = Xs_batch.shape[1] - ys_batch.shape[1]
      return np.zeros((Xs_batch.shape[0], test_size, 1))

    with mock.patch.object(
        regressor, "_batch_forward", side_effect=mock_forward
    ) as mock_batch_forward:
      regressor.fit(X, y)
      # Split allocation with sqrt(3)=1 for both crosses and SVD.
      self.assertEqual(regressor.ensemble_generator_.k_crosses_list_, [0, 1])
      self.assertEqual(regressor.ensemble_generator_.k_svd_list_, [0, 1])

      preds = regressor.predict(X_test)
      self.assertEqual(preds.shape, (5,))

      self.assertTrue(mock_batch_forward.called)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class BatchForwardTest(absltest.TestCase):

  def test_classifier_batch_forward(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )

    classifier = TabFMClassifier(
        model=model, n_estimators=4, batch_size=2
    )
    classifier.n_classes_ = 3
    classifier.classes_ = np.array([0, 1, 2])

    # Generate dummy input arrays.
    # Xs shape: (n_estimators, max_seq_len, num_features)
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.randint(0, 3, size=(4, 6))

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = classifier._batch_forward(Xs, ys)

    # After concatenation, output shape should be
    # (n_estimators, test_size, num_classes).
    # Since Xs length is 10 and ys length (train_size) is 6, test_size is 4.
    self.assertEqual(outputs.shape, (4, 4, 3))

  def test_regressor_batch_forward_rmse(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="rmse",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )

    regressor = TabFMRegressor(model=model, n_estimators=4)

    # Generate dummy input arrays.
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.rand(4, 6)

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = regressor._batch_forward(Xs, ys)

    # Output shape should be (n_estimators, test_size, out_dim).
    # Since Xs length is 10 and ys length (train_size) is 6, test_size is 4.
    # TabFM with rmse outputs 1 value per prediction.
    self.assertEqual(outputs.shape, (4, 4, 1))

  def test_regressor_batch_forward_cross_entropy(self):
    rngs = nnx.Rngs(0)
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
    )

    regressor = TabFMRegressor(model=model, n_estimators=4)

    # Generate dummy input arrays.
    Xs = np.random.rand(4, 10, 5)
    # ys shape: (n_estimators, train_size)
    ys = np.random.rand(4, 6)

    # Run _batch_forward. Uses data-parallel JAX sharding internally.
    outputs = regressor._batch_forward(Xs, ys)

    # Output shape should be (n_estimators, test_size, out_dim).
    # TabFM with cross_entropy outputs max_classes bins.
    self.assertEqual(outputs.shape, (4, 4, 10))


@unittest.skipUnless(HAS_JAX, "JAX is required")
class CalibrationTest(absltest.TestCase):

  def setUp(self):
    self.model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )

  def test_calibration_binary(self):
    y = np.concatenate(
        [np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)]
    )
    X = np.random.rand(100, 3)
    mock_oof = np.random.dirichlet([1, 1], size=(2, 100))

    for method in ["platt"]:
      classifier = TabFMClassifier(
          model=self.model,
          n_estimators=2,
          binary_calibration_method=method,
      )

      with mock.patch.object(
          classifier, "predict_oof_proba", return_value=mock_oof
      ):
        classifier.fit(X, y)

      self.assertTrue(hasattr(classifier, "calibration_params_"))

      mock_out = np.random.dirichlet([1, 1], size=(2, 100))

      with mock.patch.object(
          classifier, "_batch_forward", return_value=mock_out
      ):
        probs = classifier.predict_proba(X)
        self.assertEqual(probs.shape, (100, 2))

  def test_calibration_multiclass(self):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=3,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    y = np.concatenate([
        np.zeros(50, dtype=np.int64),
        np.ones(50, dtype=np.int64),
        np.full(50, 2, dtype=np.int64),
    ])
    X = np.random.rand(150, 3)
    mock_oof = np.random.dirichlet([1, 1, 1], size=(2, 150))

    for method in ["vector"]:
      classifier = TabFMClassifier(
          model=model,
          n_estimators=2,
          multiclass_calibration_method=method,
      )

      with mock.patch.object(
          classifier, "predict_oof_proba", return_value=mock_oof
      ):
        classifier.fit(X, y)

      self.assertTrue(hasattr(classifier, "calibration_params_"))

      mock_out = np.random.dirichlet([1, 1, 1], size=(2, 150))

      with mock.patch.object(
          classifier, "_batch_forward", return_value=mock_out
      ):
        probs = classifier.predict_proba(X)
        self.assertEqual(probs.shape, (150, 3))


@unittest.skipUnless(HAS_JAX, "JAX is required")
class StackingTest(absltest.TestCase):

  def setUp(self):
    self.model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )

  def test_classifier_crosses_and_svd(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        n_feature_crosses="sqrt",
        n_svd_features="sqrt",
        max_num_features=4,
    )

    X = pd.DataFrame({
        "num1": np.random.rand(10),
        "num2": np.random.rand(10),
        "num3": np.random.rand(10),
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "cat2": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
    })
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    classifier.fit(X, y)

    self.assertEqual(classifier.ensemble_generator_.X_.shape[1], 9)

    probs = classifier.predict_proba(X)
    self.assertEqual(probs.shape, (10, 2))

  def test_classifier_nnls_blending(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        enable_nnls=True,
        nnls_beta=0.5,
        average_logits=False,
    )

    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    mock_oof = np.random.dirichlet([1, 1], size=(2, 10))

    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      classifier.fit(X, y)

    self.assertTrue(hasattr(classifier, "ensemble_weights_"))
    self.assertEqual(classifier.ensemble_weights_.shape, (2,))
    self.assertAlmostEqual(np.sum(classifier.ensemble_weights_), 1.0)

    mock_logits = np.random.rand(2, 5, 2)
    with mock.patch.object(
        classifier, "_predict_proba_internal", return_value=mock_logits
    ):
      probs = classifier.predict_proba(X[:5])
      self.assertEqual(probs.shape, (5, 2))

  def test_classifier_nnls_and_calibration(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        enable_nnls=True,
        nnls_beta=0.5,
        binary_calibration_method="platt",
        average_logits=False,
    )

    X = np.random.rand(50, 3)
    y = np.concatenate(
        [np.zeros(25, dtype=np.int64), np.ones(25, dtype=np.int64)]
    )

    mock_oof = np.random.dirichlet([1, 1], size=(2, 50))

    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      with mock.patch.object(classifier, "_fit_calibration") as mock_fit_cal:
        classifier.fit(X, y)

        self.assertTrue(mock_fit_cal.called)
        P_arg = mock_fit_cal.call_args[0][0]
        expected_P = np.tensordot(
            classifier.ensemble_weights_, mock_oof, axes=(0, 0)
        )
        np.testing.assert_allclose(P_arg, expected_P)

  def test_classifier_nnls_and_calibration_small_dataset(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        enable_nnls=True,
        nnls_beta=0.5,
        binary_calibration_method="platt",
        calibration_lambda=0.01,
        average_logits=False,
    )

    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_oof = np.random.dirichlet([1, 1], size=(2, 10))

    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      classifier.fit(X, y)

    self.assertEqual(classifier.active_calibration_method_, "platt")
    self.assertTrue(hasattr(classifier, "calibration_params_"))
    self.assertIn("A", classifier.calibration_params_)

  def test_average_logits_and_enable_nnls_raises(self):
    with self.assertRaises(ValueError):
      TabFMClassifier(
          model=self.model,
          average_logits=True,
          enable_nnls=True,
      )

  def test_max_num_rows(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=2,
        max_num_rows=6,
    )
    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    mock_oof = np.random.dirichlet([1, 1], size=(2, 10))
    with mock.patch.object(
        classifier, "predict_oof_proba", return_value=mock_oof
    ):
      classifier.fit(X, y)

    self.assertEqual(classifier.ensemble_generator_.X_.shape[0], 10)
    patterns_none = classifier.ensemble_generator_.row_subsample_patterns_[
        "none"
    ]
    patterns_power = classifier.ensemble_generator_.row_subsample_patterns_[
        "power"
    ]
    self.assertLen(patterns_none[0], 6)
    self.assertLen(patterns_power[0], 6)

  def test_min_rows_for_single_val_split_classifier(self):
    classifier = TabFMClassifier(
        model=self.model,
        n_estimators=1,
        enable_nnls=True,
        average_logits=False,
        min_rows_for_single_val_split=2,
        num_folds_for_cv=5,
    )
    X = np.random.rand(10, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    with mock.patch.object(
        classifier, "_batch_forward", return_value=np.zeros((1, 2, 2))
    ) as mock_forward:
      classifier.fit(X, y)

    # With 10 rows and 5 folds, val fold size is 2.
    # It should therefore only run 1 fold instead of 5.
    self.assertEqual(mock_forward.call_count, 1)

  def test_min_rows_for_single_val_split_regressor(self):
    regressor = TabFMRegressor(
        model=self.model,
        n_estimators=1,
        enable_nnls=True,
        min_rows_for_single_val_split=2,
        num_folds_for_cv=5,
    )
    X = np.random.rand(10, 3)
    y = np.random.rand(10)

    with mock.patch.object(
        regressor, "_batch_forward", return_value=np.zeros((1, 2, 1))
    ) as mock_forward:
      regressor.fit(X, y)

    self.assertEqual(mock_forward.call_count, 1)


class EnsemblePresetTest(absltest.TestCase):

  def test_classifier_ensemble_preset(self):
    clf = TabFMClassifier.ensemble(model=mock.Mock())
    self.assertEqual(clf.n_estimators, 32)
    self.assertFalse(clf.average_logits)
    self.assertEqual(clf.n_feature_crosses, "sqrt")
    self.assertEqual(clf.n_svd_features, "sqrt")
    self.assertTrue(clf.enable_nnls)
    self.assertEqual(clf.binary_calibration_method, "platt")
    self.assertEqual(clf.multiclass_calibration_method, "vector")
    # Default-mode knobs are unchanged by the preset.
    self.assertEqual(clf.max_num_features, 500)

  def test_regressor_ensemble_preset(self):
    reg = TabFMRegressor.ensemble(model=mock.Mock())
    self.assertEqual(reg.n_estimators, 32)
    self.assertEqual(reg.n_feature_crosses, "sqrt")
    self.assertEqual(reg.n_svd_features, "sqrt")
    self.assertTrue(reg.enable_nnls)
    self.assertEqual(reg.max_num_features, 500)

  def test_ensemble_overrides_take_precedence(self):
    clf = TabFMClassifier.ensemble(
        model=mock.Mock(), n_estimators=8, enable_nnls=False
    )
    self.assertEqual(clf.n_estimators, 8)
    self.assertFalse(clf.enable_nnls)
    # Non-overridden preset values are preserved.
    self.assertEqual(clf.n_feature_crosses, "sqrt")

  def test_default_mode_keeps_features_off(self):
    clf = TabFMClassifier(model=mock.Mock())
    self.assertEqual(clf.n_estimators, 32)
    self.assertEqual(clf.max_num_features, 500)
    self.assertTrue(clf.average_logits)
    self.assertEqual(clf.n_feature_crosses, 0)
    self.assertEqual(clf.n_svd_features, 0)
    self.assertFalse(clf.enable_nnls)
    self.assertIsNone(clf.binary_calibration_method)


@unittest.skipUnless(HAS_JAX, "JAX is required")
class LabelEncodingTest(absltest.TestCase):

  def test_classes_are_alphabetical(self):
    # classes_ must follow sklearn's sorted (alphabetical) convention,
    # independent of label order of appearance. Regression guard for the
    # y-encoder being constructed with mode="alphabetical".
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=nnx.Rngs(0),
    )
    clf = TabFMClassifier(model=model, n_estimators=2)
    x = np.random.rand(6, 3)
    # Appearance order is ("z", "a"); alphabetical order is ("a", "z").
    y = np.array(["z", "a", "z", "a", "z", "a"])
    clf.fit(x, y)
    np.testing.assert_array_equal(clf.classes_, np.array(["a", "z"]))
    self.assertEqual(clf.y_encoder_.mode, "alphabetical")


if __name__ == "__main__":
  absltest.main()

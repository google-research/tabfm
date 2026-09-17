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

"""Estimator configuration errors should be reported before preprocessing."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from tabfm.src.classifier_and_regressor import TabFMClassifier
from tabfm.src.classifier_and_regressor import TabFMRegressor


@pytest.mark.parametrize("estimator_class", [TabFMClassifier, TabFMRegressor])
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("n_estimators", 0),
        ("n_estimators", -1),
        ("n_estimators", 1.5),
        ("max_num_features", 0),
        ("max_num_rows", -1),
        ("batch_size", -1),
        ("batch_size", 1.5),
        ("num_folds_for_cv", 1),
        ("total_svd_pool", -1),
        ("min_rows_for_single_val_split", -1),
        ("nnls_beta", -0.1),
        ("nnls_beta", 1.1),
        ("nnls_beta", np.nan),
    ],
)
def test_invalid_parameters_fail_before_preprocessing(
    estimator_class, parameter, value
):
  estimator = estimator_class(model=SimpleNamespace(max_classes=3))
  # Check set_params too: sklearn searches configure estimators after __init__.
  estimator.set_params(**{parameter: value})
  with mock.patch(
      "tabfm.src.classifier_and_regressor.TransformToNumerical.fit_transform"
  ) as preprocess:
    with pytest.raises(ValueError, match=parameter):
      estimator.fit(np.arange(12).reshape(6, 2), np.array([0, 1] * 3))
    preprocess.assert_not_called()


@pytest.mark.parametrize("value", [-1.0, np.nan])
def test_negative_or_nan_calibration_penalty_is_rejected(value):
  classifier = TabFMClassifier(
      model=SimpleNamespace(max_classes=3), calibration_lambda=value
  )
  with pytest.raises(ValueError, match="calibration_lambda"):
    classifier.fit(np.arange(12).reshape(6, 2), np.array([0, 1] * 3))


@pytest.mark.parametrize("estimator_class", [TabFMClassifier, TabFMRegressor])
@pytest.mark.parametrize("batch_size", [None, 0, 1])
@pytest.mark.parametrize("nnls_beta", [0.0, 1.0])
def test_valid_boundary_values_fit(estimator_class, batch_size, nnls_beta):
  estimator = estimator_class(
      model=SimpleNamespace(max_classes=3),
      n_estimators=1,
      norm_methods="none",
      max_num_features=None,
      max_num_rows=None,
      batch_size=batch_size,
      nnls_beta=nnls_beta,
      num_folds_for_cv=2,
      min_rows_for_single_val_split=0,
      total_svd_pool=0,
  )
  X = np.random.default_rng(42).normal(size=(12, 3))
  estimator.fit(X, np.array([0, 1] * 6))
  assert estimator.n_estimators == 1
  np.testing.assert_array_equal(estimator.ensemble_generator_.X_, X)


def test_zero_calibration_penalty_is_valid():
  estimator = TabFMClassifier(
      model=SimpleNamespace(max_classes=3), calibration_lambda=0
  )
  estimator._validate_params()

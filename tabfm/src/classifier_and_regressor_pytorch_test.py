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


if __name__ == "__main__":
  unittest.main()

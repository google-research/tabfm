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

"""Example showing how to run regression with TabFM v1.0.0."""

import numpy as np
import pandas as pd
import tabfm


def run_example(model=None) -> np.ndarray:
  """Generates dummy data and runs regression."""
  if model is None:
    model = tabfm.tabfm_v1_0_0.load(model_type="regression")

  # 2. Initialize scikit-learn compatible regressor
  reg = tabfm.TabFMRegressor(model=model)

  # 3. Generate dummy dataset
  X_train = pd.DataFrame({
      "num_feat_1": [1.5, 2.5, 3.5, 4.5, 5.5],
      "cat_feat_1": ["A", "B", "A", "B", "C"],
  })
  y_train = np.array([10.5, 20.0, 11.0, 29.5, 21.0])

  X_test = pd.DataFrame({
      "num_feat_1": [2.0, 4.0],
      "cat_feat_1": ["B", "A"],
  })

  # 4. Fit and predict
  reg.fit(X_train, y_train)
  preds = reg.predict(X_test)
  return preds


if __name__ == "__main__":
  print("Running TabFM regression model... (Note: JAX compilation and model execution may take a few minutes on first run)")
  predictions = run_example()
  print("Regression predictions:\n", predictions)

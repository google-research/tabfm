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

"""Example showing how to run classification with TabFM v1.0.0."""

import numpy as np
import pandas as pd
import tabfm


def run_example(model=None) -> np.ndarray:
  """Generates dummy data and runs classification."""
  if model is None:
    model = tabfm.tabfm_v1_0_0.load(model_type="classification")

  # 2. Initialize scikit-learn compatible classifier
  clf = tabfm.TabFMClassifier(model=model)

  # 3. Generate dummy dataset (with mixed column types)
  # Context (training) data
  X_train = pd.DataFrame({
      "num_feat_1": [1.5, 2.5, 3.5, 4.5, 5.5],
      "cat_feat_1": ["A", "B", "A", "B", "C"],
      "num_feat_2": [10.0, 20.0, 10.0, 30.0, 20.0],
  })
  y_train = np.array(["yes", "no", "yes", "no", "yes"])

  # Test samples
  X_test = pd.DataFrame({
      "num_feat_1": [2.0, 4.0],
      "cat_feat_1": ["B", "A"],
      "num_feat_2": [15.0, 25.0],
  })

  # 4. Fit the classifier (prepares internal data transformers)
  clf.fit(X_train, y_train)

  # 5. Predict class probabilities
  probs = clf.predict_proba(X_test)
  return probs


if __name__ == "__main__":
  print("Running TabFM classification model... (Note: JAX compilation and model execution may take a few minutes on first run)")
  predictions = run_example()
  print("Classification predictions:\n", predictions)

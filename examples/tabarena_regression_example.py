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

"""Regression on a real TabArena task with TabFM v1.0.0.

Loads repeat-0 / fold-0 of the TabArena ``airfoil_self_noise`` task (a small
dataset with a mix of numerical and categorical columns), then runs inference
twice:

  * default  -- ``TabFMRegressor(model=...)`` (uniform averaging)
  * ensemble -- ``TabFMRegressor.ensemble(model=...)`` (feature crosses / SVD
    features + NNLS-weighted blending)

and reports RMSE, R^2 and MAE for each so the two presets can be compared.

Requires the optional ``openml`` dependency: ``pip install tabfm[examples]``.
"""

import numpy as np
import tabfm

# OpenML task id for the TabArena "airfoil_self_noise" regression task
# (1503 rows, 4 numerical + 1 categorical features).
TASK_ID = 363612
SEED = 0


def _load_fold_0(task_id):
  """Returns (X_train, y_train, X_test, y_test) for repeat-0 / fold-0.

  Note: both X and y are passed to TabFM as raw OpenML values. TabFM routes
  each feature column by dtype internally (pandas ``category`` columns are
  handled natively by ``CategoricalOrdinalEncoder``) and coerces the target to
  numeric itself (``check_array(y, dtype="numeric")``), so no manual coercion is
  needed here.
  """
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(task_id)
  dataset = task.get_dataset()
  x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
  split = task.download_split().split[0][0][0]  # repeat 0, fold 0, sample 0

  x_train = x.iloc[split.train].copy()
  x_test = x.iloc[split.test].copy()
  y_train = y.iloc[split.train]  # raw target -- TabFM coerces to numeric itself
  y_test = y.iloc[split.test]
  return x_train, y_train, x_test, y_test


def _evaluate(reg, x_train, y_train, x_test, y_test):
  """Fits ``reg`` and returns (rmse, r2, mae) on the test fold."""
  from sklearn.metrics import (  # pylint: disable=g-import-not-at-top, g-multiple-import
      mean_absolute_error,
      mean_squared_error,
      r2_score,
  )

  reg.fit(x_train, y_train)
  pred = np.asarray(reg.predict(x_test), dtype=float).ravel()
  rmse = mean_squared_error(y_test, pred) ** 0.5
  return rmse, r2_score(y_test, pred), mean_absolute_error(y_test, pred)


def run_example(model=None):
  """Runs default and ensemble regression on TabArena fold 0.

  Args:
    model: An optional pre-loaded TabFM regression model. Loaded from Hugging
      Face when ``None``.

  Returns:
    A dict mapping "default"/"ensemble" to an (rmse, r2, mae) tuple.
  """
  if model is None:
    model = tabfm.tabfm_v1_0_0.load(model_type="regression")

  x_train, y_train, x_test, y_test = _load_fold_0(TASK_ID)

  results = {}
  results["default"] = _evaluate(
      tabfm.TabFMRegressor(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  results["ensemble"] = _evaluate(
      tabfm.TabFMRegressor.ensemble(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  return results


if __name__ == "__main__":
  print(
      "Running TabFM regression on TabArena airfoil_self_noise"
      " (fold 0)... (Note: JAX compilation and model execution may take a few"
      " minutes on first run)"
  )
  scores = run_example()
  print(f"\n{'preset':<10} {'RMSE':>12} {'R2':>9} {'MAE':>12}")
  for preset, (rmse, r2, mae) in scores.items():
    print(f"{preset:<10} {rmse:>12.5g} {r2:>9.5f} {mae:>12.5g}")

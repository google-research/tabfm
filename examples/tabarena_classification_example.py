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

"""Classification on a real TabArena task with TabFM v1.0.0.

Loads repeat-0 / fold-0 of the TabArena ``maternal_health_risk`` task (a small
3-class dataset with string labels), then runs inference twice:

  * default  -- ``TabFMClassifier(model=...)`` (uniform logit averaging)
  * ensemble -- ``TabFMClassifier.ensemble(model=...)`` (feature crosses / SVD
    features + NNLS-weighted blending + probability calibration)

and reports ROC AUC and log loss for each so the two presets can be compared.

Requires the optional ``openml`` dependency: ``pip install tabfm[examples]``.
"""

import numpy as np
import tabfm

# OpenML task id for the TabArena "maternal_health_risk" classification task
# (1014 rows, 6 numerical features, 3-class string target).
TASK_ID = 363685
SEED = 0


def _load_fold_0(task_id):
  """Returns (X_train, y_train, X_test, y_test) for repeat-0 / fold-0.

  Note: both X and y are passed to TabFM as raw OpenML values -- including the
  string class labels, which TabFM label-encodes internally (and exposes via
  ``clf.classes_``) -- so no manual coercion is needed here.
  """
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(task_id)
  dataset = task.get_dataset()
  x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
  split = task.download_split().split[0][0][0]  # repeat 0, fold 0, sample 0

  x_train = x.iloc[split.train].copy()
  x_test = x.iloc[split.test].copy()
  y_train = y.iloc[split.train]  # raw labels -- TabFM encodes them internally
  y_test = y.iloc[split.test]
  return x_train, y_train, x_test, y_test


def _evaluate(clf, x_train, y_train, x_test, y_test):
  """Fits ``clf`` on raw labels and returns (roc_auc, log_loss) on test fold."""
  from sklearn.metrics import log_loss, roc_auc_score  # pylint: disable=g-import-not-at-top

  # Fit directly on the raw string/category target -- TabFM encodes it
  # internally and exposes the original labels via ``clf.classes_``.
  clf.fit(x_train, y_train)
  proba = np.asarray(clf.predict_proba(x_test))

  # Align probability columns to sorted label order, then encode y_test against
  # those same labels -- only the sklearn metrics need numeric labels.
  order = np.argsort(clf.classes_)
  proba = proba[:, order]
  labels = np.asarray(clf.classes_)[order]
  y_test = np.asarray(y_test)

  ll = log_loss(y_test, proba, labels=labels)
  if len(labels) == 2:
    # Binary AUC needs a 0/1 indicator; treat the second sorted label as the
    # positive class (matching proba[:, 1]).
    y_true_bin = (y_test == labels[1]).astype(int)
    auc = roc_auc_score(y_true_bin, proba[:, 1])
  else:
    auc = roc_auc_score(
        y_test, proba, multi_class="ovr", labels=labels, average="macro"
    )
  return auc, ll


def run_example(model=None):
  """Runs default and ensemble classification on TabArena fold 0.

  Args:
    model: An optional pre-loaded TabFM classification model. Loaded from
      Hugging Face when ``None``.

  Returns:
    A dict mapping "default"/"ensemble" to a (roc_auc, log_loss) tuple.
  """
  if model is None:
    model = tabfm.tabfm_v1_0_0.load(model_type="classification")

  x_train, y_train, x_test, y_test = _load_fold_0(TASK_ID)

  results = {}
  results["default"] = _evaluate(
      tabfm.TabFMClassifier(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  results["ensemble"] = _evaluate(
      tabfm.TabFMClassifier.ensemble(model=model, random_state=SEED),
      x_train, y_train, x_test, y_test,
  )
  return results


if __name__ == "__main__":
  print(
      "Running TabFM classification on TabArena maternal_health_risk (fold 0)..."
      " (Note: JAX compilation and model execution may take a few minutes on"
      " first run)"
  )
  scores = run_example()
  print(f"\n{'preset':<10} {'ROC_AUC':>9} {'log_loss':>9}")
  for preset, (auc, ll) in scores.items():
    print(f"{preset:<10} {auc:>9.5f} {ll:>9.5f}")

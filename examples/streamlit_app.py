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

"""Lightweight Streamlit GUI for fitting TabFM on an uploaded CSV.

Run with:

  streamlit run examples/streamlit_app.py

The app keeps Streamlit optional by importing it only from ``main()``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


CLASSIFICATION = "classification"
REGRESSION = "regression"
SUPPORTED_TASKS = (CLASSIFICATION, REGRESSION)
SUPPORTED_BACKENDS = ("jax", "pytorch")


def split_training_and_prediction_data(
    data: pd.DataFrame,
    target_column: str,
    prediction_data: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
  """Splits uploaded data into TabFM train features, target, and predict data.

  Rows with a missing target are ignored for training. If ``prediction_data`` is
  not provided, predictions are generated for every row in ``data`` after
  dropping the target column.

  Args:
    data: Uploaded CSV data containing the target column.
    target_column: Name of the target column to predict.
    prediction_data: Optional second CSV containing rows to predict. If it also
      contains ``target_column``, that column is ignored.

  Returns:
    ``(X_train, y_train, X_predict)`` suitable for the sklearn-style TabFM API.

  Raises:
    ValueError: If the target column is missing or has no non-null values.
  """
  if target_column not in data.columns:
    raise ValueError(f"Target column {target_column!r} is not in the CSV.")

  train_rows = data[target_column].notna()
  if not train_rows.any():
    raise ValueError("At least one row must have a non-empty target value.")

  X_train = data.loc[train_rows].drop(columns=[target_column])
  y_train = data.loc[train_rows, target_column].to_numpy()

  if prediction_data is None:
    X_predict = data.drop(columns=[target_column])
  elif target_column in prediction_data.columns:
    X_predict = prediction_data.drop(columns=[target_column])
  else:
    X_predict = prediction_data.copy()

  return X_train, y_train, X_predict


def build_predictions_dataframe(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    class_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
  """Builds a CSV-friendly predictions table.

  Args:
    predictions: Predicted labels or regression values.
    probabilities: Optional class-probability matrix for classification.
    class_labels: Optional class labels matching the probability columns.

  Returns:
    DataFrame containing a ``tabfm_prediction`` column and, when supplied,
    one probability column per class.
  """
  result = pd.DataFrame({"tabfm_prediction": np.asarray(predictions)})
  if probabilities is None:
    return result

  probabilities = np.asarray(probabilities)
  if probabilities.ndim != 2:
    raise ValueError("Classification probabilities must be a 2D array.")

  if class_labels is None:
    class_labels = np.arange(probabilities.shape[1])
  if len(class_labels) != probabilities.shape[1]:
    raise ValueError("Class labels must match probability columns.")

  for index, label in enumerate(class_labels):
    result[f"probability_{label}"] = probabilities[:, index]
  return result


def _load_backend_model(task_type: str, backend: str):
  """Loads TabFM weights for the selected task and backend."""
  if task_type not in SUPPORTED_TASKS:
    raise ValueError(f"Unsupported task type: {task_type!r}")
  if backend not in SUPPORTED_BACKENDS:
    raise ValueError(f"Unsupported backend: {backend!r}")

  import tabfm  # pylint: disable=import-outside-toplevel

  module = (
      tabfm.tabfm_v1_0_0_jax
      if backend == "jax"
      else tabfm.tabfm_v1_0_0_pytorch
  )
  return module.load(model_type=task_type)


def fit_tabfm_and_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_predict: pd.DataFrame,
    task_type: str,
    backend: str,
) -> pd.DataFrame:
  """Fits the selected TabFM estimator and returns predictions as a DataFrame."""
  import tabfm  # pylint: disable=import-outside-toplevel

  model = _load_backend_model(task_type, backend)
  if task_type == CLASSIFICATION:
    estimator = tabfm.TabFMClassifier(model=model)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_predict)
    probabilities = estimator.predict_proba(X_predict)
    return build_predictions_dataframe(
        predictions, probabilities, estimator.classes_
    )

  if task_type == REGRESSION:
    estimator = tabfm.TabFMRegressor(model=model)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_predict)
    return build_predictions_dataframe(predictions)

  raise ValueError(f"Unsupported task type: {task_type!r}")


def main() -> None:
  """Runs the Streamlit app."""
  try:
    import streamlit as st  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError(
        "The TabFM demo GUI requires Streamlit. Install it with "
        "`pip install streamlit` or `pip install -e .[examples]`."
    ) from exc

  st.set_page_config(page_title="TabFM CSV GUI", layout="wide")
  st.title("TabFM CSV GUI")
  st.write(
      "Upload a CSV, select the target and task type, fit TabFM, and download "
      "predictions. This is a lightweight demo interface, not a full product."
  )

  training_file = st.file_uploader("Training CSV", type="csv")
  if training_file is None:
    st.info("Upload a CSV file to get started.")
    return

  data = pd.read_csv(training_file)
  st.subheader("Preview")
  st.dataframe(data.head())

  target_column = st.selectbox("Target column", list(data.columns))
  task_type = st.radio("Task type", SUPPORTED_TASKS, horizontal=True)
  backend = st.selectbox("Backend", SUPPORTED_BACKENDS)
  prediction_file = st.file_uploader(
      "Optional prediction CSV. If omitted, the app predicts every row in the "
      "training CSV after dropping the target column.",
      type="csv",
  )

  if st.button("Fit TabFM and generate predictions"):
    prediction_data = pd.read_csv(prediction_file) if prediction_file else None
    try:
      X_train, y_train, X_predict = split_training_and_prediction_data(
          data, target_column, prediction_data
      )
      with st.spinner("Loading TabFM, fitting, and predicting..."):
        predictions = fit_tabfm_and_predict(
            X_train, y_train, X_predict, task_type, backend
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      st.error(f"TabFM prediction failed: {exc}")
      return

    st.success(f"Generated {len(predictions)} predictions.")
    st.dataframe(predictions)
    st.download_button(
        "Download predictions CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="tabfm_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
  main()

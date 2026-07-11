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

"""Tests for the Streamlit TabFM demo helpers."""

from absl.testing import absltest
import numpy as np
import pandas as pd

from examples import streamlit_app


class StreamlitAppTest(absltest.TestCase):

  def test_split_training_and_prediction_data_uses_non_null_targets(self):
    data = pd.DataFrame({
        "feature": [1, 2, 3],
        "category": ["a", "b", "c"],
        "target": ["yes", None, "no"],
    })
    prediction_data = pd.DataFrame({
        "feature": [4, 5],
        "category": ["d", "e"],
        "target": [None, None],
    })

    X_train, y_train, X_predict = (
        streamlit_app.split_training_and_prediction_data(
            data, "target", prediction_data
        )
    )

    pd.testing.assert_frame_equal(
        X_train,
        pd.DataFrame({
            "feature": [1, 3],
            "category": ["a", "c"],
        }, index=[0, 2]),
    )
    np.testing.assert_array_equal(y_train, np.array(["yes", "no"]))
    pd.testing.assert_frame_equal(
        X_predict,
        pd.DataFrame({"feature": [4, 5], "category": ["d", "e"]}),
    )

  def test_build_predictions_dataframe_includes_class_probabilities(self):
    predictions = np.array(["cat", "dog"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9]])
    classes = np.array(["cat", "dog"])

    result = streamlit_app.build_predictions_dataframe(
        predictions, probabilities, classes
    )

    expected = pd.DataFrame({
        "tabfm_prediction": ["cat", "dog"],
        "probability_cat": [0.8, 0.1],
        "probability_dog": [0.2, 0.9],
    })
    pd.testing.assert_frame_equal(result, expected)

  def test_split_training_and_prediction_data_rejects_empty_targets(self):
    data = pd.DataFrame({"feature": [1, 2], "target": [None, None]})

    with self.assertRaisesRegex(ValueError, "non-empty target"):
      streamlit_app.split_training_and_prediction_data(data, "target")


if __name__ == "__main__":
  absltest.main()

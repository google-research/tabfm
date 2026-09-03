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

import numpy as np
import pandas as pd

from tabfm.src.generation import TabFMDataGenerator
from tabfm.src.pytorch import model as pytorch_model


def _tiny_model():
  return pytorch_model.TabFM(
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
      is_classifier=True,
  )


def _reference_frame():
  rng = np.random.default_rng(0)
  n = 24
  return pd.DataFrame({
      "num": rng.normal(size=n),
      "cat": np.where(rng.random(n) < 0.5, "a", "b"),
      "const": np.full(n, 7.0),
  })


class TabFMDataGeneratorEndToEndTest(unittest.TestCase):

  def test_sample_shapes_dtypes_and_domains(self):
    df = _reference_frame()
    gen = TabFMDataGenerator(model=_tiny_model(), n_estimators=2,
                             random_state=42).fit(df)
    out = gen.sample(n_samples=8)
    self.assertEqual(list(out.columns), ["num", "cat", "const"])
    self.assertEqual(len(out), 8)
    self.assertTrue(set(out["cat"]) <= {"a", "b"})
    self.assertTrue(np.all(out["const"].to_numpy() == 7.0))
    num = out["num"].to_numpy()
    self.assertTrue(np.all(num >= df["num"].min())
                    and np.all(num <= df["num"].max()))
    # Fresh continuous values, not copies of reference rows.
    self.assertFalse(bool(set(num) & set(df["num"])))

  def test_sample_is_deterministic_given_seed(self):
    # One shared model instance: _tiny_model() has random init, so two
    # instances would differ regardless of the generator's seeding.
    df = _reference_frame()
    model = _tiny_model()
    out1 = TabFMDataGenerator(model=model, n_estimators=2,
                              random_state=7).fit(df).sample(6)
    out2 = TabFMDataGenerator(model=model, n_estimators=2,
                              random_state=7).fit(df).sample(6)
    pd.testing.assert_frame_equal(out1, out2)

  def test_explicit_column_order_and_validation(self):
    df = _reference_frame()
    gen = TabFMDataGenerator(model=_tiny_model(), n_estimators=2,
                             random_state=0).fit(df)
    out = gen.sample(4, column_order=["cat", "num", "const"])
    self.assertEqual(list(out.columns), ["num", "cat", "const"])
    with self.assertRaises(ValueError):
      gen.sample(4, column_order=["cat", "num"])  # not a full permutation

  def test_integer_numeric_columns_round_trip(self):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "count": rng.integers(0, 1000, size=24),
        "cat": np.where(rng.random(24) < 0.5, "a", "b"),
    })
    gen = TabFMDataGenerator(model=_tiny_model(), n_estimators=2,
                             random_state=0).fit(df)
    out = gen.sample(6)
    self.assertTrue(pd.api.types.is_integer_dtype(out["count"]))

  def test_numpy_input_with_conditional_numeric_column(self):
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=24), rng.integers(0, 2, size=24)]
    gen = TabFMDataGenerator(model=_tiny_model(), n_estimators=2,
                             random_state=0).fit(X)
    out = gen.sample(4, column_order=[1, 0])
    self.assertEqual(list(out.columns), [0, 1])
    self.assertEqual(len(out), 4)

  def test_missing_values_in_conditioned_columns(self):
    rng = np.random.default_rng(0)
    n = 40
    num = rng.normal(size=n)
    num[::3] = np.nan
    cat = np.where(rng.random(n) < 0.5, "a", "b").astype(object)
    cat[1::5] = None
    df = pd.DataFrame({"num": num, "cat": cat, "num2": rng.normal(size=n)})
    gen = TabFMDataGenerator(model=_tiny_model(), n_estimators=2,
                             random_state=0).fit(df)
    # A high temperature flattens the presence classifier, so both columns
    # get NaN rows whatever the untrained model outputs.
    out = gen.sample(30, t=100.0, column_order=["num2", "num", "cat"])
    self.assertFalse(out["num2"].isna().any())
    self.assertTrue(out["num"].isna().any())
    self.assertTrue(out["cat"].isna().any())
    self.assertTrue(set(out["cat"].dropna()) <= {"a", "b"})
    self.assertEqual(list(out.dtypes), list(df.dtypes))

  def test_sample_before_fit_raises(self):
    gen = TabFMDataGenerator(model=_tiny_model())
    with self.assertRaises(ValueError):
      gen.sample(4)


if __name__ == "__main__":
  unittest.main()

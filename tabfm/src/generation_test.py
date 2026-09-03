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

from absl.testing import absltest
import numpy as np
import pandas as pd

try:
  from flax import nnx
  from tabfm.src.jax import model as tabfm_jax_model
  HAS_JAX = True
except ImportError:
  HAS_JAX = False
from tabfm.src.generation import bin_atoms
from tabfm.src.generation import bin_values
from tabfm.src.generation import digits_of
from tabfm.src.generation import indices_from_digits
from tabfm.src.generation import quantile_edges
from tabfm.src.generation import sample_rows
from tabfm.src.generation import sample_within_bins
from tabfm.src.generation import TabFMDataGenerator
from tabfm.src.generation import temperature_scale


class _FakeModel:
  """Stands in for a TabFM model; fit() only reads max_classes."""

  max_classes = 10


class SamplingPrimitivesTest(absltest.TestCase):

  def test_temperature_scale_identity_at_t_1(self):
    probs = np.array([[0.2, 0.3, 0.5]])
    np.testing.assert_allclose(temperature_scale(probs, 1.0), probs)

  def test_temperature_scale_sharpens_below_1(self):
    probs = np.array([[0.4, 0.6]])
    sharp = temperature_scale(probs, 0.1)
    self.assertGreater(sharp[0, 1], 0.95)
    np.testing.assert_allclose(sharp.sum(axis=-1), 1.0)

  def test_temperature_scale_flattens_above_1(self):
    probs = np.array([[0.1, 0.9]])
    flat = temperature_scale(probs, 10.0)
    self.assertLess(flat[0, 1], 0.9)
    self.assertGreater(flat[0, 1], 0.5)
    np.testing.assert_allclose(flat.sum(axis=-1), 1.0)

  def test_temperature_scale_keeps_zeros_zero(self):
    probs = np.array([[0.0, 1.0]])
    scaled = temperature_scale(probs, 0.5)
    self.assertEqual(scaled[0, 0], 0.0)
    self.assertEqual(scaled[0, 1], 1.0)

  def test_sample_rows_respects_degenerate_probs(self):
    rng = np.random.default_rng(0)
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] * 5)
    drawn = sample_rows(probs, rng)
    np.testing.assert_array_equal(drawn, np.array([0, 2] * 5))

  def test_sample_rows_matches_distribution(self):
    rng = np.random.default_rng(0)
    probs = np.tile(np.array([[0.2, 0.8]]), (20000, 1))
    drawn = sample_rows(probs, rng)
    self.assertAlmostEqual(drawn.mean(), 0.8, delta=0.02)

  def test_quantile_edges_equal_mass(self):
    values = np.arange(100, dtype=float)
    edges = quantile_edges(values, 4)
    self.assertLen(edges, 5)
    self.assertEqual(edges[0], 0.0)
    self.assertEqual(edges[-1], 99.0)

  def test_quantile_edges_collapses_duplicates_on_skewed_data(self):
    values = np.array([0.0] * 90 + [1.0] * 10)
    edges = quantile_edges(values, 10)
    self.assertLen(edges, len(np.unique(edges)))
    self.assertLess(len(edges), 11)

  def test_bin_values_covers_range_inclusively(self):
    edges = np.array([0.0, 1.0, 2.0])
    ids = bin_values(np.array([0.0, 0.5, 1.0, 2.0]), edges)
    np.testing.assert_array_equal(ids, np.array([0, 0, 1, 1]))

  def test_sample_within_bins_stays_inside(self):
    rng = np.random.default_rng(0)
    edges = np.array([0.0, 1.0, 10.0])
    ids = np.array([0, 1] * 50)
    vals = sample_within_bins(ids, edges, rng)
    self.assertTrue(np.all(vals[0::2] >= 0.0) and np.all(vals[0::2] <= 1.0))
    self.assertTrue(np.all(vals[1::2] >= 1.0) and np.all(vals[1::2] <= 10.0))

  def test_bin_atoms_flags_repeated_values_only(self):
    values = np.array([0.0, 0.0, 0.0, 0.5, 1.5, 2.5])
    ids = np.array([0, 0, 0, 0, 1, 1])
    atom_value, atom_frac = bin_atoms(values, ids, n_bins=3)
    self.assertEqual(atom_value[0], 0.0)
    self.assertAlmostEqual(atom_frac[0], 0.75)
    self.assertTrue(np.isnan(atom_value[1]))
    self.assertEqual(atom_frac[1], 0.0)
    self.assertTrue(np.isnan(atom_value[2]))  # empty bin
    self.assertEqual(atom_frac[2], 0.0)

  def test_sample_within_bins_returns_atoms_with_their_share(self):
    rng = np.random.default_rng(0)
    edges = np.array([0.0, 1.0, 10.0])
    ids = np.array([0, 1] * 2000)
    atom_value = np.array([0.0, np.nan])
    atom_frac = np.array([0.6, 0.0])
    vals = sample_within_bins(ids, edges, rng, atom_value, atom_frac)
    self.assertAlmostEqual((vals[0::2] == 0.0).mean(), 0.6, delta=0.03)
    self.assertTrue(np.all(vals[1::2] > 1.0) and np.all(vals[1::2] <= 10.0))

  def test_digit_round_trip(self):
    idx = np.arange(1000)
    digits = digits_of(idx, base=10, n_digits=3)
    self.assertEqual(digits.shape, (1000, 3))
    self.assertTrue(np.all(digits >= 0) and np.all(digits < 10))
    np.testing.assert_array_equal(indices_from_digits(digits, base=10), idx)

  def test_digits_most_significant_first(self):
    digits = digits_of(np.array([472]), base=10, n_digits=3)
    np.testing.assert_array_equal(digits[0], np.array([4, 7, 2]))


class FitColumnSpecTest(absltest.TestCase):

  def _fit(self, df, **kwargs):
    gen = TabFMDataGenerator(model=_FakeModel(), random_state=0)
    return gen.fit(df, **kwargs)

  def _spec(self, gen, name):
    return next(s for s in gen.columns_ if s.name == name)

  def test_infers_kinds_from_dtypes(self):
    df = pd.DataFrame({
        "num": np.linspace(0.0, 1.0, 40),
        "cat": ["a", "b"] * 20,
        "const": [7.0] * 40,
    })
    gen = self._fit(df)
    self.assertEqual(self._spec(gen, "num").kind, "numeric")
    self.assertEqual(self._spec(gen, "cat").kind, "categorical")
    self.assertEqual(self._spec(gen, "const").kind, "constant")
    self.assertEqual(gen.feature_names_, ["num", "cat", "const"])

  def test_numeric_base_capped_by_max_classes(self):
    class TinyModel:
      max_classes = 3

    df = pd.DataFrame({"num": np.linspace(0.0, 1.0, 40),
                       "num2": np.linspace(0.0, 1.0, 40)})
    gen = TabFMDataGenerator(model=TinyModel(), random_state=0).fit(df)
    spec = self._spec(gen, "num")
    self.assertEqual(spec.base, 3)
    self.assertLessEqual(len(spec.edges) - 1, 3**2)  # n_levels=2 default

  def test_hierarchical_levels_and_single_level_fallback(self):
    df = pd.DataFrame({"num": np.linspace(0.0, 1.0, 400),
                       "cat": ["a", "b"] * 200})
    gen = TabFMDataGenerator(model=_FakeModel(), n_levels=2,
                             random_state=0).fit(df)
    spec = self._spec(gen, "num")
    self.assertLen(spec.edges, 101)  # 10^2 fine bins
    self.assertEqual(spec.n_digits, 2)
    gen1 = TabFMDataGenerator(model=_FakeModel(), n_levels=1,
                              random_state=0).fit(df)
    self.assertEqual(gen1.columns_[0].n_digits, 1)
    self.assertLessEqual(len(gen1.columns_[0].edges) - 1, 10)

  def test_low_cardinality_numeric_becomes_categorical(self):
    df = pd.DataFrame({"few": [1.5, 2.5, 3.5] * 10,
                       "num": np.linspace(0.0, 1.0, 30)})
    gen = self._fit(df)
    spec = self._spec(gen, "few")
    self.assertEqual(spec.kind, "categorical")
    np.testing.assert_array_equal(np.sort(spec.categories),
                                  np.array([1.5, 2.5, 3.5]))

  def test_explicit_categorical_features_override(self):
    df = pd.DataFrame({"code": np.arange(40) % 15,
                       "num": np.linspace(0.0, 1.0, 40)})
    gen = self._fit(df, categorical_features=["code"])
    self.assertEqual(self._spec(gen, "code").kind, "categorical")

  def test_high_cardinality_categorical_merges_tail(self):
    labels = (["common%d" % i for i in range(9) for _ in range(10)]
              + ["rare%d" % i for i in range(6)])
    df = pd.DataFrame({"cat": labels, "num": np.linspace(0, 1, len(labels))})
    gen = self._fit(df)
    spec = self._spec(gen, "cat")
    self.assertLen(spec.categories, 9)  # max_classes - 1 top classes
    self.assertLen(spec.other, 6)
    np.testing.assert_allclose(spec.other_freqs.sum(), 1.0)

  def test_mixed_type_object_column_is_categorical(self):
    df = pd.DataFrame({"mixed": ["a", 1, "b", 2] * 6,
                       "num": np.linspace(0.0, 1.0, 24)})
    spec = self._spec(self._fit(df), "mixed")
    self.assertEqual(spec.kind, "categorical")
    self.assertCountEqual(spec.categories.tolist(), ["a", 1, "b", 2])

  def test_numeric_spec_records_point_masses(self):
    df = pd.DataFrame({"spike": [0.0] * 60 + list(np.linspace(1.0, 2.0, 40)),
                       "cat": ["a", "b"] * 50})
    spec = self._spec(self._fit(df), "spike")
    self.assertEqual(spec.kind, "numeric")
    self.assertEqual(spec.atom_value[0], 0.0)
    self.assertGreater(spec.atom_frac[0], 0.9)
    self.assertEqual(np.count_nonzero(spec.atom_frac), 1)

  def test_nan_rows_excluded_from_values(self):
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0, 4.0] * 10,
                       "cat": ["a", "b", "a", "b"] * 10})
    gen = self._fit(df)
    self.assertLen(self._spec(gen, "num").values, 30)


class MarginalSamplingTest(absltest.TestCase):

  def _gen(self, df, **kwargs):
    return TabFMDataGenerator(model=_FakeModel(), random_state=0).fit(
        df, **kwargs)

  def _spec(self, gen, name):
    return next(s for s in gen.columns_ if s.name == name)

  def test_constant_marginal(self):
    df = pd.DataFrame({"const": [7.0] * 10, "num": np.linspace(0, 1, 10)})
    gen = self._gen(df)
    rng = np.random.default_rng(0)
    out = gen._sample_marginal(self._spec(gen, "const"), 5, 1.0, rng)
    np.testing.assert_array_equal(out, np.full(5, 7.0))

  def test_categorical_marginal_matches_frequencies(self):
    df = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 20,
                       "num": np.linspace(0, 1, 100)})
    gen = self._gen(df)
    rng = np.random.default_rng(0)
    out = gen._sample_marginal(self._spec(gen, "cat"), 5000, 1.0, rng)
    self.assertAlmostEqual((out == "a").mean(), 0.8, delta=0.03)

  def test_numeric_marginal_within_range(self):
    df = pd.DataFrame({"num": np.linspace(-5.0, 5.0, 100),
                       "cat": ["a", "b"] * 50})
    gen = self._gen(df)
    rng = np.random.default_rng(0)
    out = gen._sample_marginal(self._spec(gen, "num"), 200, 1.0, rng)
    self.assertTrue(np.all(out >= -5.0) and np.all(out <= 5.0))
    self.assertGreater(len(np.unique(out)), 100)  # fresh values, not copies

  def test_numeric_marginal_keeps_point_mass(self):
    rng = np.random.default_rng(0)
    spike = np.concatenate([np.zeros(150), rng.exponential(size=50)])
    df = pd.DataFrame({"spike": spike, "cat": ["a", "b"] * 100})
    gen = self._gen(df)
    out = gen._sample_marginal(self._spec(gen, "spike"), 2000, 1.0,
                               np.random.default_rng(0))
    self.assertAlmostEqual((out == 0.0).mean(), 0.75, delta=0.04)
    self.assertGreater(len(np.unique(out[out > 0])), 100)

  def test_encode_decode_round_trip_with_tail(self):
    labels = (["common%d" % i for i in range(9) for _ in range(10)]
              + ["rare%d" % i for i in range(6)])
    df = pd.DataFrame({"cat": labels, "num": np.linspace(0, 1, len(labels))})
    gen = self._gen(df)
    spec = self._spec(gen, "cat")
    codes = gen._encode_target(spec, spec.values)
    self.assertEqual(codes.max(), len(spec.categories))  # tail code present
    rng = np.random.default_rng(0)
    decoded = gen._decode_categorical(spec, codes, rng)
    top = codes < len(spec.categories)
    np.testing.assert_array_equal(decoded[top], spec.values[top])
    for v in decoded[~top]:
      self.assertIn(v, set(spec.other.tolist()))

  def test_marginal_temperature_zero_ish_picks_mode(self):
    df = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 20,
                       "num": np.linspace(0, 1, 100)})
    gen = self._gen(df)
    rng = np.random.default_rng(0)
    out = gen._sample_marginal(self._spec(gen, "cat"), 50, 1e-6, rng)
    self.assertTrue(np.all(out == "a"))


class SampleWithoutModelTest(absltest.TestCase):
  """sample() paths that need no model call: marginals and validation."""

  def _gen(self, df):
    return TabFMDataGenerator(model=_FakeModel(), random_state=0).fit(df)

  def test_rejects_non_positive_or_non_finite_temperature(self):
    gen = self._gen(pd.DataFrame({"num": np.linspace(0.0, 1.0, 20),
                                  "const": [1.0] * 20}))
    for t in (0.0, -1.0, float("nan"), float("inf")):
      with self.assertRaises(ValueError):
        gen.sample(4, t=t)

  def test_missing_values_are_reproduced(self):
    num = np.linspace(0.0, 1.0, 1000)
    num[::4] = np.nan
    gen = self._gen(pd.DataFrame({"num": num, "const": [1.0] * 1000}))
    out = gen.sample(2000)
    self.assertAlmostEqual(out["num"].isna().mean(), 0.25, delta=0.03)
    self.assertFalse(out["const"].isna().any())

  def test_all_nan_column_samples_nan(self):
    gen = self._gen(pd.DataFrame({"empty": [np.nan] * 20,
                                  "num": np.linspace(0.0, 1.0, 20)}))
    out = gen.sample(5)
    self.assertTrue(out["empty"].isna().all())
    self.assertEqual(out["empty"].dtype, np.float64)
    self.assertFalse(out["num"].isna().any())

  def test_successive_calls_continue_the_random_stream(self):
    df = pd.DataFrame({"num": np.linspace(0.0, 1.0, 50), "const": [1.0] * 50})
    gen = self._gen(df)
    first, second = gen.sample(10), gen.sample(10)
    self.assertFalse(first.equals(second))
    pd.testing.assert_frame_equal(self._gen(df).sample(10), first)


@unittest.skipUnless(HAS_JAX, "JAX backend not installed")
class JaxEndToEndTest(absltest.TestCase):

  def test_sample_with_jax_backend(self):
    model = tabfm_jax_model.TabFM(
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
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "num": rng.normal(size=16),
        "cat": np.where(rng.random(16) < 0.5, "a", "b"),
    })
    gen = TabFMDataGenerator(model=model, n_estimators=2,
                             random_state=0).fit(df)
    out = gen.sample(n_samples=4)
    self.assertEqual(list(out.columns), ["num", "cat"])
    self.assertLen(out, 4)
    self.assertContainsSubset(set(out["cat"]), {"a", "b"})


if __name__ == "__main__":
  absltest.main()

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

"""Synthetic tabular data generation on top of TabFM.

Generates new rows that mimic a reference dataset by factorizing the joint
distribution over columns with the chain rule,
p(x_1, ..., x_d) = prod_j p(x_j | x_{<j}), and sampling each column in turn
from a TabFM classifier's predictive distribution. Numerical columns reach
fine resolution despite the model's max_classes limit via hierarchical
quantile refinement: the column is split into n_bins ** n_levels equal-mass
bins and the bin index is sampled digit by digit in base n_bins, with the
already-sampled digits appended to the conditioning features. A value is then
drawn uniformly within the sampled bin, so the per-column conditionals form a
piecewise-uniform density. Only the classification model is used; the
regression head emits a point estimate and is never needed here.
"""

import dataclasses
from typing import Any, List, Optional

import numpy as np
import pandas as pd

import jaxtyping as jt
import typeguard

from tabfm.src.classifier_and_regressor import TabFMClassifier

jt.typed = jt.jaxtyped(typechecker=typeguard.typechecked)

# pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# Sampling primitives
# ---------------------------------------------------------------------------


@jt.typed
def temperature_scale(
    probs: jt.Float[np.ndarray, "N K"], t: float
) -> jt.Float[np.ndarray, "N K"]:
  """Sharpens (t < 1) or flattens (t > 1) row-wise probabilities."""
  if t == 1.0:
    return probs
  with np.errstate(divide="ignore"):
    logits = np.log(probs) / t  # zeros -> -inf, stay zero after re-softmax
  logits -= logits.max(axis=-1, keepdims=True)
  scaled = np.exp(logits)
  return scaled / scaled.sum(axis=-1, keepdims=True)


@jt.typed
def sample_rows(
    probs: jt.Float[np.ndarray, "N K"], rng: np.random.Generator
) -> jt.Int[np.ndarray, "N"]:
  """Draws one class index per row from row-wise probabilities."""
  cdf = np.cumsum(probs, axis=-1)
  cdf /= cdf[:, -1:]  # guard against floating-point drift
  u = rng.random((probs.shape[0], 1))
  return (u > cdf).sum(axis=-1)


@jt.typed
def quantile_edges(
    values: jt.Float[np.ndarray, "N"], n_bins: int
) -> jt.Float[np.ndarray, "E"]:
  """Equal-mass bin edges; duplicate quantiles collapse for skewed data."""
  qs = np.linspace(0.0, 1.0, n_bins + 1)
  return np.unique(np.quantile(values, qs))


@jt.typed
def bin_values(
    values: jt.Float[np.ndarray, "N"], edges: jt.Float[np.ndarray, "E"]
) -> jt.Int[np.ndarray, "N"]:
  """Assigns each value the id of its bin, in [0, len(edges) - 2]."""
  ids = np.searchsorted(edges, values, side="right") - 1
  return np.clip(ids, 0, len(edges) - 2)


@jt.typed
def sample_within_bins(
    bin_ids: jt.Int[np.ndarray, "N"],
    edges: jt.Float[np.ndarray, "E"],
    rng: np.random.Generator,
) -> jt.Float[np.ndarray, "N"]:
  """Draws uniformly inside each row's bin."""
  lo = edges[bin_ids]
  hi = edges[bin_ids + 1]
  return lo + rng.random(len(bin_ids)) * (hi - lo)


@jt.typed
def digits_of(
    indices: jt.Int[np.ndarray, "N"], base: int, n_digits: int
) -> jt.Int[np.ndarray, "N D"]:
  """Decomposes indices into base-`base` digits, most significant first."""
  out = np.empty((len(indices), n_digits), dtype=np.int64)
  rest = indices.astype(np.int64)
  for d in range(n_digits - 1, -1, -1):
    out[:, d] = rest % base
    rest = rest // base
  return out


@jt.typed
def indices_from_digits(
    digits: jt.Int[np.ndarray, "N D"], base: int
) -> jt.Int[np.ndarray, "N"]:
  """Recomposes indices from base-`base` digits (inverse of digits_of)."""
  out = np.zeros(len(digits), dtype=np.int64)
  for d in range(digits.shape[1]):
    out = out * base + digits[:, d]
  return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ColumnSpec:
  """Per-column sampling strategy decided at fit() time."""

  name: str
  kind: str  # "constant" | "categorical" | "numeric"
  dtype: Any
  values: np.ndarray  # NaN-free reference values for this column
  categories: Optional[np.ndarray] = None  # categorical: modeled classes
  edges: Optional[np.ndarray] = None  # numeric: fine quantile bin edges
  base: Optional[int] = None  # numeric: classes per refinement level
  n_digits: Optional[int] = None  # numeric: refinement levels
  other: Optional[np.ndarray] = None  # categorical: merged tail classes
  other_freqs: Optional[np.ndarray] = None  # empirical freqs of the tail


class TabFMDataGenerator:
  """Samples synthetic rows that mimic a reference DataFrame using TabFM.

  Attributes:
    X_: Reference DataFrame stored at fit() time.
    feature_names_: Column names of the reference data, in original order.
    columns_: Fitted per-column ``ColumnSpec`` strategies.
  """

  X_: pd.DataFrame
  feature_names_: List[str]
  columns_: List[ColumnSpec]

  def __init__(
      self,
      model: Any,
      n_bins: int = 10,
      n_levels: int = 2,
      n_estimators: int = 4,
      random_state: Optional[int] = None,
  ):
    """Initialises the generator.

    Args:
      model: Pre-trained TabFM classification model (NNX or PyTorch module).
      n_bins: Classes per refinement level for numerical columns, capped at
        the model's ``max_classes``.
      n_levels: Refinement levels; numerical columns are sampled over up to
        ``n_bins ** n_levels`` equal-mass quantile bins.
      n_estimators: Ensemble members for each internal ``TabFMClassifier``.
      random_state: Seed for column-order and sampling randomness.
    """
    self.model = model
    self.n_bins = n_bins
    self.n_levels = n_levels
    self.n_estimators = n_estimators
    self.random_state = random_state

  def fit(
      self, X: Any, categorical_features: Optional[List[str]] = None
  ) -> "TabFMDataGenerator":
    """Stores the reference data and decides each column's strategy.

    No model call happens here; like ``TabFMClassifier.fit``, this only
    prepares metadata. A column is treated as categorical if its dtype is
    object/category/bool, it is listed in ``categorical_features``, or it has
    at most ``min(n_bins, model.max_classes)`` unique values.

    Args:
      X: Reference data of shape (n_samples, n_features).
      categorical_features: Optional names of columns to force categorical.

    Returns:
      self.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    if X.shape[1] == 0 or X.shape[0] < 2:
      raise ValueError("fit() needs at least 2 rows and 1 column.")
    max_classes = int(getattr(self.model, "max_classes", 10))
    n_bins = min(self.n_bins, max_classes)
    categorical_features = set(categorical_features or [])

    self.X_ = X
    self.feature_names_ = list(X.columns)
    self.columns_ = []
    for name in self.feature_names_:
      values = X[name].dropna().to_numpy()
      uniques, counts = np.unique(values, return_counts=True)
      dtype = X[name].dtype
      # Everything non-numeric (object, string, category, ...) plus bool is
      # sampled as categorical; numeric columns may still fold into the
      # categorical path below when their cardinality is low enough.
      is_cat_dtype = not pd.api.types.is_numeric_dtype(dtype) or dtype == bool
      if len(uniques) <= 1:
        spec = ColumnSpec(name=name, kind="constant", dtype=dtype,
                          values=values)
      elif (is_cat_dtype or name in categorical_features
            or len(uniques) <= n_bins):
        if len(uniques) <= max_classes:
          spec = ColumnSpec(name=name, kind="categorical", dtype=dtype,
                            values=values, categories=uniques)
        else:
          # More classes than the model supports: model the most frequent
          # ones directly and merge the tail into one class that is
          # re-sampled from its empirical frequencies when drawn.
          order = np.argsort(counts)[::-1]
          top, tail = order[: max_classes - 1], order[max_classes - 1 :]
          tail_counts = counts[tail].astype(float)
          spec = ColumnSpec(name=name, kind="categorical", dtype=dtype,
                            values=values, categories=uniques[top],
                            other=uniques[tail],
                            other_freqs=tail_counts / tail_counts.sum())
      else:
        edges = quantile_edges(values.astype(float), n_bins ** self.n_levels)
        n_fine = len(edges) - 1
        # Digits needed to index the fine bins in base n_bins; duplicate
        # quantiles on skewed data may shrink n_fine below the full power.
        n_digits = max(1, int(np.ceil(np.log(n_fine) / np.log(n_bins))))
        spec = ColumnSpec(name=name, kind="numeric", dtype=dtype,
                          values=values, edges=edges, base=n_bins,
                          n_digits=n_digits)
      self.columns_.append(spec)
    return self

  def sample(
      self,
      n_samples: int,
      t: float = 1.0,
      column_order: Optional[List[str]] = None,
  ) -> pd.DataFrame:
    """Generates new synthetic rows mimicking the reference data.

    Columns are visited in ``column_order`` (or a seeded random permutation)
    and sampled via the chain rule: the first non-constant column from its
    empirical marginal, each later column from a TabFM classifier conditioned
    on the columns already sampled. Constant columns never join the
    conditioning set.

    Args:
      n_samples: Number of synthetic rows to generate.
      t: Sampling temperature; < 1 concentrates near the modes of the
        reference data, > 1 flattens the sampled distributions.
      column_order: Optional explicit visitation order; must be a permutation
        of the fitted column names.

    Returns:
      DataFrame of shape (n_samples, n_features) in the original column
      order, with dtypes matching the reference data.
    """
    if not hasattr(self, "columns_"):
      raise ValueError(
          "This TabFMDataGenerator is not fitted yet; call fit(X) first."
      )
    if column_order is not None and sorted(column_order) != sorted(
        self.feature_names_
    ):
      raise ValueError(
          "column_order must be a permutation of the fitted columns "
          f"{self.feature_names_}, got {column_order}."
      )
    rng = np.random.default_rng(self.random_state)
    specs = {s.name: s for s in self.columns_}
    if column_order is not None:
      order = list(column_order)
    else:
      order = [self.feature_names_[i]
               for i in rng.permutation(len(self.feature_names_))]

    synth = pd.DataFrame(index=range(n_samples))
    conditioning = []
    for name in order:
      spec = specs[name]
      if spec.kind == "constant" or not conditioning:
        col = self._sample_marginal(spec, n_samples, t, rng)
      else:
        col = self._sample_conditional(spec, synth[conditioning], t, rng)
      if spec.kind == "numeric" and pd.api.types.is_integer_dtype(spec.dtype):
        col = np.round(col)
      synth[name] = col
      if spec.kind != "constant":
        conditioning.append(name)
    # Restore the reference dtypes (categoricals were sampled as object,
    # integer-dtype numerics as rounded floats).
    dtypes = {s.name: s.dtype for s in self.columns_}
    return synth[self.feature_names_].astype(dtypes)

  def _encode_target(self, spec: ColumnSpec, values: np.ndarray) -> np.ndarray:
    """Maps column values to integer class codes for classifier fitting."""
    code_map = {v: i for i, v in enumerate(spec.categories)}
    codes = pd.Series(values).map(code_map)
    # Values outside `categories` are the merged tail; they share one code.
    return codes.fillna(len(spec.categories)).to_numpy(dtype=np.int64)

  def _decode_categorical(
      self, spec: ColumnSpec, codes: np.ndarray, rng: np.random.Generator
  ) -> np.ndarray:
    """Maps sampled codes back to values; the tail code draws empirically."""
    out = np.empty(len(codes), dtype=object)
    top = codes < len(spec.categories)
    out[top] = spec.categories[codes[top]]
    n_other = int((~top).sum())
    if n_other:
      out[~top] = rng.choice(spec.other, size=n_other, p=spec.other_freqs)
    return out

  def _sample_marginal(
      self, spec: ColumnSpec, n_samples: int, t: float,
      rng: np.random.Generator
  ) -> np.ndarray:
    """Model-free draw from the column's empirical marginal distribution."""
    if spec.kind == "constant":
      return np.full(n_samples, spec.values[0])
    if spec.kind == "categorical":
      codes = self._encode_target(spec, spec.values)
      n_codes = len(spec.categories) + (1 if spec.other is not None else 0)
      counts = np.bincount(codes, minlength=n_codes).astype(float)
      probs = temperature_scale((counts / counts.sum())[None, :], t)[0]
      drawn = rng.choice(len(probs), size=n_samples, p=probs)
      return self._decode_categorical(spec, drawn, rng)
    ids = bin_values(spec.values.astype(float), spec.edges)
    counts = np.bincount(ids, minlength=len(spec.edges) - 1).astype(float)
    probs = temperature_scale((counts / counts.sum())[None, :], t)[0]
    drawn = rng.choice(len(probs), size=n_samples, p=probs)
    return sample_within_bins(drawn, spec.edges, rng)

  def _classify_and_sample(
      self,
      x_ref: pd.DataFrame,
      y_train: np.ndarray,
      x_synth: pd.DataFrame,
      t: float,
      rng: np.random.Generator,
  ) -> np.ndarray:
    """Fits a fresh TabFMClassifier and samples one class per synthetic row."""
    clf = TabFMClassifier(
        model=self.model,
        n_estimators=self.n_estimators,
        random_state=int(rng.integers(2**31 - 1)),
    )
    clf.fit(x_ref, y_train)
    probs = np.asarray(clf.predict_proba(x_synth)).astype(np.float64)
    probs = probs[:, : len(clf.classes_)]
    probs = probs / probs.sum(axis=-1, keepdims=True)
    probs = temperature_scale(probs, t)
    return np.asarray(clf.classes_)[sample_rows(probs, rng)]

  def _sample_conditional(
      self,
      spec: ColumnSpec,
      x_cond_synth: pd.DataFrame,
      t: float,
      rng: np.random.Generator,
  ) -> np.ndarray:
    """Samples one column for all synthetic rows given the prior columns.

    Categorical columns need a single classifier fit, with the integer class
    codes as the target. Numeric columns sample their fine-bin index digit by
    digit: level l predicts base-`spec.base` digit l with digits 0..l-1
    appended to the conditioning features, so every level trains on all
    reference rows while resolution grows as base ** levels.
    """
    mask = self.X_[spec.name].notna().to_numpy()
    x_cond_ref = self.X_.loc[mask, list(x_cond_synth.columns)]
    if spec.kind == "categorical":
      y_train = self._encode_target(spec, spec.values)
      if len(np.unique(y_train)) < 2:
        return self._sample_marginal(spec, len(x_cond_synth), t, rng)
      drawn = self._classify_and_sample(x_cond_ref, y_train, x_cond_synth,
                                        t, rng)
      return self._decode_categorical(spec, drawn, rng)

    fine_ids = bin_values(spec.values.astype(float), spec.edges)
    ref_digits = digits_of(fine_ids, spec.base, spec.n_digits)
    synth_digits = np.zeros((len(x_cond_synth), spec.n_digits), dtype=np.int64)
    x_ref = x_cond_ref.reset_index(drop=True)
    x_synth = x_cond_synth.reset_index(drop=True)
    for level in range(spec.n_digits):
      y_level = ref_digits[:, level]
      if len(np.unique(y_level)) < 2:
        synth_digits[:, level] = y_level[0]
      else:
        drawn = self._classify_and_sample(x_ref, y_level, x_synth, t, rng)
        # classes_ round-trips through the label encoder, which may widen
        # the integer digits to float; restore ints.
        synth_digits[:, level] = drawn.astype(np.int64)
      digit_col = f"_digit_{level}"
      while digit_col in x_ref.columns:  # avoid clobbering a real column
        digit_col += "_"
      x_ref = x_ref.assign(**{digit_col: ref_digits[:, level]})
      x_synth = x_synth.assign(**{digit_col: synth_digits[:, level]})
    drawn_ids = indices_from_digits(synth_digits, spec.base)
    # Digit combinations past the last bin can only arise when duplicate
    # quantiles shrank the bin count below a full power of base; clamp them.
    drawn_ids = np.minimum(drawn_ids, len(spec.edges) - 2)
    return sample_within_bins(drawn_ids, spec.edges, rng)

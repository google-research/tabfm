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

"""Classifier and Regressor interfaces for TabFM.

This module provides scikit-learn compatible classifier and regressor classes
built on top of the TabFM (Tabular Foundation Model) foundation model, as
well as the data preprocessing and ensemble-generation utilities they rely on.

Key classes:
  - CategoricalOrdinalEncoder: Ordinal encoding by appearance or frequency.
  - TransformToNumerical: Mixed-type DataFrame --> numeric array pipeline.
  - EnsembleGenerator: Creates diverse data views for ensemble inference.
  - TabFMClassifier: sklearn-compatible TabFM classifier.
  - TabFMRegressor:  sklearn-compatible TabFM regressor.
"""

import argparse
import collections
import itertools
import math
import os
import random
import sys
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import flags, logging
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

import jaxtyping as jt; import typeguard; import numpy as np; jt.typed = jt.jaxtyped(typechecker=typeguard.typechecked)


# ---------------------------------------------------------------------------
# Preprocessing utilities
# ---------------------------------------------------------------------------


class CategoricalOrdinalEncoder(BaseEstimator, TransformerMixin):
  """Ordinal Encoder that assigns indices based on order of appearance or frequency.

  Args:
    dtype: Output dtype for encoded values.
    handle_unknown: How to handle unknown categories at transform time. Only
      ``"use_encoded_value"`` is currently supported.
    unknown_value: Encoded value used for unknown categories.
    encoded_missing_value: Encoded value used for NaN / missing values.
    min_frequency: Categories appearing fewer than this many times in the
      training data are treated as unknown.
    mode: Encoding order strategy.
      - ``"appearance"``: categories are ordered by first occurrence in the
        data (original behaviour).
      - ``"frequency"``: categories are sorted by descending frequency, so
        the most common category receives index 0.
      - ``"alphabetical"``: categories are sorted ascending (alphabetically),
        matching scikit-learn's ``LabelEncoder`` convention.
  """

  categories_: List[np.ndarray]

  def __init__(
      self,
      dtype=np.float64,
      handle_unknown: str = "use_encoded_value",
      unknown_value: int = -1,
      encoded_missing_value: int = -1,
      min_frequency: int = 1,
      mode: str = "appearance",
  ):
    self.dtype = dtype
    self.handle_unknown = handle_unknown
    self.unknown_value = unknown_value
    self.encoded_missing_value = encoded_missing_value
    self.min_frequency = min_frequency
    self.mode = mode

  def fit(self, X: Any, y: Any = None) -> "CategoricalOrdinalEncoder":
    """Fits the encoder to the data.

    Args:
      X: Input array-like of shape (n_samples, n_features).
      y: Ignored. Kept for sklearn compatibility.

    Returns:
      self.
    """
    if hasattr(X, "iloc"):
      X = X.values
    self.categories_ = []
    n_features = X.shape[1]

    for i in range(n_features):
      col = X[:, i]

      # Count occurrences to filter rare categories
      counts = pd.Series(col).value_counts()
      rare_cats = counts[counts < self.min_frequency].index

      # Determine the candidate categories in the order dictated by the mode.
      if self.mode == "frequency":
        # Descending frequency. value_counts is already frequency-sorted, with
        # ties broken by appearance order via pandas' stable sort.
        candidates = counts.index.tolist()
      elif self.mode == "appearance":
        # Order of first appearance.
        candidates = pd.unique(col)
      elif self.mode == "alphabetical":
        # Order of first appearance; sorted below once NaNs are removed.
        candidates = pd.unique(col)
      else:
        raise ValueError(
            f"Unknown mode: {self.mode!r}. Expected one of 'appearance', "
            "'alphabetical', or 'frequency'."
        )

      # Drop NaNs, the literal string "nan" (to match TF behavior), and rare
      # categories.
      uniques = [
          u
          for u in candidates
          if not pd.isna(u) and str(u) != "nan" and u not in rare_cats
      ]
      if self.mode == "alphabetical":
        # Sort categories so encoding matches sklearn's LabelEncoder convention
        # (classes ordered ascending / alphabetically).
        uniques = sorted(uniques)

      self.categories_.append(np.array(uniques))

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transforms the data using the fitted encoder.

    Args:
      X: Input array-like of shape (n_samples, n_features).

    Returns:
      Encoded array of shape (n_samples, n_features) with dtype ``self.dtype``.
    """
    check_is_fitted(self)
    if hasattr(X, "iloc"):
      X = X.values
    n_samples, n_features = X.shape
    X_out = np.full(
        (n_samples, n_features), self.unknown_value, dtype=self.dtype
    )

    for i in range(n_features):
      col = X[:, i]
      cats = self.categories_[i]
      cat_to_idx = {c: idx for idx, c in enumerate(cats)}
      mapped = pd.Series(col).map(cat_to_idx)

      # Convert to numeric to avoid Categorical/fillna issues
      if mapped.dtype.name == "category":
        mapped = mapped.astype(object)

      # Fill NaNs/Unknowns with unknown_value
      X_out[:, i] = mapped.fillna(self.unknown_value).values.astype(self.dtype)

    return X_out

  def inverse_transform(self, X: np.ndarray) -> np.ndarray:
    """Decodes indices back to original categorical values.

    Args:
      X: Encoded array of shape (n_samples, n_features).

    Returns:
      Decoded object array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    if hasattr(X, "iloc"):
      X = X.values
    n_samples, n_features = X.shape
    X_out = np.empty((n_samples, n_features), dtype=object)

    for i in range(n_features):
      col = X[:, i]
      cats = self.categories_[i]
      valid_mask = col != self.unknown_value
      if np.any(valid_mask):
        indices = col[valid_mask].astype(int)
        X_out[valid_mask, i] = cats[indices]

    return X_out


def check_if_datetime_as_object(X: pd.Series) -> bool:
  """Checks if a pandas Series contains datetime information stored as objects.

  Args:
    X: A pandas Series whose dtype may or may not be object.

  Returns:
    True if the series looks like datetime data stored as object dtype.
  """
  if not pd.api.types.is_object_dtype(X.dtype):
    return False
  if X.isnull().all():
    return False
  try:
    pd.to_numeric(X)
  except (ValueError, TypeError):
    try:
      if len(X) > 500:
        X = X.sample(n=500, random_state=0)
      result = pd.to_datetime(X, errors="coerce", format="mixed")
      if result.isnull().mean() > 0.8:
        return False
      return True
    except Exception:  # pylint: disable=broad-except
      return False
  return False


class DatetimeTransformer(BaseEstimator, TransformerMixin):
  """Transformer that converts raw datetime columns into numeric features.

  Each datetime column is expanded into derived features (year, month, day,
  dayofweek) in addition to its Unix-nanosecond integer representation.
  """

  features_in: List[str]
  _fillna_map: Dict[str, Any]

  def __init__(self, features: Optional[List[str]] = None):
    """Initialises the transformer.

    Args:
      features: List of datetime sub-fields to extract per column.
        Defaults to ``['year', 'month', 'day', 'dayofweek']``.
    """
    if features is None:
      self.features = ["year", "month", "day", "dayofweek"]
    else:
      self.features = features
    self.features_in = []
    self._fillna_map = {}

  def fit(self, X: Any, y: Any = None) -> "DatetimeTransformer":
    """Fits the transformer, recording fill-in values for missing datetimes.

    Args:
      X: DataFrame or array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    if not isinstance(X, pd.DataFrame):
      X = pd.DataFrame(X)
    self.features_in = list(X.columns)
    for feature in self.features_in:
      series = pd.to_datetime(X[feature], utc=True, errors="coerce", format="mixed")
      self._fillna_map[feature] = series.mean()
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transforms datetime columns into numeric arrays.

    Args:
      X: DataFrame or array-like of shape (n_samples, n_features).

    Returns:
      Numeric array of shape (n_samples, n_features * (1 + len(self.features))).
    """
    if not isinstance(X, pd.DataFrame):
      X = pd.DataFrame(X)
    X_datetime = pd.DataFrame(index=X.index)
    for feature in self.features_in:
      series = pd.to_datetime(X[feature].copy(), utc=True, errors="coerce", format="mixed")
      broken_idx = series[(series == "NaT") | series.isna() | series.isnull()].index
      if len(broken_idx) > 0:
        series.loc[broken_idx] = self._fillna_map[feature]
      X_datetime[feature] = pd.to_numeric(series)
      for dt_feature in self.features:
        X_datetime[feature + "." + dt_feature] = getattr(
            series.dt, dt_feature
        ).astype(np.int64)
    return X_datetime.values


class TransformToNumerical(TransformerMixin, BaseEstimator):
  """Transforms non-numerical data in a DataFrame to numerical representations.

  Automatically detects categorical, datetime, and numeric columns, applying
  appropriate encoding for each type via a sklearn ``ColumnTransformer``.
  Falls through to an identity transform for non-DataFrame inputs.

  Attributes:
    tfm_: The fitted ``ColumnTransformer`` (or ``FunctionTransformer`` for
      non-DataFrame inputs).
  """

  tfm_: Union[ColumnTransformer, FunctionTransformer]

  def __init__(
      self,
      verbose: bool = False,
      cat_encoder_mode: str = "appearance",
      min_cat_frequency: int = 2,
  ):
    """Initialises the transformer.

    Args:
      verbose: Whether to print column-classification information.
      cat_encoder_mode: Encoding mode passed to ``CategoricalOrdinalEncoder``
        (``"appearance"`` or ``"frequency"``).
      min_cat_frequency: Minimum frequency for a category to be kept distinct;
        rarer categories are encoded as unknown.
    """
    self.verbose = verbose
    self.cat_encoder_mode = cat_encoder_mode
    self.min_cat_frequency = min_cat_frequency

  def fit(self, X: Any, y: Any = None) -> "TransformToNumerical":
    """Configure transformers for different column types in the input data.

    Args:
      X: Array-like of shape (n_samples, n_features).  If a DataFrame, column
        types are used to determine the appropriate transformation.
      y: Ignored.

    Returns:
      self.
    """
    if not hasattr(
        X, "columns"
    ):  # proxy way to check whether X is a dataframe without importing pandas
      # no dataframe
      self.tfm_ = FunctionTransformer()
      return self

    datetime_cols = []
    cat_cols = []
    numeric_cols = []

    for col in X.columns:
      series = X[col]
      if pd.api.types.is_datetime64_any_dtype(series.dtype):
        datetime_cols.append(col)
      elif check_if_datetime_as_object(series):
        datetime_cols.append(col)
      elif pd.api.types.is_numeric_dtype(series.dtype):
        numeric_cols.append(col)
      else:
        # fallback to categorical if unknown
        cat_cols.append(col)

    cat_pos = [X.columns.get_loc(col) for col in cat_cols]
    numeric_pos = [X.columns.get_loc(col) for col in numeric_cols]
    datetime_pos = [X.columns.get_loc(col) for col in datetime_cols]

    self.tfm_ = ColumnTransformer(
        transformers=[
            (
                "categorical",
                CategoricalOrdinalEncoder(
                    dtype=np.int64,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    min_frequency=self.min_cat_frequency,
                    mode=self.cat_encoder_mode,
                ),
                cat_pos,
            ),
            ("continuous", SimpleImputer(), numeric_pos),
            ("datetime", DatetimeTransformer(), datetime_pos),
        ]
    )
    self.tfm_.fit(X)

    if self.verbose:
      selected_cols = []
      for name, tfm, pos in self.tfm_.transformers_:
        if tfm != "drop":
          cols = list(X.columns[pos])
          selected_cols.extend(cols)
          print(f"Columns classified as {name}: {cols}")
      dropped_cols = set(X.columns).difference(set(selected_cols))
      if len(dropped_cols) >= 1:
        print(
            "The following columns are not used due to their data type:"
            f" {list(dropped_cols)}"
        )

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transform features using the fitted transformer.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Numeric array of shape (n_samples, n_features_out).
    """
    return self.tfm_.transform(X)


class UniqueFeatureFilter(TransformerMixin, BaseEstimator):
  """Removes features with only one unique value in the training set.

  Attributes:
    n_features_in_ : int
    Number of features in the training data.

    n_features_out_ : int
    Number of features after filtering.

    features_to_keep_ : ndarray
    Boolean mask for features to keep.
  """

  features_to_keep_: np.ndarray
  n_features_out_: int
  n_features_in_: int

  def __init__(self, threshold: int = 1):
    """Initialises the filter.

    Notes
    -----
    1. Features with unique values <= threshold are removed.
    2. When the input dataset has very few samples (n_samples <= threshold), all
       features are preserved
       regardless of their unique value counts. This is a safety mechanism
       because:
       - With few samples, it's difficult to reliably assess feature variability
       - A feature might appear constant in few samples but vary in the complete
       dataset

    Args:
      threshold: Features with at most this many unique values are removed.
    """
    self.threshold = threshold

  def fit(self, X: Any, y: Any = None) -> "UniqueFeatureFilter":
    """Learn which features to keep based on unique value counts.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)
    self.n_features_in_ = X.shape[1]
    if X.shape[0] <= self.threshold:
      self.features_to_keep_ = np.ones(self.n_features_in_, dtype=bool)
    else:
      # For each feature, check if it has more than threshold unique values
      self.features_to_keep_ = np.array([
          len(np.unique(X[:, i])) > self.threshold
          for i in range(self.n_features_in_)
      ])

    self.n_features_out_ = np.sum(self.features_to_keep_)

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Filter features according to unique value counts.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Array of shape (n_samples, n_features_out_).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    return X[:, self.features_to_keep_]


class OutlierRemover(TransformerMixin, BaseEstimator):
  """Clips extreme values based on training data distribution.

  This implementation uses a two-stage Z-score based approach to identify and
  clip outliers:
  1. First stage: Identify values beyond z standard deviations and mark as
  missing
  2. Second stage: Recompute statistics without outliers for more robust bounds
  3. Final stage: Apply log-based clipping to maintain data distribution

  Attributes:
    means_: Per-feature means after outlier removal.
    stds_: Per-feature standard deviations after outlier removal.
    lower_bounds_: Per-feature clipping lower bounds.
    upper_bounds_: Per-feature clipping upper bounds.
  """

  means_: np.ndarray
  stds_: np.ndarray
  lower_bounds_: np.ndarray
  upper_bounds_: np.ndarray

  def __init__(self, threshold: float = 4.0):
    """Initialises the remover.

    Args:
      threshold: Values beyond this many standard deviations are outliers.
    """
    self.threshold = threshold

  def fit(self, X: Any, y: Any = None) -> "OutlierRemover":
    """Learn clipping bounds from training data.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)

    # First stage: Identify outliers using initial statistics
    self.means_ = np.nanmean(X, axis=0)
    self.stds_ = np.nanstd(X, axis=0, ddof=1 if X.shape[0] > 1 else 0)

    # Ensure standard deviations are not zero
    self.stds_ = np.maximum(self.stds_, 1e-6)

    # Create a clean copy with outliers replaced by NaN
    X_clean = X.copy()
    lower_bounds = self.means_ - self.threshold * self.stds_
    upper_bounds = self.means_ + self.threshold * self.stds_

    # Create masks for values outside bounds
    lower_mask = X < lower_bounds[np.newaxis, :]
    upper_mask = X > upper_bounds[np.newaxis, :]
    outlier_mask = np.logical_or(lower_mask, upper_mask)

    # Set outliers to NaN
    X_clean[outlier_mask] = np.nan

    # Second stage: Recompute statistics without outliers
    self.means_ = np.nanmean(X_clean, axis=0)
    self.stds_ = np.nanstd(X_clean, axis=0, ddof=1 if X.shape[0] > 1 else 0)

    # Ensure standard deviations are not zero
    self.stds_ = np.maximum(self.stds_, 1e-6)

    # Compute final bounds
    self.lower_bounds_ = self.means_ - self.threshold * self.stds_
    self.upper_bounds_ = self.means_ + self.threshold * self.stds_

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Clip values based on learned bounds with log-based adjustments.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Clipped array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    X = np.maximum(-np.log1p(np.abs(X)) + self.lower_bounds_, X)
    X = np.minimum(np.log1p(np.abs(X)) + self.upper_bounds_, X)
    return X


class CustomStandardScaler(TransformerMixin, BaseEstimator):
  """Standard scaling with clipping.

  This scaler computes the mean and standard deviation of the training data,
  adds a small epsilon to the standard deviation to avoid division by zero,
  and clips the transformed values to a reasonable range.

  Attributes:
    mean_ : ndarray of shape (n_features,)
      The mean value for each feature in the training set.

    scale_ : ndarray of shape (n_features,)
      The standard deviation for each feature in the training set with epsilon
      added.
  """

  mean_: np.ndarray
  scale_: np.ndarray

  def __init__(
      self,
      clip_min: float = -100,
      clip_max: float = 100,
      epsilon: float = 1e-6,
  ):
    """Initialises the scaler.

    Args:
      clip_min: Lower bound for clipping scaled values.
      clip_max: Upper bound for clipping scaled values.
      epsilon: Small constant added to the standard deviation to avoid
        division by zero.
    """
    self.clip_min = clip_min
    self.clip_max = clip_max
    self.epsilon = epsilon

  def fit(self, X: Any, y: Any = None) -> "CustomStandardScaler":
    """Compute the mean and std to be used for scaling.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X)
    self.mean_ = np.mean(X, axis=0)
    self.scale_ = np.std(X, axis=0) + self.epsilon
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Scaled and clipped array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False)
    X_scaled = (X - self.mean_) / self.scale_
    return np.clip(X_scaled, self.clip_min, self.clip_max)


class RTDLQuantileTransformer(BaseEstimator, TransformerMixin):
  """Quantile transformer adapted for tabular deep learning models.

  This implementation is based on research from the RTDL group and adds noise to
  training
  data before applying quantile transformation, improving robustness and
  generalization.
  It also dynamically adjusts the number of quantiles based on data size.

  Attributes:
    normalizer_: The fitted ``QuantileTransformer``.

  Notes:
    Adapted from
    https://github.com/yandex-research/tabular-dl-tabr/blob/75105013189c76bc4f247633c2fb856bc948e579/lib/data.py#L262
    following
    https://github.com/dholzmueller/pytabkit/blob/949bf81e3964f65a33dd2c252c3713c239c17b2d/pytabkit/models/utils.py#L431
  """

  normalizer_: QuantileTransformer

  def __init__(
      self,
      noise: float = 1e-3,
      n_quantiles: int = 1000,
      subsample: int = 1_000_000_000,
      output_distribution: str = "normal",
      random_state: Optional[int] = None,
  ):
    """Initialises the transformer.

    Args:
      noise: Relative magnitude of Gaussian noise to add. Set to 0 to disable.
      n_quantiles: Maximum number of quantiles. Actual number is determined
        dynamically as ``min(n_samples // 30, n_quantiles)``, with a floor of
        10.
      subsample: Maximum samples used to estimate quantiles.
      output_distribution: Target marginal distribution (``"uniform"`` or
        ``"normal"``).
      random_state: Seed for reproducibility.
    """
    self.noise = noise
    self.n_quantiles = n_quantiles
    self.subsample = subsample
    self.output_distribution = output_distribution
    self.random_state = random_state

  def fit(self, X: Any, y: Any = None) -> "RTDLQuantileTransformer":
    """Fit the quantile transformer to training data with optional noise.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    # Calculate the number of quantiles based on data size
    n_quantiles = max(min(X.shape[0] // 30, self.n_quantiles), 10)

    # Initialize QuantileTransformer
    normalizer = QuantileTransformer(
        output_distribution=self.output_distribution,
        n_quantiles=n_quantiles,
        subsample=self.subsample,
        random_state=self.random_state,
    )

    # Add noise if required
    X_modified = self._add_noise(X) if self.noise > 0 else X

    # Fit the normalizer
    normalizer.fit(X_modified)

    # Show that it's fitted
    self.normalizer_ = normalizer

    return self

  def transform(self, X: Any) -> np.ndarray:
    """Transform data using the fitted quantile transformer.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Transformed array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    return self.normalizer_.transform(X)

  def _add_noise(self, X: np.ndarray) -> np.ndarray:
    """Add noise to the input data proportional to feature standard deviations.

    The noise magnitude is controlled by the 'noise' parameter and is scaled
    inversely to the standard deviation of each feature to ensure
    consistent noise levels across features of different scales.

    Args:
      X: Array of shape (n_samples, n_features).

    Returns:
      Noisy array of shape (n_samples, n_features).
      The input data with added Gaussian noise.
    """
    stds = np.std(X, axis=0, keepdims=True)
    noise_std = self.noise / np.maximum(stds, self.noise)
    rng = np.random.default_rng(self.random_state)
    return X + noise_std * rng.standard_normal(X.shape)


class RecursionLimitManager:
  """Context manager to temporarily set the Python recursion limit."""

  def __init__(self, limit: int):
    """Initialises the manager.

    Args:
      limit: The temporary recursion limit to use.
    """
    self.limit = limit
    self.original_limit: Optional[int] = None

  def __enter__(self):
    self.original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(self.limit)
    return self

  def __exit__(self, type, value, traceback):
    sys.setrecursionlimit(self.original_limit)
    return False


class FeatureShuffler:
  """Generates feature permutations for ensemble creation.

  Attributes:
    rng_: The random number generator used for shuffling.
  """

  rng_: random.Random

  def __init__(
      self,
      n_features: int,
      method: str = "latin",
      max_features_for_latin: int = 4000,
      random_state: Optional[int] = None,
  ):
    """Initialises the shuffler.

    Args:
      n_features: Number of features to shuffle.
      method: Shuffling strategy: ``"latin"``, ``"random"``, ``"shift"``, or
        ``"none"``.
      max_features_for_latin: When ``n_features`` exceeds this value, the method
        falls back from ``"latin"`` to ``"random"``.
      random_state: Seed for reproducibility.
    """
    self.n_features = n_features
    self.method = method
    self.max_features_for_latin = max_features_for_latin
    self.random_state = random_state
    self.rng_ = random.Random(self.random_state)

  def shuffle(self, n_estimators: int) -> List[np.ndarray]:
    """Generates a list of feature-shuffle patterns.

    Args:
      n_estimators: Number of shuffle patterns to generate.

    Returns:
      List of ``n_estimators`` arrays, each of length ``self.n_features``,
      representing a permutation of feature indices.
    """
    self.rng_ = random.Random(self.random_state)
    feature_indices = list(range(self.n_features))

    if self.n_features > self.max_features_for_latin and self.method == "latin":
      method = "random"
    else:
      method = self.method

    if method == "none" or n_estimators == 1:
      shuffle_patterns = [feature_indices]
    elif method == "shift":
      shuffle_patterns = [
          feature_indices[-i:] + feature_indices[:-i]
          for i in range(min(n_estimators, self.n_features))
      ]
    elif method == "random":
      if self.n_features <= 5:
        all_perms = [list(perm) for perm in itertools.permutations(feature_indices)]
        shuffle_patterns = self.rng_.sample(all_perms, min(n_estimators, len(all_perms)))
      else:
        shuffle_patterns = [
            self.rng_.sample(feature_indices, self.n_features)
            for _ in range(n_estimators)
        ]
    elif method == "latin":
      with RecursionLimitManager(100000):
        shuffle_patterns = self._latin_squares()
        if len(shuffle_patterns) > n_estimators:
          shuffle_patterns = self.rng_.sample(shuffle_patterns, n_estimators)
    else:
      raise ValueError(
          f"Unknown method: {method}. Use 'shift', 'random', 'latin', or 'none'."
      )

    return [np.array(p) for p in shuffle_patterns]

  def _latin_squares(self) -> List[List[int]]:
    """Generates a random Latin square as a list of row permutations."""

    def _shuffle_transpose_shuffle(matrix):
      square = deepcopy(matrix)
      self.rng_.shuffle(square)
      trans = list(zip(*square))
      self.rng_.shuffle(trans)
      return trans

    def _rls(symbols):
      n = len(symbols)
      if n == 1:
        return [symbols]
      sym = self.rng_.choice(symbols)
      symbols_copy = list(symbols)
      symbols_copy.remove(sym)
      square = _rls(symbols_copy)
      square.append(list(square[0]))
      for i in range(n):
        square[i].insert(i, sym)
      return square

    symbols = list(range(self.n_features))
    square = _rls(symbols)
    feature_shuffles = _shuffle_transpose_shuffle(square)
    return [list(shuffle) for shuffle in feature_shuffles]


class PreprocessingPipeline(TransformerMixin, BaseEstimator):
  """Preprocessing pipeline combining scaling, normalization, and outlier handling.

  Attributes:
    standard_scaler_: Fitted ``CustomStandardScaler``.
    normalizer_: Fitted normalization transformer (or ``None``).
    outlier_remover_: Fitted ``OutlierRemover``.
    X_transformed_: Cached transformed training data.
    X_min_: Per-feature minimum of scaled data (used for clipping at transform).
    X_max_: Per-feature maximum of scaled data (used for clipping at transform).
  """

  n_features_in_: int
  standard_scaler_: Optional[CustomStandardScaler]
  normalizer_: Optional[Any]
  outlier_remover_: Optional[OutlierRemover]
  X_transformed_: np.ndarray
  X_min_: np.ndarray
  X_max_: np.ndarray

  def __init__(
      self,
      normalization_method: str = "none",
      outlier_threshold: float = 4.0,
      random_state: Optional[int] = None,
  ):
    """Initialises the pipeline.

    Args:
      normalization_method: Normalization strategy: ``"none"``, ``"power"``,
        ``"quantile"``, ``"quantile_rtdl"``, or ``"robust"``.
      outlier_threshold: Z-score threshold for outlier detection.
      random_state: Seed for reproducible normalization.
    """
    self.normalization_method = normalization_method
    self.outlier_threshold = outlier_threshold
    self.random_state = random_state

  def fit(self, X: Any, y: Any = None) -> "PreprocessingPipeline":
    """Fit the preprocessing pipeline.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Ignored.

    Returns:
      self.
    """
    X = validate_data(self, X, ensure_min_features=0)
    # If there are no features, there's nothing to preprocess.
    if self.n_features_in_ == 0:
      self.standard_scaler_ = None
      self.normalizer_ = None
      self.outlier_remover_ = None
      self.X_transformed_ = X
      return self
    # 1. Apply standard scaling
    self.standard_scaler_ = CustomStandardScaler()
    X_scaled = self.standard_scaler_.fit_transform(X)

    # 2. Apply normalization
    if self.normalization_method != "none":
      if self.normalization_method == "power":
        self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
      elif self.normalization_method == "quantile":
        self.normalizer_ = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )
      elif self.normalization_method == "quantile_rtdl":
        self.normalizer_ = Pipeline([
            (
                "quantile_rtdl",
                RTDLQuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                ),
            ),
            ("std", StandardScaler()),
        ])
      elif self.normalization_method == "robust":
        self.normalizer_ = RobustScaler(unit_variance=True)
      else:
        raise ValueError(
            f"Unknown normalization method: {self.normalization_method}"
        )
      self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
      self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
      X_normalized = self.normalizer_.fit_transform(X_scaled)
    else:
      self.normalizer_ = None
      X_normalized = X_scaled

    # 3. Handle outliers
    self.outlier_remover_ = OutlierRemover(threshold=self.outlier_threshold)
    self.X_transformed_ = self.outlier_remover_.fit_transform(X_normalized)
    return self

  def transform(self, X: Any) -> np.ndarray:
    """Apply the preprocessing pipeline.

    Args:
      X: Array-like of shape (n_samples, n_features).

    Returns:
      Preprocessed array of shape (n_samples, n_features).
    """
    check_is_fitted(self)
    X = validate_data(self, X, reset=False, copy=True, ensure_min_features=0)
    if self.n_features_in_ == 0:
      return X
    if self.standard_scaler_ is not None:
      X = self.standard_scaler_.transform(X)
    if self.normalizer_ is not None:
      try:
        # this can fail in rare cases if there is an outlier in X that was not present in fit()
        X = self.normalizer_.transform(X)
      except ValueError:
        # clip values to train min/max
        X = np.clip(X, self.X_min_, self.X_max_)
        X = self.normalizer_.transform(X)
    if self.outlier_remover_ is not None:
      X = self.outlier_remover_.transform(X)
    return X


class EnsembleGenerator(TransformerMixin, BaseEstimator):
  """Generate diverse ensemble variants for robust tabular prediction with TabFM.

  Creates multiple views of a dataset by combining different feature shuffles,
  class-label shifts, categorical value permutations, and normalization methods.

  Attributes:
    norm_methods_: List of normalization methods in use.
    cat_features_: Indices of categorical features after unique filtering.
    cat_values_: Unique values per categorical feature column index.
    X_: Training features after unique filtering (cached).
    y_: Training labels (cached).
    n_features_in_: Number of features after unique filtering.
    n_classes_: Number of unique classes (classification only).
    rng_: Random-number generator.
    ensemble_configs_: Ordered dict mapping norm method -> list of
      (shuffle_pattern, shift_offset, cat_perm) tuples.
    feature_shuffle_patterns_: Ordered dict mapping norm method -> shuffle
      patterns.
    class_shift_offsets_: Ordered dict mapping norm method -> shift offsets.
    cat_permutations_: Ordered dict mapping norm method -> cat permutation dicts.
    preprocessors_: Fitted ``PreprocessingPipeline`` per norm method.
  """

  norm_methods_: List[str]
  cat_features_: np.ndarray
  cat_values_: Dict[int, np.ndarray]
  X_: np.ndarray
  y_: np.ndarray
  n_features_in_: int
  n_classes_: int
  rng_: random.Random
  ensemble_configs_: collections.OrderedDict
  feature_shuffle_patterns_: collections.OrderedDict
  class_shift_offsets_: collections.OrderedDict
  cat_permutations_: collections.OrderedDict
  preprocessors_: Dict[str, PreprocessingPipeline]

  def __init__(
      self,
      n_estimators: int,
      norm_methods: Union[str, List[str], None] = None,
      feat_shuffle_method: str = "latin",
      class_shift: bool = True,
      cat_features: Optional[List[int]] = None,
      permute_categorical: bool = False,
      outlier_threshold: float = 4.0,
      random_state: Optional[int] = None,
      task: str = "classification",
  ):
    """Initialises the generator.

    Args:
      n_estimators: Number of ensemble members to generate.
      norm_methods: Normalization method(s) to use. If ``None``, defaults to
        ``["none", "power"]``. May be a single string or a list.
      feat_shuffle_method: Feature-permutation strategy (``"latin"``,
        ``"random"``, ``"shift"``, or ``"none"``).
      class_shift: Whether to apply random class-label shifts (classification
        only).
      cat_features: Indices of categorical features in the *encoded* input.
      permute_categorical: Whether to randomly permute categorical values across
        ensemble members.
      outlier_threshold: Z-score threshold forwarded to``OutlierRemover``.
      random_state: Seed for reproducibility.
      task: Either ``"classification"`` or ``"regression"``.
    """
    self.n_estimators = n_estimators
    self.norm_methods = norm_methods
    self.feat_shuffle_method = feat_shuffle_method
    self.class_shift = class_shift
    self.cat_features = cat_features
    self.permute_categorical = permute_categorical
    self.outlier_threshold = outlier_threshold
    self.random_state = random_state
    self.task = task

  def fit(self, X: Any, y: Any) -> "EnsembleGenerator":
    """Fit the ensemble generator to training data.

    Args:
      X: Array-like of shape (n_samples, n_features).
      y: Label array of shape (n_samples,).

    Returns:
      self.
    """
    validate_data(self, X, y)

    if self.norm_methods is None:
      self.norm_methods_ = ["none", "power"]
    elif isinstance(self.norm_methods, str):
      self.norm_methods_ = [self.norm_methods]
    else:
      self.norm_methods_ = list(self.norm_methods)

    self.unique_filter_ = UniqueFeatureFilter()
    X = self.unique_filter_.fit_transform(X)

    # Update categorical features indices after filtering
    if self.cat_features is not None:
      # Create mask of original features
      mask = np.zeros(self.unique_filter_.n_features_in_, dtype=bool)
      mask[self.cat_features] = True
      # Filter
      mask = mask[self.unique_filter_.features_to_keep_]
      self.cat_features_ = np.where(mask)[0]
      self.cat_values_ = {idx: np.unique(X[:, idx]) for idx in self.cat_features_}
    else:
      self.cat_features_ = np.array([])
      self.cat_values_ = {}

    self.X_ = X
    self.y_ = y
    self.n_features_in_ = X.shape[1]
    if self.task == "classification":
      self.n_classes_ = len(np.unique(y))
    else:
      self.n_classes_ = 0

    self.rng_ = random.Random(self.random_state)

    # Generate and unpack all ensemble components
    (
        self.ensemble_configs_,
        self.feature_shuffle_patterns_,
        self.class_shift_offsets_,
        self.cat_permutations_,
    ) = self._generate_ensemble()

    self.preprocessors_ = {}
    for norm_method in self.ensemble_configs_:
      if norm_method not in self.preprocessors_:
        preprocessor = PreprocessingPipeline(
            normalization_method=norm_method,
            outlier_threshold=self.outlier_threshold,
            random_state=self.random_state,
        )
        preprocessor.fit(X)
        self.preprocessors_[norm_method] = preprocessor

    return self

  def _generate_ensemble(
      self,
  ) -> Tuple[
      collections.OrderedDict,
      collections.OrderedDict,
      collections.OrderedDict,
      collections.OrderedDict,
  ]:
    """Create diverse ensemble configurations grouped by normalization method.

    Returns:
      A 4-tuple of OrderedDicts:
        (ensemble_configs, feature_shuffle_patterns, class_shift_offsets,
         cat_permutations_grouped).
    """
    shuffler = FeatureShuffler(
        n_features=self.n_features_in_,
        method=self.feat_shuffle_method,
        random_state=self.random_state,
    )
    shuffle_patterns = shuffler.shuffle(self.n_estimators)

    # Ensure shuffle_patterns matches n_estimators (FeatureShuffler might return fewer)
    if len(shuffle_patterns) < self.n_estimators:
      num_cycles = (
          self.n_estimators + len(shuffle_patterns) - 1
      ) // len(shuffle_patterns)
      shuffle_patterns = (shuffle_patterns * num_cycles)[: self.n_estimators]

    # 2. Generate Class Shifts
    if (
        self.task == "classification"
        and self.class_shift
        and self.n_estimators > 1
        and self.n_classes_ > 1
    ):
      base_offsets = self.rng_.sample(range(self.n_classes_), self.n_classes_)
      num_cycles = (
          self.n_estimators + len(base_offsets) - 1
      ) // len(base_offsets)
      shift_offsets = (base_offsets * num_cycles)[: self.n_estimators]
    else:
      shift_offsets = [0] * self.n_estimators

    # 3. Generate Categorical Permutations
    cat_permutations: List[Optional[Dict[int, Dict[Any, Any]]]] = []
    if self.permute_categorical and len(self.cat_features_) > 0:
      for _ in range(self.n_estimators):
        perm_dict = {
            col_idx: dict(
                zip(vals, self.rng_.sample(list(vals), len(vals)))
            )
            for col_idx, vals in self.cat_values_.items()
        }
        cat_permutations.append(perm_dict)
    else:
      cat_permutations = [None] * self.n_estimators

    # 4. Combine into configurations
    shuffle_shift_cat_configs = list(
        zip(shuffle_patterns, shift_offsets, cat_permutations)
    )
    self.rng_.shuffle(shuffle_shift_cat_configs)

    # 5. Assign Normalization Methods
    num_cycles = (
        self.n_estimators + len(self.norm_methods_) - 1
    ) // len(self.norm_methods_)
    norm_methods_for_estimators = (self.norm_methods_ * num_cycles)[: self.n_estimators]
    full_configs = list(zip(norm_methods_for_estimators, shuffle_shift_cat_configs))

    # Group by normalization method and separate components
    ensemble_configs: collections.OrderedDict = collections.OrderedDict()
    feature_shuffle_patterns: collections.OrderedDict = collections.OrderedDict()
    class_shift_offsets_dict: collections.OrderedDict = collections.OrderedDict()
    cat_permutations_grouped: collections.OrderedDict = collections.OrderedDict()

    for norm_method in self.norm_methods_:
      configs = [config for norm, config in full_configs if norm == norm_method]
      if configs:
        ensemble_configs[norm_method] = configs
        feature_shuffle_patterns[norm_method] = [c[0] for c in configs]
        class_shift_offsets_dict[norm_method] = [c[1] for c in configs]
        cat_permutations_grouped[norm_method] = [c[2] for c in configs]

    return (
        ensemble_configs,
        feature_shuffle_patterns,
        class_shift_offsets_dict,
        cat_permutations_grouped,
    )

  def transform(self, X: Any) -> collections.OrderedDict:
    """Generate ensemble data views for test samples.

    Args:
      X: Array-like of shape (n_test_samples, n_features).

    Returns:
      OrderedDict mapping each norm_method to a tuple
      ``(Xs, ys)`` where ``Xs`` has shape
      ``(n_configs, n_train + n_test, n_features)`` and ``ys`` has shape
      ``(n_configs, n_train)``.
    """
    check_is_fitted(self, ["ensemble_configs_"])
    X = self.unique_filter_.transform(X)
    y = self.y_

    data: collections.OrderedDict = collections.OrderedDict()
    for norm_method, shuffle_shift_cat_configs in self.ensemble_configs_.items():
      preprocessor = self.preprocessors_[norm_method]
      X_ensemble = []
      y_ensemble = []

      for shuffle_pattern, shift_offset, cat_perm in shuffle_shift_cat_configs:
        if cat_perm:
          # If we have categorical permutations, we must apply them before preprocessing
          # Note: self.X_ is the fitted training data. X is the test data.

          # We need to construct the full dataset (Train + Test)
          X_full = np.concatenate([self.X_, X], axis=0)

          # Apply value permutation
          for col, mapping in cat_perm.items():
            col_vals = X_full[:, col]
            u_vals, inverse = np.unique(col_vals, return_inverse=True)
            mapped_u_vals = u_vals.copy()
            for i, val in enumerate(u_vals):
              if val in mapping:
                mapped_u_vals[i] = mapping[val]
            X_full[:, col] = mapped_u_vals[inverse]
          X_variant_instance = preprocessor.transform(X_full)
        else:
          X_train_trans = getattr(
              preprocessor, "X_transformed_", preprocessor.transform(self.X_)
          )
          X_test_trans = preprocessor.transform(X)
          X_variant_instance = np.concatenate([X_train_trans, X_test_trans], axis=0)

        # Apply feature shuffling
        X_ensemble.append(X_variant_instance[:, shuffle_pattern])

        # Apply class shifting
        if self.task == "classification":
          y_ensemble.append((y + shift_offset) % self.n_classes_)
        else:
          y_ensemble.append(y)

      data[norm_method] = (
          np.stack(X_ensemble, axis=0),
          np.stack(y_ensemble, axis=0),
      )

    return data


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@jt.typed
def _pad_batch_to_multiple_of(
    x: jax.Array | np.ndarray, divisor: int
) -> jax.Array | np.ndarray:
  """Pad axis 0 of array (at the end) to a multiple of ``divisor``.

  Args:
    x: Input array of any shape.
    divisor: Target multiple for axis 0.

  Returns:
    Array whose first dimension is the smallest multiple of ``divisor`` that
    is >= ``x.shape[0]``.  If ``divisor <= 1`` or no padding is required,
    the original array is returned unchanged.
  """
  if divisor <= 1:
    return x
  pad_size = (divisor - (x.shape[0] % divisor)) % divisor
  if pad_size == 0:
    return x
  pad_width = ((0, pad_size),) + ((0, 0),) * (x.ndim - 1)
  return np.pad(x, pad_width, constant_values=0)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TabFMClassifier(ClassifierMixin, BaseEstimator):
  """TabFM (Tabular Foundation Model) classifier with scikit-learn interface.

  The TabFM model is a pre-trained foundation model that makes predictions via
  in-context learning: at inference time it is shown the training data as
  context and then asked to predict on test samples.  This class wraps the model
  with scikit-learn's ``ClassifierMixin`` / ``BaseEstimator`` interface,
  handling data pre-processing and ensemble aggregation automatically.

  Attributes:
    y_encoder_: Fitted ``CategoricalOrdinalEncoder`` for class labels.
    classes_: Array of original class labels.
    n_classes_: Number of unique classes.
    X_encoder_: Fitted ``TransformToNumerical`` for input features.
    ensemble_generator_: Fitted ``EnsembleGenerator``.
  """

  y_encoder_: CategoricalOrdinalEncoder
  classes_: np.ndarray
  n_classes_: int
  X_encoder_: TransformToNumerical
  ensemble_generator_: EnsembleGenerator

  def __init__(
      self,
      model: Any,
      config: Optional[Union[argparse.Namespace, flags.FlagValues]] = None,
      n_estimators: int = 32,
      norm_methods: Optional[Union[str, List[str]]] = None,
      feat_shuffle_method: str = "latin",
      class_shift: bool = True,
      permute_categorical: bool = False,
      outlier_threshold: float = 4.0,
      softmax_temperature: float = 0.9,
      average_logits: bool = True,
      use_amp: bool = True,
      batch_size: Optional[int] = 1,
      random_state: Optional[int] = 42,
      verbose: bool = False,
      cat_encoder_mode: str = "appearance",
  ):
    """Initialises the classifier.

    Args:
      model: Pre-trained TabFM model (NNX module).
      config: Model configuration (absl flags or argparse namespace).
      n_estimators: Number of ensemble members.
      norm_methods: Normalization method(s) for the ensemble. Defaults to
        ``["none", "power"]``.
      feat_shuffle_method: Feature-permutation strategy for the ensemble.
      class_shift: Whether to apply random class-label shifts.
      permute_categorical: Whether to randomly permute categorical values.
      outlier_threshold: Z-score threshold for outlier clipping.
      softmax_temperature: Temperature applied before the final softmax.
      average_logits: If True, average logits before applying softmax;
        otherwise average probabilities.
      use_amp: Whether to use automatic mixed precision (currently informational
        only).
      batch_size: Number of ensemble members to forward through the model at
        once.  ``None`` or 0 means all at once.
      random_state: Seed for ensemble randomness.
      verbose: Whether to print informational messages.
      cat_encoder_mode: Categorical encoding order (``"appearance"`` or
        ``"frequency"``).
    """
    self.model = model
    self.config = config
    self.n_estimators = n_estimators
    self.norm_methods = norm_methods
    self.feat_shuffle_method = feat_shuffle_method
    self.class_shift = class_shift
    self.permute_categorical = permute_categorical
    self.outlier_threshold = outlier_threshold
    self.softmax_temperature = softmax_temperature
    self.average_logits = average_logits
    self.use_amp = use_amp
    self.batch_size = batch_size
    self.random_state = random_state
    self.verbose = verbose
    self.cat_encoder_mode = cat_encoder_mode

  def _more_tags(self):
    """Mark classifier as non-deterministic to bypass certain sklearn tests."""
    return dict(non_deterministic=True)

  def __sklearn_tags__(self):
    tags = super().__sklearn_tags__()
    tags.non_deterministic = True
    return tags

  def fit(self, X: Any, y: Any) -> "TabFMClassifier":
    """Fit the classifier to training data.

    Prepares the model for prediction by:
    1. Encoding class labels using LabelEncoder
    2. Converting input features to numerical values
    3. Fitting the ensemble generator to create transformed dataset views
    4. Using the pre-trained TabFM model

    The model itself is not trained on the data; it uses in-context learning
    at inference time. This method only prepares the data transformations.

    Args:
      X: Training features of shape (n_samples, n_features).
      y: Training class labels of shape (n_samples,).

    Returns:
      self.

    Raises:
      ValueError
        If the number of classes exceeds the model's maximum supported classes
        and hierarchical classification is disabled.
    """
    check_classification_targets(y)

    # Encode class labels (sorted/alphabetical to match sklearn convention)
    self.y_encoder_ = CategoricalOrdinalEncoder(dtype=np.int64, mode="alphabetical")
    # Reshape for CategoricalOrdinalEncoder
    y_2d = y.reshape(-1, 1) if isinstance(y, np.ndarray) else np.array(y).reshape(-1, 1)
    y_encoded = self.y_encoder_.fit_transform(y_2d)
    y = y_encoded.flatten()

    # CategoricalOrdinalEncoder stores categories in a list of arrays
    self.classes_ = self.y_encoder_.categories_[0]
    self.n_classes_ = len(self.classes_)

    if self.n_classes_ > self.model.max_classes and self.verbose:
      print(
          f"The number of classes ({self.n_classes_}) exceeds the max number of"
          f" classes ({self.model.max_classes}) natively supported by the"
          " model. Therefore, hierarchical classification is used."
      )

    #  Transform input features
    self.X_encoder_ = TransformToNumerical(
        verbose=self.verbose, cat_encoder_mode=self.cat_encoder_mode
    )
    X = self.X_encoder_.fit_transform(X)

    # Identify categorical feature indices (assumes OrdinalEncoder is the first transformer)
    # The ColumnTransformer in TransformToNumerical puts categorical features first.
    if hasattr(self.X_encoder_.tfm_, "transformers_"):
      n_cat = len(getattr(self.X_encoder_.tfm_, "transformers_")[0][2])
    else:
      n_cat = 0
    cat_features = list(range(n_cat))

    # Fit ensemble generator to create multiple dataset views
    self.ensemble_generator_ = EnsembleGenerator(
        n_estimators=self.n_estimators,
        norm_methods=self.norm_methods or ["none", "power"],
        feat_shuffle_method=self.feat_shuffle_method,
        class_shift=self.class_shift,
        cat_features=cat_features,
        permute_categorical=self.permute_categorical,
        outlier_threshold=self.outlier_threshold,
        random_state=self.random_state,
    )
    self.ensemble_generator_.fit(X, y)

    return self

  @jt.typed
  def _batch_forward(
      self,
      Xs: jt.Float[jax.Array | np.ndarray, "B T H"],
      ys: jt.Shaped[jax.Array | np.ndarray, "B T_train"],
      cat_masks: Optional[jt.Bool[jax.Array | np.ndarray, "B H"]] = None,
  ) -> jt.Float[jax.Array | np.ndarray, "B T_test K"]:
    """Process model forward passes in batches to manage memory efficiently.

    This method handles the batched inference through the TabFM model,
    dividing the ensemble members into smaller batches to avoid out-of-memory
    errors.

    Parameters
    ----------
    Xs : np.ndarray
        Input features of shape (n_datasets, n_samples, n_features), where
        n_datasets
        is the number of ensemble members.

    ys : np.ndarray
        Training labels of shape (n_datasets, train_size), where train_size is
        the
        number of samples used for in-context learning.

    cat_masks : np.ndarray or None, optional
        Boolean mask of shape (n_datasets, n_features) indicating which features
        are categorical (True) vs. numerical (False). The model uses this to
        apply feature-type-specific processing: for example, categorical
        features
        may use different Fourier frequencies or random embeddings compared to
        numerical features. If None, all features are treated as numerical.

    Returns
    -------
    np.ndarray
        Model outputs (logits or probabilities) of shape (n_datasets, test_size,
        n_classes)
        where test_size = n_samples - train_size.
    """
    mesh = jax.sharding.get_mesh()
    if mesh and "data" in mesh.axis_names:
      num_data_shards = mesh.axis_sizes[mesh.axis_names.index("data")]
      data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    else:
      num_data_shards = 1
      data_sharding = None

    num_classes = self.n_classes_

    _has_compiled_attr = (
        "_predict_step_compiled_with_cat"
        if (cat_masks is not None and hasattr(self.model, "cell_embedder"))
        else "_predict_step_compiled_no_cat"
    )

    if not hasattr(self, _has_compiled_attr):
      data_sharding = NamedSharding(
          jax.sharding.Mesh(jax.devices(), ("data",)), PartitionSpec("data")
      )

      if cat_masks is not None and hasattr(self.model, "cell_embedder"):

        @nnx.jit(
            in_shardings=(
                None,
                data_sharding,
                data_sharding,
                data_sharding,
                data_sharding,
                data_sharding,
            ),
            out_shardings=data_sharding,
        )
        def _predict_step_fn(model, X, y, train_size, d, cat_mask):
          return model(
              X,
              y,
              train_size=train_size,
              d=d,
              cat_mask=cat_mask,
              num_classes=num_classes,
          )

      else:

        @nnx.jit(
            in_shardings=(
                None,
                data_sharding,
                data_sharding,
                data_sharding,
                data_sharding,
            ),
            out_shardings=data_sharding,
        )
        def _predict_step_fn(model, X, y, train_size, d):
          return model(
              X,
              y,
              train_size=train_size,
              d=d,
              num_classes=num_classes,
          )

      setattr(self, _has_compiled_attr, _predict_step_fn)

    _predict_step_compiled = getattr(self, _has_compiled_attr)

    batch_size_per_process = self.batch_size or Xs.shape[0]
    n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
    if n_batches > 1:
      Xs_split = np.array_split(Xs, n_batches)
      ys_split = np.array_split(ys, n_batches)
      if cat_masks is not None:
        cat_masks_split = np.array_split(cat_masks, n_batches)
    else:
      Xs_split = [Xs]
      ys_split = [ys]
      if cat_masks is not None:
        cat_masks_split = [cat_masks]

    outputs = []
    cat_masks_iter = cat_masks_split if cat_masks is not None else [None] * len(Xs_split)
    for X_batch, y_batch, cat_mask_batch in zip(Xs_split, ys_split, cat_masks_iter):
      orig_batch_size = X_batch.shape[0]
      X_batch = _pad_batch_to_multiple_of(X_batch, num_data_shards)
      y_batch = _pad_batch_to_multiple_of(y_batch, num_data_shards)

      X_batch = jax.device_put(jnp.array(X_batch, dtype=jnp.float32), data_sharding)
      y_batch = jax.device_put(jnp.array(y_batch, dtype=jnp.float32), data_sharding)
      batch_size_padded = X_batch.shape[0]
      train_size_val = y_batch.shape[1]
      train_size = jax.device_put(
          jnp.repeat(train_size_val, batch_size_padded), data_sharding
      )
      d_batch = jax.device_put(
          jnp.full((batch_size_padded,), X_batch.shape[-1], dtype=jnp.int32),
          data_sharding,
      )

      # Pad y to match X length along sequence dimension
      if y_batch.shape[1] < X_batch.shape[1]:
        y_batch = jnp.pad(
            y_batch,
            ((0, 0), (0, X_batch.shape[1] - y_batch.shape[1])),
            constant_values=-100.0,
        )



      # No gradient calculation needed for inference
      if cat_mask_batch is not None and hasattr(self.model, "cell_embedder"):
        cat_mask_batch = _pad_batch_to_multiple_of(cat_mask_batch, num_data_shards)
        cat_mask_batch = jax.device_put(
            jnp.array(cat_mask_batch, dtype=jnp.bool_), data_sharding
        )
        out = _predict_step_compiled(
            self.model, X_batch, y_batch, train_size, d_batch, cat_mask_batch
        )
      else:
        out = _predict_step_compiled(
            self.model, X_batch, y_batch, train_size, d_batch
        )

      # Slice output to keep only test predictions and unpadded batch.
      out = out[:orig_batch_size, train_size_val:, :]
      from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top
      out = multihost_utils.process_allgather(out, tiled=True)
      outputs.append(out)

    return np.concatenate(outputs, axis=0)

  @jt.typed
  def predict_proba(self, X: Any) -> jt.Float[jax.Array | np.ndarray, "T K"]:
    """Predict class probabilities for test samples.

    Applies the ensemble of TabFM models to make predictions, with each
    ensemble
    member providing predictions that are then averaged. The method:
    1. Transforms input data using the fitted encoders
    2. Applies the ensemble generator to create multiple views
    3. Forwards each view through the model
    4. Corrects for class shifts
    5. Averages predictions across ensemble members

    Args:
      X : array-like of shape (n_samples, n_features)
        Test samples for prediction.

    Returns:
      np.ndarray of shape (n_samples, n_classes)
        Class probabilities for each test sample.
    """
    check_is_fitted(self)
    if isinstance(X, np.ndarray) and len(X.shape) == 1:
      # Reject 1D arrays to maintain sklearn compatibility
      raise ValueError(
          f"The provided input X is one-dimensional. Reshape your data."
      )

    X = self.X_encoder_.transform(X)
    data = self.ensemble_generator_.transform(X)
    Xs_all, ys_all, cat_masks_all = [], [], []
    for norm_method, (Xs, ys) in data.items():
      Xs_all.append(Xs)
      ys_all.append(ys)
      configs = self.ensemble_generator_.ensemble_configs_[norm_method]
      for shuffle_pattern, _, _ in configs:
        mask = np.zeros(self.ensemble_generator_.n_features_in_, dtype=np.bool_)
        if hasattr(self.ensemble_generator_, "cat_features_"):
          mask[self.ensemble_generator_.cat_features_] = True
        cat_masks_all.append(mask[shuffle_pattern])

    # Concatenate all ensemble generation variants to run evaluate forward logic in single batch
    Xs_all = np.concatenate(Xs_all, axis=0)
    ys_all = np.concatenate(ys_all, axis=0)
    cat_masks_all = np.stack(cat_masks_all, axis=0)
    outputs = self._batch_forward(Xs_all, ys_all, cat_masks_all)
    outputs = outputs[..., :self.n_classes_]

    # Extract class shift offsets from ensemble generator
    class_shift_offsets = []
    for offsets in self.ensemble_generator_.class_shift_offsets_.values():
      class_shift_offsets.extend(offsets)

    # Determine actual number of ensemble members
    # May be fewer than requested if dataset has quite limited features and classes
    n_estimators = len(class_shift_offsets)

    # Aggregate predictions from all ensemble members, correcting for class shifts
    avg = None
    for i, offset in enumerate(class_shift_offsets):
      out = outputs[i]

      # Slice to the actual number of classes to avoid rotating padding/garbage

      # Reverse the class shift
      out = np.concatenate([out[..., offset:], out[..., :offset]], axis=-1)

      if not self.average_logits:
        out = self.softmax(out, axis=1, temperature=self.softmax_temperature)

      if avg is None:
        avg = out
      else:
        avg += out

    # Calculate ensemble average
    avg /= n_estimators

    if self.average_logits:
      return self.softmax(avg, axis=1, temperature=self.softmax_temperature)

    # Normalize probabilities to sum to 1
    return avg / avg.sum(axis=1, keepdims=True)

  @jt.typed
  def predict(self, X: Any) -> np.ndarray:
    """Predict class labels for test samples.

    Uses predict_proba to get class probabilities and returns the class with
    the highest probability for each sample.

    Args:
      X : array-like of shape (n_samples, n_features)
          Test samples for prediction.

    Returns:
      array-like of shape (n_samples,)
        Predicted class labels for each test sample.
    """
    proba = self.predict_proba(X)
    y = np.argmax(proba[:, : self.n_classes_], axis=1)
    y_2d = y.reshape(-1, 1)
    y_decoded = self.y_encoder_.inverse_transform(y_2d)
    return y_decoded.flatten()

  @jt.typed
  @staticmethod
  def softmax(
      x: np.ndarray, axis: int = -1, temperature: float = 0.9
  ) -> np.ndarray:
    """Compute temperature-scaled softmax.

    Args:
      x: Input logit array of any shape.
      axis: Axis along which to compute softmax.
      temperature: Scaling factor applied before the softmax; values < 1
        produce a sharper distribution.

    Returns:
      Softmax probabilities with the same shape as ``x``.
    """
    x = x / temperature
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    # Compute softmax
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


class TabFMRegressor(RegressorMixin, BaseEstimator):
  """TabFM (Tabular Foundation Model) regressor with scikit-learn interface.

  The pre-trained TabFM model is used for in-context regression.  Target
  values are standardized before being passed to the model and are
  inverse-transformed on output.

  Attributes:
    X_encoder_: Fitted ``TransformToNumerical`` for input features.
    y_scaler_: Fitted ``StandardScaler`` for target standardization.
    ensemble_generator_: Fitted ``EnsembleGenerator``.
  """

  X_encoder_: TransformToNumerical
  y_scaler_: StandardScaler
  ensemble_generator_: EnsembleGenerator

  def __init__(
      self,
      model: Any,
      config: Optional[Union[argparse.Namespace, flags.FlagValues]] = None,
      n_estimators: int = 32,
      norm_methods: Optional[Union[str, List[str]]] = None,
      feat_shuffle_method: str = "latin",
      permute_categorical: bool = False,
      outlier_threshold: float = 4.0,
      use_amp: bool = True,
      batch_size: Optional[int] = 1,
      random_state: Optional[int] = 42,
      verbose: bool = False,
      cat_encoder_mode: str = "appearance",
  ):
    """Initialises the regressor.

    Args:
      model: Pre-trained TabFM model (NNX module).
      config: Model configuration (absl flags or argparse namespace).
      n_estimators: Number of ensemble members.
      norm_methods: Normalization method(s) for the ensemble.  Defaults to
        ``["none", "power"]``.
      feat_shuffle_method: Feature-permutation strategy for the ensemble.
      permute_categorical: Whether to randomly permute categorical values.
      outlier_threshold: Z-score threshold for outlier clipping.
      use_amp: Whether to use automatic mixed precision (informational only).
      batch_size: Number of ensemble members to forward at once.  ``None``
        or 0 means all at once.
      random_state: Seed for ensemble randomness.
      verbose: Whether to print informational messages.
      cat_encoder_mode: Categorical encoding order (``"appearance"`` or
        ``"frequency"``).
    """
    self.model = model
    self.config = config
    self.n_estimators = n_estimators
    self.norm_methods = norm_methods
    self.feat_shuffle_method = feat_shuffle_method
    self.permute_categorical = permute_categorical
    self.outlier_threshold = outlier_threshold
    self.use_amp = use_amp
    self.batch_size = batch_size
    self.random_state = random_state
    self.verbose = verbose
    self.cat_encoder_mode = cat_encoder_mode

  def _more_tags(self):
    """Mark regressor as non-deterministic to bypass certain sklearn tests."""
    return dict(non_deterministic=True)

  def fit(self, X: Any, y: Any) -> "TabFMRegressor":
    """Fit the regressor to training data.

    Prepares the model for prediction by:
    1. Converting input features to numerical values
    2. Fitting the ensemble generator to create transformed dataset views

    The model itself is not trained on the data; it uses in-context learning
    at inference time. This method only prepares the data transformations.

    Args:
      X : array-like of shape (n_samples, n_features)
          Training input data.

      y : array-like of shape (n_samples,)
          Training target values.

    Returns:
      self : TabFMRegressor
          Fitted regressor instance.
    """
    y = check_array(y, ensure_2d=False, dtype="numeric")
    self.X_encoder_ = TransformToNumerical(
        verbose=self.verbose, cat_encoder_mode=self.cat_encoder_mode
    )
    X = self.X_encoder_.fit_transform(X)

    if hasattr(self.X_encoder_.tfm_, "transformers_"):
      n_cat = len(getattr(self.X_encoder_.tfm_, "transformers_")[0][2])
    else:
      n_cat = 0
    cat_features = list(range(n_cat))

    self.y_scaler_ = StandardScaler()
    y = self.y_scaler_.fit_transform(y.reshape(-1, 1)).flatten()

    self.ensemble_generator_ = EnsembleGenerator(
        n_estimators=self.n_estimators,
        norm_methods=self.norm_methods or ["none", "power"],
        feat_shuffle_method=self.feat_shuffle_method,
        class_shift=False,
        cat_features=cat_features,
        permute_categorical=self.permute_categorical,
        outlier_threshold=self.outlier_threshold,
        random_state=self.random_state,
        task="regression",
    )
    self.ensemble_generator_.fit(X, y)
    return self

  @jt.typed
  def _batch_forward(
      self,
      Xs: jt.Float[jax.Array | np.ndarray, "B T H"],
      ys: jt.Shaped[jax.Array | np.ndarray, "B T_train"],
      cat_masks: Optional[jt.Bool[jax.Array | np.ndarray, "B H"]] = None,
  ) -> jt.Float[jax.Array | np.ndarray, "B T_test L_out"]:
    """Process model forward passes in batches to manage memory efficiently.

    Args:
      Xs: Input features of shape (n_datasets, n_samples, n_features).
      ys: Training labels of shape (n_datasets, train_size).
      cat_masks: Optional boolean mask of shape (n_datasets, n_features)
        indicating which features are categorical (True) vs. numerical (False).
        The model uses this to apply feature-type-specific processing: for
        example, categorical features may use different Fourier frequencies or
        random embeddings compared to numerical features. If None, all features
        are treated as numerical.
    Returns:
      Model outputs of shape (n_datasets, n_test, output_dim).
    """
    mesh = jax.sharding.get_mesh()
    if mesh and "data" in mesh.axis_names:
      num_data_shards = mesh.axis_sizes[mesh.axis_names.index("data")]
      data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    else:
      num_data_shards = 1
      data_sharding = None

    _has_compiled_attr = (
        "_predict_step_compiled_with_cat"
        if (cat_masks is not None and hasattr(self.model, "cell_embedder"))
        else "_predict_step_compiled_no_cat"
    )

    if not hasattr(self, _has_compiled_attr):
      data_sharding = NamedSharding(
          jax.sharding.Mesh(jax.devices(), ("data",)), PartitionSpec("data")
      )

      if cat_masks is not None and hasattr(self.model, "cell_embedder"):

        @nnx.jit(
            in_shardings=(
                None,
                data_sharding,
                data_sharding,
                data_sharding,
                data_sharding,
                data_sharding,
            ),
            out_shardings=data_sharding,
        )
        def _predict_step_fn(model, X, y, train_size, d, cat_mask):
          return model(
              X,
              y,
              train_size=train_size,
              d=d,
              cat_mask=cat_mask,
          )

      else:

        @nnx.jit(
            in_shardings=(
                None,
                data_sharding,
                data_sharding,
                data_sharding,
                data_sharding,
            ),
            out_shardings=data_sharding,
        )
        def _predict_step_fn(model, X, y, train_size, d):
          return model(
              X,
              y,
              train_size=train_size,
              d=d,
          )

      setattr(self, _has_compiled_attr, _predict_step_fn)

    _predict_step_compiled = getattr(self, _has_compiled_attr)
    batch_size_per_process = getattr(self, "batch_size", 1) or Xs.shape[0]
    n_batches = math.ceil(Xs.shape[0] / batch_size_per_process)
    if n_batches > 1:
      Xs_split = np.array_split(Xs, n_batches)
      ys_split = np.array_split(ys, n_batches)
      if cat_masks is not None:
        cat_masks_split = np.array_split(cat_masks, n_batches)
    else:
      Xs_split, ys_split = [Xs], [ys]
      if cat_masks is not None:
        cat_masks_split = [cat_masks]

    outputs = []
    cat_masks_iter = cat_masks_split if cat_masks is not None else [None] * len(Xs_split)
    for X_batch, y_batch, cat_mask_batch in zip(Xs_split, ys_split, cat_masks_iter):
      orig_batch_size = X_batch.shape[0]
      X_batch = _pad_batch_to_multiple_of(X_batch, num_data_shards)
      y_batch = _pad_batch_to_multiple_of(y_batch, num_data_shards)

      X_batch = jax.device_put(jnp.array(X_batch, dtype=jnp.float32), data_sharding)
      y_batch = jax.device_put(jnp.array(y_batch, dtype=jnp.float32), data_sharding)
      batch_size_padded = X_batch.shape[0]
      train_size_val = y_batch.shape[1]
      train_size = jax.device_put(
          jnp.repeat(train_size_val, batch_size_padded), data_sharding
      )
      d_batch = jax.device_put(
          jnp.full((batch_size_padded,), X_batch.shape[-1], dtype=jnp.int32),
          data_sharding,
      )

      if y_batch.shape[1] < X_batch.shape[1]:
        y_batch = jnp.pad(
            y_batch,
            ((0, 0), (0, X_batch.shape[1] - y_batch.shape[1])),
            constant_values=-100.0,
        )



      if cat_mask_batch is not None and hasattr(self.model, "cell_embedder"):
        cat_mask_batch = _pad_batch_to_multiple_of(cat_mask_batch, num_data_shards)
        cat_mask_batch = jax.device_put(
            jnp.array(cat_mask_batch, dtype=jnp.bool_), data_sharding
        )
        out = _predict_step_compiled(
            self.model, X_batch, y_batch, train_size, d_batch, cat_mask_batch
        )
      else:
        out = _predict_step_compiled(
            self.model, X_batch, y_batch, train_size, d_batch
        )

      out = out[:orig_batch_size, train_size_val:, :]
      from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top
      out = multihost_utils.process_allgather(out, tiled=True)
      outputs.append(out)

    return np.concatenate(outputs, axis=0)

  @jt.typed
  def predict(self, X: Any) -> jt.Float[jax.Array | np.ndarray, "T"]:
    """Predict regression target for test samples.

    Applies the ensemble of TabFM models to make predictions, with each
    ensemble member providing predictions that are then averaged.

    Args:
      X : array-like of shape (n_samples, n_features)
          Test samples for prediction.

    Returns:
      np.ndarray of shape (n_samples,)
          Predicted target values for each test sample.
    """
    if isinstance(X, np.ndarray) and len(X.shape) == 1:
      raise ValueError("The provided input X is one-dimensional. Reshape your data.")

    X = self.X_encoder_.transform(X)
    data = self.ensemble_generator_.transform(X)
    Xs_all, ys_all, cat_masks_all = [], [], []
    for norm_method, (Xs, ys) in data.items():
      Xs_all.append(Xs)
      ys_all.append(ys)
      configs = self.ensemble_generator_.ensemble_configs_[norm_method]
      for shuffle_pattern, _, _ in configs:
        mask = np.zeros(self.ensemble_generator_.n_features_in_, dtype=np.bool_)
        if hasattr(self.ensemble_generator_, "cat_features_"):
          mask[self.ensemble_generator_.cat_features_] = True
        cat_masks_all.append(mask[shuffle_pattern])

    Xs_all = np.concatenate(Xs_all, axis=0)
    ys_all = np.concatenate(ys_all, axis=0)
    cat_masks_all = np.stack(cat_masks_all, axis=0)

    output = self._batch_forward(Xs_all, ys_all, cat_masks_all)
    loss = self.model.loss if hasattr(self.model, "loss") else (self.config.loss if self.config else "mse")
    if loss == "rmse" or loss == "mse":
      predictions = output.squeeze(-1)
    else:
      raise ValueError(
          f"Unsupported loss for regression predict: {loss}"
      )

    avg_predictions = np.mean(predictions, axis=0)
    return self.y_scaler_.inverse_transform(avg_predictions.reshape(-1, 1)).flatten()

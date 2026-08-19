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

"""Tests for sequence-parallel JAX inference.

These run on simulated CPU devices (two host devices requested before JAX
initializes, which is why this lives in its own module), so they exercise the
sharded forward -- including the cross-device softmax combine and the padded,
bias-masked K/V gathers -- in CI without GPUs. Predictions are compared
against the estimators' plain single-device prediction paths.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

try:
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from tabfm.src.jax import model as tabfm_model
  from tabfm.src.jax import seqpar

  # Full-precision fp32 matmuls so the comparison against the single-device
  # path stays within tolerance on GPU dev machines too (XLA otherwise uses
  # TF32 for float32 matmuls on Ampere+).
  jax.config.update("jax_default_matmul_precision", "highest")

  HAS_JAX = True
except ImportError:
  HAS_JAX = False

from tabfm.src.classifier_and_regressor import TabFMClassifier
from tabfm.src.classifier_and_regressor import TabFMRegressor

# pylint: disable=invalid-name


def _tiny_model(loss):
  return tabfm_model.TabFM(
      loss=loss,
      max_classes=2,
      embed_dim=8,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=8,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=1,
      icl_num_blocks=1,
      icl_nhead=2,
      dtype=jnp.float32,
      rngs=nnx.Rngs(0),
  )


@absltest.skipIf(not HAS_JAX, "JAX not installed")
class SeqparPredictTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("regressor_even", False, 20, 10, 1),
      ("regressor_ragged", False, 21, 9, 1),
      ("regressor_two_members", False, 20, 10, 2),
      ("classifier_even", True, 20, 10, 1),
      ("classifier_ragged", True, 23, 7, 2),
  )
  def test_matches_single_device(
      self, is_classifier, n_train, n_test, n_estimators
  ):
    rng = np.random.RandomState(0)
    X = rng.rand(n_train + n_test, 4)
    if is_classifier:
      y = rng.randint(0, 2, size=n_train)
      est = TabFMClassifier(
          model=_tiny_model("cross_entropy"),
          n_estimators=n_estimators,
          random_state=0,
      )
    else:
      y = X[:n_train] @ rng.rand(4)
      est = TabFMRegressor(
          model=_tiny_model("mse"), n_estimators=n_estimators, random_state=0
      )
    est.fit(X[:n_train], y)

    if is_classifier:
      ref = est.predict_proba(X[n_train:])
      out = seqpar.predict_proba(est, X[n_train:])
    else:
      ref = est.predict(X[n_train:])
      out = seqpar.predict(est, X[n_train:])

    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()

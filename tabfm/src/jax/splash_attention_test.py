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

"""Tests for the SPLASH attention implementation.

The splash kernel is TPU-only; these tests run it in Pallas interpret mode
(CPU-executable, exact kernel semantics) and compare against the stock JAX
attention implementation on identical weights and inputs, with and without a
key-prefix mask. This validates the segment-id mask translation and layout
handling in CI; kernel performance is only measurable on real TPUs.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

try:
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from tabfm.src.jax import model as tabfm_model

  jax.config.update("jax_default_matmul_precision", "highest")
  HAS_JAX = True
except ImportError:
  HAS_JAX = False


def _mha(impl):
  return tabfm_model.MultiheadAttention(
      embed_dim=16,
      num_heads=2,
      attention_impl=impl,
      dtype=jnp.float32,
      rngs=nnx.Rngs(0),
  )


@absltest.skipIf(not HAS_JAX, "JAX not installed")
class SplashAttentionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tabfm_model.set_splash_interpret(True)
    self.addCleanup(tabfm_model.set_splash_interpret, False)

  @parameterized.named_parameters(
      ("full", None),
      ("prefix_mask", 96),
  )
  def test_matches_jax_attention(self, prefix):
    rng = np.random.RandomState(0)
    tgt_len, src_len = 128, 256
    query = jnp.asarray(rng.randn(1, tgt_len, 16), jnp.float32)
    key = jnp.asarray(rng.randn(1, src_len, 16), jnp.float32)
    mask = None
    if prefix is not None:
      mask = (jnp.arange(src_len)[None, :] < prefix)[:, None, None, :]

    ref = _mha(tabfm_model.AttentionImplementation.JAX)(
        query, key, key, attn_mask=mask
    )
    out = _mha(tabfm_model.AttentionImplementation.SPLASH)(
        query, key, key, attn_mask=mask
    )
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(ref), rtol=1e-4, atol=1e-4
    )


if __name__ == "__main__":
  absltest.main()

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

"""Release gate: MLX vs PyTorch on the REAL v1.0.0 checkpoint, in float32.

model_test.py checks parity on small randomly initialized models, which pins
the wiring but not the released weights: a wrong tensor in the conversion, a
transposed projection or a mis-set config key can pass there and still be
wrong on the 1.6B checkpoint. This runs the shipped estimators end to end
against PyTorch on CPU in float32 -- the reference both bfloat16 backends are
approximating -- and covers the uncached forward, the cached prefill/decode
path at a context length that is not a multiple of 128 (MLX skips the
sharding padding the other backends apply), and the int8 KV cache.

Expensive: downloads/uses the ~6 GB checkpoint, holds two copies of the model
(~13 GB RSS) and takes a few minutes, so it is opt-in:

  TABFM_REAL_PARITY=1 python -m pytest tabfm/src/mlx/real_checkpoint_parity_test.py
"""

import os
import unittest

import numpy as np
import pandas as pd

try:
  import torch  # noqa: F401

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

try:
  import mlx.core as mx
  import tabfm
  from tabfm.src.mlx import tabfm_v1_0_0 as mlx_ckpt

  HAS_MLX = True
except ImportError:  # MLX ships macOS/arm64 wheels only.
  HAS_MLX = False

RUN = os.environ.get("TABFM_REAL_PARITY") == "1"

# Deliberately not a multiple of 128: exercises the unpadded MLX prefill.
N_TRAIN, N_TEST = 117, 20
# float32 vs float32 on identical weights: differences are reassociation only.
FP32_TOL = 1e-5
# int8 K/V is a lossy cache; this bounds how lossy it is on real weights.
INT8_TOL = 5e-3


def _dataset(seed=3):
  rng = np.random.default_rng(seed)
  n = N_TRAIN + N_TEST
  cols = {f"n{i}": rng.normal(size=n) for i in range(6)}
  cols["c0"] = rng.choice(list("abcd"), size=n)
  X = pd.DataFrame(cols)
  y = rng.choice(["A", "B", "C"], size=n)
  return X.iloc[:N_TRAIN], y[:N_TRAIN], X.iloc[N_TRAIN:]


@unittest.skipUnless(HAS_MLX, "mlx is required (Apple silicon only)")
@unittest.skipUnless(
    RUN, "set TABFM_REAL_PARITY=1 (slow, needs the 6 GB checkpoint)"
)
@unittest.skipUnless(HAS_TORCH, "torch is required for the reference")
class RealCheckpointParityTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    from tabfm.src.pytorch import tabfm_v1_0_0 as pt_ckpt

    cls.Xtr, cls.ytr, cls.Xte = _dataset()
    # dtype=None keeps the checkpoint's float32 storage: the reference.
    torch_model = pt_ckpt.load(
        model_type="classification", device="cpu", dtype=None
    )
    cls.ref_uncached = cls._proba(torch_model)
    cls.ref_cached = cls._proba(
        torch_model, cache_context=True, maybe_quantize_kv_cache=False
    )
    del torch_model
    pt_ckpt._LOAD_CACHE.clear()
    cls.mlx_model = mlx_ckpt.load(
        model_type="classification", dtype=mx.float32, use_cache=False
    )

  @classmethod
  def _proba(cls, model, **kwargs):
    clf = tabfm.TabFMClassifier(
        model=model, n_estimators=2, random_state=0, **kwargs
    )
    clf.fit(cls.Xtr, cls.ytr)
    return clf.predict_proba(cls.Xte)

  def test_torch_cached_matches_torch_uncached(self):
    """Sanity: the reference's own two paths agree, so it can be a gold."""
    np.testing.assert_allclose(
        self.ref_cached, self.ref_uncached, rtol=FP32_TOL, atol=FP32_TOL
    )

  def test_uncached_forward_matches_torch_fp32(self):
    got = self._proba(self.mlx_model)
    print(
        "\nreal-checkpoint fp32, uncached: "
        f"max abs diff = {np.abs(got - self.ref_uncached).max():.3e}"
    )
    np.testing.assert_allclose(
        got, self.ref_uncached, rtol=FP32_TOL, atol=FP32_TOL
    )

  def test_cached_decode_matches_torch_fp32(self):
    """Unpadded MLX prefill/decode vs the padded PyTorch forward."""
    got = self._proba(
        self.mlx_model, cache_context=True, maybe_quantize_kv_cache=False
    )
    print(
        f"\nreal-checkpoint fp32, cached (T={N_TRAIN}, not 128|T): "
        f"max abs diff = {np.abs(got - self.ref_cached).max():.3e}"
    )
    np.testing.assert_allclose(
        got, self.ref_cached, rtol=FP32_TOL, atol=FP32_TOL
    )

  def test_cross_member_batching_matches_per_member(self):
    """mlx_batch_size must not change predictions on the real checkpoint."""
    merged = self._proba(
        self.mlx_model,
        cache_context=True,
        maybe_quantize_kv_cache=False,
        mlx_batch_size=None,
    )
    per_member = self._proba(
        self.mlx_model,
        cache_context=True,
        maybe_quantize_kv_cache=False,
        mlx_batch_size=1,
    )
    np.testing.assert_allclose(merged, per_member, rtol=FP32_TOL, atol=FP32_TOL)

  def test_int8_kv_cache_stays_within_tolerance(self):
    """The quantized cache is lossy; keep the loss bounded and measured."""
    got = self._proba(
        self.mlx_model, cache_context=True, maybe_quantize_kv_cache=True
    )
    diff = float(np.abs(got - self.ref_cached).max())
    print(
        f"\nreal-checkpoint fp32, cached + int8 KV: max abs diff = {diff:.3e}"
    )
    self.assertLess(diff, INT8_TOL)
    self.assertGreater(
        diff, FP32_TOL, "int8 cache appears to be a no-op; is it applied?"
    )


if __name__ == "__main__":
  unittest.main()

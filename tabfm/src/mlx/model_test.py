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

"""Parity tests: MLX backend vs PyTorch backend (torch weights copied over).

Gates for the MLX port (mirrors the JAX<->PyTorch parity approach):
  - forward parity (classifier + regression, with cat_mask and d): the release
    gate, must match to ~1e-4 in float32.
  - prefill/decode self-consistency: decode(prefill(train), test) matches the
    full forward on train+test (validates the cache path for Phase 3).
  - cross-backend cache check: MLX decode matches the torch full forward.

Requires torch + mlx (CI: CPU-only torch is fine). Skipped if torch is absent.
"""

import unittest

import numpy as np

try:
  import torch
  from tabfm.src.pytorch import model as torch_model_mod

  HAS_TORCH = True
except ImportError:  # MLX-only environments (e.g. the timesfm-style venv).
  HAS_TORCH = False

try:
  import mlx.core as mx
  from tabfm.src.mlx import model as mlx_model_mod

  HAS_MLX = True
except ImportError:  # MLX ships macOS/arm64 wheels only.
  HAS_MLX = False

try:
  from tabfm.src.classifier_and_regressor import _concat_caches_mlx

  HAS_ESTIMATOR = True
except ImportError:
  HAS_ESTIMATOR = False


CFG = dict(
    embed_dim=32,
    max_classes=4,
    col_num_blocks=2,
    col_nhead=4,
    col_num_inds=16,
    row_num_blocks=2,
    row_nhead=4,
    row_num_cls=4,
    icl_num_blocks=3,
    icl_nhead=4,
    ff_factor=4,
    feature_group_size=3,
)


def _copy_torch_to_mlx(torch_model, mlx_model):
  """Copies torch params+buffers into the mirror MLX module (same key names)."""
  weights = [
      (name, mx.array(p.detach().cpu().numpy()))
      for name, p in list(torch_model.named_parameters()) + list(
          torch_model.named_buffers()
      )
  ]
  mlx_model.load_weights(weights)
  mx.eval(mlx_model.parameters())
  return [n for n, _ in weights]


def _random_inputs(rng, b=3, t=5, h=8, n_classes=4, is_classifier=True):
  """Builds a random (x, y, cat_mask, d) input tuple for the test models."""
  x_np = rng.normal(size=(b, t, h)).astype(np.float32)
  if is_classifier:
    y_np = rng.integers(0, n_classes, size=(b, t)).astype(np.float32)
  else:
    y_np = rng.normal(size=(b, t)).astype(np.float32)
  train_size_np = np.array([2, 3, 4][:b], dtype=np.int32)
  d_np = np.array([5, 6, 7][:b], dtype=np.int32)  # active counts (d < h)
  cat_mask_np = np.zeros((b, h), dtype=bool)
  cat_mask_np[0, :3] = True
  if b > 1:
    cat_mask_np[1, :4] = True
  return x_np, y_np, train_size_np, d_np, cat_mask_np


@unittest.skipUnless(HAS_MLX, "mlx is required (Apple silicon only)")
@unittest.skipUnless(HAS_TORCH, "torch is required for parity tests")
class MlxParityTest(unittest.TestCase):

  def _build_pair(self, is_classifier):
    torch_model = torch_model_mod.TabFM(is_classifier=is_classifier, **CFG)
    torch_model.eval()
    mlx_model = mlx_model_mod.TabFM(is_classifier=is_classifier, **CFG)
    keys = _copy_torch_to_mlx(torch_model, mlx_model)
    # Every torch leaf must land in the MLX tree (guards name drift).
    mlx_keys = set(
        k
        for k, _ in __import__(
            "mlx.utils", fromlist=["tree_flatten"]
        ).tree_flatten(mlx_model.parameters())
    )
    self.assertEqual(set(keys), mlx_keys)
    return torch_model, mlx_model

  def _forward_pair(self, torch_model, mlx_model, inputs):
    x_np, y_np, train_size_np, d_np, cat_mask_np = inputs
    with torch.no_grad():
      torch_out = torch_model(
          torch.from_numpy(x_np),
          torch.from_numpy(y_np),
          torch.from_numpy(train_size_np),
          cat_mask=torch.from_numpy(cat_mask_np),
          d=torch.from_numpy(d_np),
      ).numpy()
    mlx_out = mlx_model(
        mx.array(x_np),
        mx.array(y_np),
        mx.array(train_size_np),
        cat_mask=mx.array(cat_mask_np),
        d=mx.array(d_np),
    )
    mx.eval(mlx_out)
    return torch_out, np.array(mlx_out)

  def test_forward_parity(self):
    """MLX forward matches torch forward (the release gate, ~1e-4 fp32)."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        torch_model, mlx_model = self._build_pair(is_classifier)
        rng = np.random.default_rng(123)
        torch_out, mlx_out = self._forward_pair(
            torch_model,
            mlx_model,
            _random_inputs(
                rng, is_classifier=is_classifier, n_classes=CFG["max_classes"]
            ),
        )
        self.assertEqual(torch_out.shape, mlx_out.shape)
        max_abs = np.max(np.abs(torch_out - mlx_out))
        print(
            f"\nforward parity is_classifier={is_classifier}: "
            f"max abs diff = {max_abs:.3e}"
        )
        np.testing.assert_allclose(mlx_out, torch_out, rtol=1e-4, atol=1e-4)

  def test_prefill_decode_consistency(self):
    """decode(prefill(train), test) == forward(train+test); MLX and cross."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        torch_model, mlx_model = self._build_pair(is_classifier)
        rng = np.random.default_rng(7)
        b, t_tr, t_te, h = 2, 5, 3, 6
        x_tr = rng.normal(size=(b, t_tr, h)).astype(np.float32)
        x_te = rng.normal(size=(b, t_te, h)).astype(np.float32)
        if is_classifier:
          y_tr = rng.integers(0, CFG["max_classes"], size=(b, t_tr)).astype(
              np.float32
          )
        else:
          y_tr = rng.normal(size=(b, t_tr)).astype(np.float32)
        d_np = np.array([4, 5], dtype=np.int32)
        cat_mask_np = np.zeros((b, h), dtype=bool)
        cat_mask_np[0, :2] = True

        # Reference: full forward on train rows + (-100-padded) test rows.
        x_full = np.concatenate([x_tr, x_te], axis=1)
        y_full = np.concatenate(
            [y_tr, np.full((b, t_te), -100.0, dtype=np.float32)], axis=1
        )
        ts_full = np.full((b,), t_tr, dtype=np.int32)
        torch_ref, mlx_ref = self._forward_pair(
            torch_model, mlx_model, (x_full, y_full, ts_full, d_np, cat_mask_np)
        )
        np.testing.assert_allclose(mlx_ref, torch_ref, rtol=1e-4, atol=1e-4)

        # MLX prefill on train, decode on test.
        logits, cache = mlx_model.prefill(
            mx.array(x_tr),
            mx.array(y_tr),
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        dec = mlx_model.decode(
            mx.array(x_te),
            cache,
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        mx.eval(logits, dec)
        dec_np = np.array(dec)
        # Decode covers test rows only: compare against the tail of forward.
        ref_tail = mlx_ref[:, t_tr:, :]
        max_abs = np.max(np.abs(dec_np - ref_tail))
        print(
            f"\nprefill/decode self-consistency is_classifier={is_classifier}:"
            f" max abs diff = {max_abs:.3e}"
        )
        np.testing.assert_allclose(dec_np, ref_tail, rtol=1e-4, atol=1e-4)
        # Cross-backend: MLX decode matches the torch full forward tail.
        np.testing.assert_allclose(
            dec_np, torch_ref[:, t_tr:, :], rtol=1e-4, atol=1e-4
        )

  def test_reference_padding_is_inert(self):
    """The 128-row padding cannot carry information into the real rows.

    Licenses the MLX backend skipping it. The reference pads prefill/decode
    to a multiple of 128 with the -100.0 sentinel; train_size counts
    non-sentinel labels, so padded rows should be excluded everywhere. Here
    the caller pre-pads to 128 -- once with the sentinel and once with
    garbage feature values -- and the real-row logits must not move at all.
    """
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        torch_model, _ = self._build_pair(is_classifier)
        rng = np.random.default_rng(1234)
        b, h, t_tr, t_te = 2, 6, 100, 20
        pad = 128 - t_tr
        d_np = np.array([4, 5], dtype=np.int32)
        cat_mask_np = np.zeros((b, h), dtype=bool)
        cat_mask_np[0, :2] = True
        x_tr = rng.normal(size=(b, t_tr, h)).astype(np.float32)
        x_te = rng.normal(size=(b, t_te, h)).astype(np.float32)
        if is_classifier:
          y_tr = rng.integers(0, CFG["max_classes"], size=(b, t_tr)).astype(
              np.float32
          )
        else:
          y_tr = rng.normal(size=(b, t_tr)).astype(np.float32)
        y_pad = np.concatenate(
            [y_tr, np.full((b, pad), -100.0, dtype=np.float32)], axis=1
        )

        def decode(xtr, ytr):
          with torch.no_grad():
            _, cache = torch_model.prefill(
                torch.tensor(xtr),
                torch.tensor(ytr),
                cat_mask=torch.tensor(cat_mask_np),
                d=torch.tensor(d_np),
            )
            return torch_model.decode(
                torch.tensor(x_te),
                cache,
                cat_mask=torch.tensor(cat_mask_np),
                d=torch.tensor(d_np),
            ).numpy()

        internal = decode(x_tr, y_tr)  # model pads 100 -> 128 itself
        sentinel = decode(
            np.concatenate(
                [x_tr, np.full((b, pad, h), -100.0, dtype=np.float32)], axis=1
            ),
            y_pad,
        )
        garbage = decode(
            np.concatenate(
                [x_tr, rng.normal(size=(b, pad, h)).astype(np.float32) * 50],
                axis=1,
            ),
            y_pad,
        )
        # Bit-identical, not merely close: padded rows are masked out, so no
        # value placed in them may perturb the result.
        np.testing.assert_array_equal(internal, sentinel)
        np.testing.assert_array_equal(internal, garbage)

  def test_unpadded_matches_torch_at_awkward_lengths(self):
    """MLX prefill/decode (no 128-padding) matches torch (padded) forward.

    Regression gate for the MLX no-padding divergence: the JAX/PyTorch
    backends pad prefill/decode sequences to multiples of 128 for sharding,
    MLX skips it. Padded rows are masked/per-row independent everywhere, so
    real-row values must agree at any length, including non-multiples.
    """
    for is_classifier in [True, False]:
      torch_model, mlx_model = self._build_pair(is_classifier)
      rng = np.random.default_rng(99)
      b, h = 2, 6
      d_np = np.array([4, 5], dtype=np.int32)
      cat_mask_np = np.zeros((b, h), dtype=bool)
      cat_mask_np[0, :2] = True
      for t_tr, t_te in [(100, 100), (100, 20), (5, 100), (1, 1), (127, 129)]:
        with self.subTest(is_classifier=is_classifier, t_tr=t_tr, t_te=t_te):
          x_tr = rng.normal(size=(b, t_tr, h)).astype(np.float32)
          x_te = rng.normal(size=(b, t_te, h)).astype(np.float32)
          if is_classifier:
            y_tr = rng.integers(0, CFG["max_classes"], size=(b, t_tr)).astype(
                np.float32
            )
          else:
            y_tr = rng.normal(size=(b, t_tr)).astype(np.float32)
          x_full = np.concatenate([x_tr, x_te], axis=1)
          y_full = np.concatenate(
              [y_tr, np.full((b, t_te), -100.0, dtype=np.float32)], axis=1
          )
          ts_full = np.full((b,), t_tr, dtype=np.int32)
          torch_ref, _ = self._forward_pair(
              torch_model,
              mlx_model,
              (x_full, y_full, ts_full, d_np, cat_mask_np),
          )
          logits, cache = mlx_model.prefill(
              mx.array(x_tr),
              mx.array(y_tr),
              cat_mask=mx.array(cat_mask_np),
              d=mx.array(d_np),
          )
          dec = mlx_model.decode(
              mx.array(x_te),
              cache,
              cat_mask=mx.array(cat_mask_np),
              d=mx.array(d_np),
          )
          mx.eval(logits, dec)
          dec_np = np.array(dec)
          max_abs = np.max(np.abs(dec_np - torch_ref[:, t_tr:, :]))
          print(
              f"\nunpadded-vs-torch cls={is_classifier} "
              f"t_tr={t_tr} t_te={t_te}: max abs diff = {max_abs:.3e}"
          )
          np.testing.assert_allclose(
              dec_np, torch_ref[:, t_tr:, :], rtol=1e-4, atol=1e-4
          )


@unittest.skipUnless(HAS_MLX, "mlx is required (Apple silicon only)")
class MlxSelfConsistencyTest(unittest.TestCase):
  """Torch-free tests: padding edges, quantization, cache concatenation."""

  def test_padding_edges(self):
    """Prefill/decode self-consistency at awkward test lengths.

    Covers pad_len=0 (T=128, exact multiple), tiny (T=1), and multi-block
    (T=200) decodes against the full-forward tail.
    """
    for is_classifier in [True, False]:
      mlx_model = mlx_model_mod.TabFM(is_classifier=is_classifier, **CFG)
      rng = np.random.default_rng(11)
      b, t_tr, h = 2, 5, 6
      x_tr = rng.normal(size=(b, t_tr, h)).astype(np.float32)
      y_tr = (
          rng.integers(0, CFG["max_classes"], size=(b, t_tr)).astype(np.float32)
          if is_classifier
          else rng.normal(size=(b, t_tr)).astype(np.float32)
      )
      d_np = np.array([4, 5], dtype=np.int32)
      cat_mask_np = np.zeros((b, h), dtype=bool)
      for t_te in [1, 20, 128, 200]:
        with self.subTest(is_classifier=is_classifier, t_te=t_te):
          x_te = rng.normal(size=(b, t_te, h)).astype(np.float32)
          x_full = np.concatenate([x_tr, x_te], axis=1)
          y_full = np.concatenate(
              [y_tr, np.full((b, t_te), -100.0, dtype=np.float32)], axis=1
          )
          ts_full = np.full((b,), t_tr, dtype=np.int32)
          ref = mlx_model(
              mx.array(x_full),
              mx.array(y_full),
              mx.array(ts_full),
              cat_mask=mx.array(cat_mask_np),
              d=mx.array(d_np),
          )
          logits, cache = mlx_model.prefill(
              mx.array(x_tr),
              mx.array(y_tr),
              cat_mask=mx.array(cat_mask_np),
              d=mx.array(d_np),
          )
          dec = mlx_model.decode(
              mx.array(x_te),
              cache,
              cat_mask=mx.array(cat_mask_np),
              d=mx.array(d_np),
          )
          mx.eval(ref, logits, dec)
          np.testing.assert_allclose(
              np.array(dec), np.array(ref)[:, t_tr:, :], rtol=1e-4, atol=1e-4
          )

  def test_quantized_decode_parity(self):
    """int8-quantized cache decode matches full-precision decode (~5e-3)."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        mlx_model = mlx_model_mod.TabFM(is_classifier=is_classifier, **CFG)
        rng = np.random.default_rng(13)
        b, t_tr, t_te, h = 2, 5, 20, 6
        x_tr = rng.normal(size=(b, t_tr, h)).astype(np.float32)
        x_te = rng.normal(size=(b, t_te, h)).astype(np.float32)
        y_tr = (
            rng.integers(0, CFG["max_classes"], size=(b, t_tr)).astype(
                np.float32
            )
            if is_classifier
            else rng.normal(size=(b, t_tr)).astype(np.float32)
        )
        d_np = np.array([4, 5], dtype=np.int32)
        cat_mask_np = np.zeros((b, h), dtype=bool)
        _, cache = mlx_model.prefill(
            mx.array(x_tr),
            mx.array(y_tr),
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        dec = mlx_model.decode(
            mx.array(x_te),
            cache,
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        qcache = {
            "col1": cache["col1"],
            "col2": cache["col2"],
            "icl": cache["icl"].quantize(),
        }
        decq = mlx_model.decode(
            mx.array(x_te),
            qcache,
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        mx.eval(dec, decq)
        max_abs = np.max(np.abs(np.array(decq) - np.array(dec)))
        print(
            f"\nquantized decode parity is_classifier={is_classifier}: "
            f"max abs diff = {max_abs:.3e}"
        )
        np.testing.assert_allclose(
            np.array(decq), np.array(dec), rtol=5e-3, atol=5e-3
        )

  @unittest.skipUnless(HAS_ESTIMATOR, "estimator module required")
  def test_concat_caches_equivalence(self):
    """One batched decode over concatenated caches == per-group decodes."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        mlx_model = mlx_model_mod.TabFM(is_classifier=is_classifier, **CFG)
        rng = np.random.default_rng(17)
        t_tr, t_te, h = 5, 7, 6
        d_np = np.array([4, 5], dtype=np.int32)
        cat_mask_np = np.zeros((2, h), dtype=bool)
        caches, decs = [], []
        for m in range(2):
          x_tr = rng.normal(size=(1, t_tr, h)).astype(np.float32)
          x_te = rng.normal(size=(1, t_te, h)).astype(np.float32)
          if is_classifier:
            y_tr = rng.integers(0, CFG["max_classes"], size=(1, t_tr)).astype(
                np.float32
            )
          else:
            y_tr = rng.normal(size=(1, t_tr)).astype(np.float32)
          _, cache = mlx_model.prefill(
              mx.array(x_tr),
              mx.array(y_tr),
              cat_mask=mx.array(cat_mask_np[m : m + 1]),
              d=mx.array(d_np[m : m + 1]),
          )
          dec = mlx_model.decode(
              mx.array(x_te),
              cache,
              cat_mask=mx.array(cat_mask_np[m : m + 1]),
              d=mx.array(d_np[m : m + 1]),
          )
          mx.eval(dec)
          caches.append(cache)
          decs.append(np.array(dec))
          if m == 0:
            x_te_all = x_te
          else:
            x_te_all = np.concatenate([x_te_all, x_te], axis=0)
        big = _concat_caches_mlx(caches, mlx_model.cls_tokens.dtype)
        dec_big = mlx_model.decode(
            mx.array(x_te_all),
            big,
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        mx.eval(dec_big)
        np.testing.assert_allclose(
            np.array(dec_big),
            np.concatenate(decs, axis=0),
            rtol=1e-5,
            atol=1e-5,
        )

  def test_quantize_all_zero_tensor(self):
    """An all-zero K/V must quantize to a positive scale, not 0/0."""
    q = mlx_model_mod._quantize_tensor(mx.zeros((2, 4, 2, 3)))
    mx.eval(q.data, q.scale)
    self.assertGreater(float(q.scale), 0.0)
    deq = np.array(q.dequantize(mx.float32))
    self.assertTrue(np.all(np.isfinite(deq)))
    np.testing.assert_array_equal(deq, np.zeros_like(deq))

  @unittest.skipUnless(HAS_ESTIMATOR, "estimator module required")
  def test_concat_quantized_caches_stays_quantized(self):
    """Merging int8 caches keeps them int8 and does not change decode values.

    Cross-member batching used to dequantize every layer up front, so the
    merged cache cost a full-precision copy of the whole KV cache -- the
    memory that quantization was supposed to save. Codes now concatenate
    directly, with one scale per member.
    """
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        mlx_model = mlx_model_mod.TabFM(is_classifier=is_classifier, **CFG)
        rng = np.random.default_rng(23)
        t_tr, t_te, h = 6, 9, 6
        d_np = np.array([4, 5], dtype=np.int32)
        cat_mask_np = np.zeros((2, h), dtype=bool)
        caches, decs, x_tes = [], [], []
        for m in range(2):
          x_tr = rng.normal(size=(1, t_tr, h)).astype(np.float32)
          x_te = rng.normal(size=(1, t_te, h)).astype(np.float32)
          if is_classifier:
            y_tr = rng.integers(0, CFG["max_classes"], size=(1, t_tr)).astype(
                np.float32
            )
          else:
            y_tr = rng.normal(size=(1, t_tr)).astype(np.float32)
          _, cache = mlx_model.prefill(
              mx.array(x_tr),
              mx.array(y_tr),
              cat_mask=mx.array(cat_mask_np[m : m + 1]),
              d=mx.array(d_np[m : m + 1]),
          )
          cache["icl"] = cache["icl"].quantize()
          dec = mlx_model.decode(
              mx.array(x_te),
              cache,
              cat_mask=mx.array(cat_mask_np[m : m + 1]),
              d=mx.array(d_np[m : m + 1]),
          )
          mx.eval(dec)
          caches.append(cache)
          decs.append(np.array(dec))
          x_tes.append(x_te)

        big = _concat_caches_mlx(caches, mlx_model.cls_tokens.dtype)
        for k, v in big["icl"].layer_caches:
          for t in (k, v):
            self.assertIsInstance(
                t,
                mlx_model_mod.QuantizedTensor,
                "merged cache must stay quantized",
            )
            self.assertEqual(t.data.dtype, mx.int8)
            self.assertEqual(
                t.scale.shape,
                (2, 1, 1, 1),
                "one scale per member is what makes the merge "
                "exact without dequantizing",
            )
        dec_big = mlx_model.decode(
            mx.array(np.concatenate(x_tes, axis=0)),
            big,
            cat_mask=mx.array(cat_mask_np),
            d=mx.array(d_np),
        )
        mx.eval(dec_big)
        np.testing.assert_allclose(
            np.array(dec_big),
            np.concatenate(decs, axis=0),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
  unittest.main()

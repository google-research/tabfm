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

"""Loader tests for the MLX backend: config translation and the npz cache.

Runs against a tiny synthetic checkpoint (a few MB) rather than the 6 GB
release weights, so it is CI-fast while still covering safetensors -> npz
conversion, the staleness check, and the dtype keying of the cache file.
"""

import json
import os
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

try:
  import mlx.core as mx
  from mlx.utils import tree_flatten
  from tabfm.src.mlx import model as mlx_model_mod
  from tabfm.src.mlx import tabfm_v1_0_0 as loader

  HAS_MLX = True
except ImportError:  # MLX ships macOS/arm64 wheels only.
  HAS_MLX = False


CFG = dict(
    embed_dim=16,
    max_classes=4,
    col_num_blocks=1,
    col_nhead=2,
    col_num_inds=8,
    row_num_blocks=1,
    row_nhead=2,
    row_num_cls=2,
    icl_num_blocks=1,
    icl_nhead=2,
    ff_factor=2,
    feature_group_size=3,
)


def _write_checkpoint(dirname, is_classifier=True):
  """Writes a tiny model.safetensors + config.json, returns the source model."""
  model = mlx_model_mod.TabFM(is_classifier=is_classifier, **CFG)
  mx.eval(model.parameters())
  tensors = {
      k: np.array(v.astype(mx.float32))
      for k, v in tree_flatten(model.parameters())
  }
  save_file(tensors, os.path.join(dirname, "model.safetensors"))
  cfg = dict(CFG)
  cfg["task"] = "classification" if is_classifier else "regression"
  cfg["model_type"] = "tabfm"  # hub-only key the loader must drop
  with open(os.path.join(dirname, "config.json"), "w") as f:
    json.dump(cfg, f)
  return model


@unittest.skipUnless(HAS_MLX, "mlx is required (Apple silicon only)")
class ConvertCachePathTest(unittest.TestCase):

  def test_path_is_keyed_on_dtype(self):
    """Two dtypes are two artifacts: loading one must not evict the other."""
    a = loader._convert_cache_path(
        "google/tabfm", "classification", mx.bfloat16
    )
    b = loader._convert_cache_path("google/tabfm", "classification", mx.float32)
    c = loader._convert_cache_path("google/tabfm", "regression", mx.bfloat16)
    self.assertNotEqual(a, b)
    self.assertNotEqual(a, c)
    self.assertIn("bfloat16", os.path.basename(a))
    self.assertIn("float32", os.path.basename(b))

  def test_none_dtype_is_named_float32(self):
    """dtype=None keeps the checkpoint's float32 storage; name says so."""
    p = loader._convert_cache_path("google/tabfm", "classification", None)
    self.assertIn("float32", os.path.basename(p))


@unittest.skipUnless(HAS_MLX, "mlx is required (Apple silicon only)")
class LoadTest(unittest.TestCase):

  def setUp(self):
    self._home = os.environ.get("HOME")
    self._tmp = tempfile.TemporaryDirectory()
    os.environ["HOME"] = self._tmp.name  # redirect ~/.cache/tabfm_mlx

  def tearDown(self):
    if self._home is not None:
      os.environ["HOME"] = self._home
    self._tmp.cleanup()

  def _cache_files(self):
    d = os.path.join(self._tmp.name, ".cache", "tabfm_mlx")
    return sorted(f for f in os.listdir(d) if f.endswith(".npz"))

  def test_load_reproduces_source_weights(self):
    """Round-tripping through safetensors -> npz preserves the forward pass."""
    with tempfile.TemporaryDirectory() as ckpt:
      src = _write_checkpoint(ckpt)
      loaded = loader.load(
          checkpoint_path=ckpt, dtype=mx.float32, use_cache=False
      )
      rng = np.random.default_rng(0)
      x = mx.array(rng.normal(size=(2, 6, 5)).astype(np.float32))
      y = mx.array(
          rng.integers(0, CFG["max_classes"], size=(2, 6)).astype(np.float32)
      )
      ts = mx.array(np.array([3, 4], dtype=np.int32))
      out_src, out_loaded = src(x, y, ts), loaded(x, y, ts)
      mx.eval(out_src, out_loaded)
      np.testing.assert_allclose(
          np.array(out_loaded), np.array(out_src), rtol=1e-6, atol=1e-6
      )

  def test_second_load_reuses_npz_and_dtypes_coexist(self):
    """The conversion is once per dtype, and dtypes do not evict each other."""
    with tempfile.TemporaryDirectory() as ckpt:
      _write_checkpoint(ckpt)
      loader.load(checkpoint_path=ckpt, dtype=mx.float32, use_cache=False)
      self.assertEqual(len(self._cache_files()), 1)
      npz = os.path.join(
          self._tmp.name, ".cache", "tabfm_mlx", self._cache_files()[0]
      )
      stamp = os.stat(npz).st_mtime_ns

      loader.load(checkpoint_path=ckpt, dtype=mx.float32, use_cache=False)
      self.assertEqual(
          os.stat(npz).st_mtime_ns,
          stamp,
          "second load re-converted instead of reusing the npz",
      )

      loader.load(checkpoint_path=ckpt, dtype=mx.bfloat16, use_cache=False)
      self.assertEqual(
          len(self._cache_files()),
          2,
          "a second dtype must not overwrite the first",
      )
      self.assertEqual(os.stat(npz).st_mtime_ns, stamp)

  def test_stale_checkpoint_triggers_reconversion(self):
    """Rewriting the checkpoint invalidates the cached npz."""
    with tempfile.TemporaryDirectory() as ckpt:
      _write_checkpoint(ckpt)
      loader.load(checkpoint_path=ckpt, dtype=mx.float32, use_cache=False)
      npz = os.path.join(
          self._tmp.name, ".cache", "tabfm_mlx", self._cache_files()[0]
      )
      stamp = os.stat(npz).st_mtime_ns
      src2 = _write_checkpoint(ckpt)  # new random weights, same path
      loaded = loader.load(
          checkpoint_path=ckpt, dtype=mx.float32, use_cache=False
      )
      self.assertNotEqual(
          os.stat(npz).st_mtime_ns,
          stamp,
          "stale npz was reused after the checkpoint changed",
      )
      np.testing.assert_allclose(
          np.array(loaded.cls_tokens),
          np.array(src2.cls_tokens),
          rtol=1e-6,
          atol=1e-6,
      )

  def test_regression_task_from_config(self):
    """task=regression in config.json selects the regression head."""
    with tempfile.TemporaryDirectory() as ckpt:
      _write_checkpoint(ckpt, is_classifier=False)
      loaded = loader.load(
          checkpoint_path=ckpt, dtype=mx.float32, use_cache=False
      )
      self.assertFalse(loaded.is_classifier)

  def test_rejects_unknown_model_type(self):
    with self.assertRaises(ValueError):
      loader.load(model_type="clustering")

  def test_float16_is_rejected(self):
    """float16 silently produced NaN before this guard."""
    with self.assertRaisesRegex(ValueError, "float16 overflows"):
      loader.load(dtype=mx.float16)


if __name__ == "__main__":
  unittest.main()

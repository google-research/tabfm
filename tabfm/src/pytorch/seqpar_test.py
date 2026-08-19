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

"""Tests for sequence-parallel PyTorch inference.

These run with the gloo backend on CPU processes, so they exercise the
sharded forward (including the cross-rank softmax combine and padded K/V
gathers) in CI without needing GPUs. Outputs are compared against the plain
single-process model forward.
"""

import functools
import multiprocessing
import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch.distributed as dist

from tabfm.src.pytorch import model as tabfm_model
from tabfm.src.pytorch import seqpar

# pylint: disable=invalid-name

_WORLD = 2


def _tiny_model(is_classifier):
  torch.manual_seed(0)
  m = tabfm_model.TabFM(
      embed_dim=8,
      max_classes=3,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=4,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=2,
      icl_num_blocks=2,
      icl_nhead=2,
      is_classifier=is_classifier,
  )
  # Randomize parameters so the comparison is not trivially zeros.
  with torch.no_grad():
    for p in m.parameters():
      p.uniform_(-0.05, 0.05)
  return m.eval()


def _make_data(n_train, n_test, h, is_classifier, seed=0):
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((1, n_train + n_test, h)).astype(np.float32)
  if is_classifier:
    y_train = rng.integers(0, 2, n_train).astype(np.float32)
  else:
    y_train = rng.standard_normal(n_train).astype(np.float32)
  y = np.concatenate([y_train, np.full(n_test, -100.0)]).astype(np.float32)[
      None
  ]
  return x, y


def _reference_forward(model, x, y, n_train, cat_mask=None, d=None):
  ts = torch.tensor([n_train], dtype=torch.long)
  y_pad = torch.from_numpy(y)
  with torch.inference_mode():
    out = model(
        torch.from_numpy(x),
        y_pad,
        ts,
        cat_mask=torch.from_numpy(cat_mask) if cat_mask is not None else None,
        d=torch.from_numpy(d) if d is not None else None,
    )
  return out[0, n_train:, :].float().numpy()


def _worker(rank, port, is_classifier, n_train, n_test, h, use_cat_and_d, q):
  """Runs the sharded forward on one CPU rank and reports rank-0's result."""
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
  try:
    model = _tiny_model(is_classifier)
    x, y = _make_data(n_train, n_test, h, is_classifier)
    cat_mask = d = None
    if use_cat_and_d:
      cat_mask = np.zeros((1, h), dtype=bool)
      cat_mask[0, 0] = True
      d = np.array([h - 1], dtype=np.int64)  # last feature column is padding

    c0, c1 = seqpar._shard_bounds(n_train, _WORLD, rank)  # pylint: disable=protected-access
    t0, t1 = seqpar._shard_bounds(n_test, _WORLD, rank)  # pylint: disable=protected-access
    x_local = np.concatenate(
        [x[0, c0:c1], x[0, n_train + t0 : n_train + t1]], axis=0
    )[None]
    y_local = np.concatenate([y[0, c0:c1], np.full(t1 - t0, -100.0)])[None]
    out = seqpar.seqpar_forward(
        model, x_local, y_local, c1 - c0, cat_mask=cat_mask, d=d
    )
    gathered = [None] * _WORLD
    dist.all_gather_object(gathered, out)
    if rank == 0:
      q.put(np.concatenate(gathered, axis=0))
  finally:
    dist.destroy_process_group()


class SeqparForwardTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("regressor_even", False, 64, 16, 5, False),
      ("classifier_even", True, 64, 16, 5, False),
      ("regressor_ragged", False, 63, 15, 5, False),
      ("classifier_ragged", True, 61, 17, 5, False),
      ("regressor_cat_and_d", False, 64, 16, 5, True),
  )
  def test_matches_single_process(
      self, is_classifier, n_train, n_test, h, use_cat_and_d
  ):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + hash(self._testMethodName) % 1000
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, port, is_classifier, n_train, n_test, h, use_cat_and_d, q),
        )
        for r in range(_WORLD)
    ]
    for p in procs:
      p.start()
    sharded = q.get(timeout=120)
    for p in procs:
      p.join(timeout=120)
      self.assertEqual(p.exitcode, 0)

    model = _tiny_model(is_classifier)
    x, y = _make_data(n_train, n_test, h, is_classifier)
    cat_mask = d = None
    if use_cat_and_d:
      cat_mask = np.zeros((1, h), dtype=bool)
      cat_mask[0, 0] = True
      d = np.array([h - 1], dtype=np.int64)
    ref = _reference_forward(model, x, y, n_train, cat_mask=cat_mask, d=d)

    np.testing.assert_allclose(sharded, ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main()

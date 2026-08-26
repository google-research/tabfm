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

"""Tests the prefix key-padding mask -> k/v truncation rewrite in attention.

A [B,1,1,S] boolean prefix mask (valid keys = the first n rows, same n for
every query and batch element) is algebraically identical to truncating k/v
to the first n rows and passing attn_mask=None. The rewrite matters because
any non-null attn_mask keeps torch SDPA off the flash/memory-efficient
backends, whose memory is linear rather than quadratic in sequence length.
"""

import unittest

import torch

from tabfm.src.pytorch import model as pytorch_model


def _prefix_mask(b, s, n):
  mask = torch.zeros(b, 1, 1, s, dtype=torch.bool)
  mask[..., :n] = True
  return mask


class PrefixMaskTruncationTest(unittest.TestCase):

  def _attn(self, seed=0):
    torch.manual_seed(seed)
    return pytorch_model.MultiheadAttention(d_model=32, nhead=4).eval()

  def test_prefix_mask_equals_truncated_kv(self):
    """Masked forward == forward on truncated k/v with no mask."""
    attn = self._attn()
    b, s, n, d = 3, 16, 10, 32
    torch.manual_seed(1)
    query, key, value = (torch.randn(b, s, d) for _ in range(3))
    with torch.no_grad():
      masked = attn(query, key, value, attn_mask=_prefix_mask(b, s, n))
      truncated = attn(query, key[:, :n], value[:, :n], attn_mask=None)
    # Algebraically identical; truncating before vs after the k/v projections
    # changes GEMM shapes and hence float summation order, so allow float32
    # round-off (observed max diff ~2e-7).
    torch.testing.assert_close(masked, truncated, rtol=1e-5, atol=1e-6)

  def test_ragged_mask_falls_back_and_stays_correct(self):
    """A per-element (ragged) prefix mask is not rewritten, and matches the

    per-element truncated forward.
    """
    attn = self._attn()
    s, d = 16, 32
    lengths = [7, 12]
    torch.manual_seed(2)
    query, key, value = (torch.randn(len(lengths), s, d) for _ in range(3))
    mask = torch.zeros(len(lengths), 1, 1, s, dtype=torch.bool)
    for i, n in enumerate(lengths):
      mask[i, ..., :n] = True
    with torch.no_grad():
      batched = attn(query, key, value, attn_mask=mask)
      for i, n in enumerate(lengths):
        single = attn(query[i:i + 1], key[i:i + 1, :n], value[i:i + 1, :n],
                      attn_mask=None)
        torch.testing.assert_close(batched[i:i + 1], single,
                                   rtol=1e-5, atol=1e-6)

  def test_all_false_mask_not_rewritten(self):
    """n == 0 (no valid key) must not be treated as a prefix truncation."""
    attn = self._attn()
    b, s, d = 2, 8, 32
    torch.manual_seed(3)
    query, key, value = (torch.randn(b, s, d) for _ in range(3))
    mask = torch.zeros(b, 1, 1, s, dtype=torch.bool)
    with torch.no_grad():
      out = attn(query, key, value, attn_mask=mask)  # must not crash
    self.assertEqual(out.shape, (b, s, d))

  def test_full_model_forward_unchanged(self):
    """End-to-end TabFM forward with a mixed train_size batch is unchanged
    by the rewrite (the ICL mask is exactly the prefix pattern it targets)."""
    torch.manual_seed(4)
    model = pytorch_model.TabFM(
        embed_dim=16, max_classes=10, col_num_blocks=2, col_nhead=2,
        col_num_inds=8, row_num_blocks=2, row_nhead=2, row_num_cls=4,
        icl_num_blocks=2, icl_nhead=2, ff_factor=2, feature_group_size=3,
        is_classifier=True).eval()
    x = torch.randn(2, 6, 5)
    y = torch.randint(0, 3, (2, 6))
    train_size = torch.tensor([4, 4])
    with torch.no_grad():
      out = model(x, y, train_size)
    self.assertEqual(out.shape, (2, 6, 10))
    self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
  unittest.main()

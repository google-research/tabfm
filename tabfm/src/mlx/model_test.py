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
import numpy as np
import mlx.core as mx
import torch

from tabfm.src.mlx import model as MLXTabFM
from tabfm.src.pytorch import model as PyTorchTabFM


def _randomized_torch_state_dict(torch_model, seed=0):
  """Returns the torch state dict with zero-initialized tensors randomized.

  cls_tokens, ind_vectors, per_dim_scale and the Fourier-frequency buffers are
  zero at init; randomizing them makes the parity test exercise those paths
  (RoPE freqs and Linear weights are already non-trivially initialized).
  """
  rng = np.random.default_rng(seed)
  state_dict = {
      **dict(torch_model.named_parameters()),
      **dict(torch_model.named_buffers()),
  }
  out = {}
  for key, value in state_dict.items():
    tensor = value.detach().clone()
    if torch.count_nonzero(tensor) == 0:
      tensor = torch.tensor(
          rng.normal(scale=0.5, size=tuple(tensor.shape)).astype(np.float32)
      )
    out[key] = tensor
  return out


class MLXModelTest(unittest.TestCase):

  def test_mlx_model_instantiation(self):
    """Verifies that the MLX model instantiates with config and runs a forward pass."""
    model = MLXTabFM.TabFM(
        embed_dim=16,
        max_classes=10,
        col_num_blocks=2,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=2,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        is_classifier=True,
    )
    self.assertIsNotNone(model)

    # Run dummy forward pass
    np.random.seed(0)
    x = mx.array(np.random.randn(2, 4, 6).astype(np.float32))  # [B, T, H]
    y = mx.array(np.random.randint(0, 3, size=(2, 4)))  # [B, T]
    train_size = mx.array(np.array([2, 3], dtype=np.int32))  # [B]
    out = model(x, y, train_size)
    self.assertEqual(out.shape, (2, 4, 10))

  def test_torch_mlx_parity(self):
    """Verifies PyTorch vs MLX model outputs are numerically equal up to 1e-4."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        # 1. Config definition (shared by both backends)
        cfg = dict(
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
            is_classifier=is_classifier,
        )

        # 2. Instantiate PyTorch model (random init) and randomize its
        # zero-initialized tensors so all paths are exercised.
        torch_model = PyTorchTabFM.TabFM(**cfg)
        state_dict = _randomized_torch_state_dict(torch_model, seed=7)
        torch_model.load_state_dict(state_dict, strict=True)
        torch_model.eval()

        # 3. Load the same weights into the MLX model. Names and layouts
        # mirror the PyTorch port, so the state dict maps one-to-one.
        mlx_model = MLXTabFM.TabFM(**cfg)
        mlx_model.load_weights(
            [(k, mx.array(v.numpy())) for k, v in state_dict.items()],
            strict=True,
        )
        mlx_model.eval()

        # 4. Prepare random input data
        b, t, h = 3, 5, 8
        np.random.seed(123)
        x_np = np.random.normal(size=(b, t, h)).astype(np.float32)

        if is_classifier:
          y_np = np.random.randint(
              0, cfg["max_classes"], size=(b, t)
          ).astype(np.float32)
        else:
          y_np = np.random.normal(size=(b, t)).astype(np.float32)

        train_size_np = np.array([2, 3, 4], dtype=np.int32)
        d_np = np.array([5, 6, 7], dtype=np.int32)  # active feature counts
        cat_mask_np = np.zeros((b, h), dtype=bool)
        cat_mask_np[0, :3] = True
        cat_mask_np[1, :4] = True

        # 5. Forward passes
        with torch.no_grad():
          torch_out = torch_model(
              torch.from_numpy(x_np),
              torch.from_numpy(y_np),
              torch.from_numpy(train_size_np),
              cat_mask=torch.from_numpy(cat_mask_np),
              d=torch.from_numpy(d_np),
          ).numpy()

        mlx_out = np.array(
            mlx_model(
                mx.array(x_np),
                mx.array(y_np),
                mx.array(train_size_np),
                cat_mask=mx.array(cat_mask_np),
                d=mx.array(d_np),
            )
        )

        # 6. Compare PyTorch vs MLX outputs
        diff = np.abs(torch_out - mlx_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        self.assertLess(
            max_diff,
            1e-4,
            f"Fidelity discrepancy found: max diff = {max_diff}, mean diff ="
            f" {mean_diff} for is_classifier={is_classifier}",
        )


if __name__ == "__main__":
  unittest.main()

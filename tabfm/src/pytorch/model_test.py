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
import torch
from flax import nnx
import jax.numpy as jnp

from tabfm.src.jax.model import TabFM as JaxTabFM, YEmbeddingScheme
from tabfm.src.pytorch import model as PyTorchTabFM
from tabfm.src.hugging_face.torch_convert import convert, jax_params


class PyTorchModelTest(unittest.TestCase):

  def test_pytorch_model_instantiation(self):
    """Verifies that the PyTorch model instantiates with config and runs a forward pass."""
    model = PyTorchTabFM.TabFM(
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
        is_classifier=True
    )
    self.assertIsNotNone(model)
    
    # Run dummy forward pass
    x = torch.randn(2, 4, 6) # [B, T, H]
    y = torch.randint(0, 3, (2, 4)) # [B, T]
    train_size = torch.tensor([2, 3]) # [B]
    out = model(x, y, train_size)
    self.assertEqual(out.shape, (2, 4, 10))

  def test_jax_pytorch_parity(self):
    """Verifies JAX vs PyTorch model outputs are numerically equal up to 1e-4."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        # 1. Config definitions
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
            use_bias=False
        )

        # 2. Instantiate JAX model deterministically (random init)
        jax_model = JaxTabFM(
            loss="cross_entropy" if is_classifier else "rmse",
            activation="swiglu",
            feature_group=True,
            **cfg,
            y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
            rngs=nnx.Rngs(42),
            dtype=jnp.float32
        )

        # 3. Instantiate PyTorch model with matching config
        torch_model = PyTorchTabFM.TabFM(
            embed_dim=cfg["embed_dim"],
            max_classes=cfg["max_classes"],
            col_num_blocks=cfg["col_num_blocks"],
            col_nhead=cfg["col_nhead"],
            col_num_inds=cfg["col_num_inds"],
            row_num_blocks=cfg["row_num_blocks"],
            row_nhead=cfg["row_nhead"],
            row_num_cls=cfg["row_num_cls"],
            icl_num_blocks=cfg["icl_num_blocks"],
            icl_nhead=cfg["icl_nhead"],
            ff_factor=cfg["ff_factor"],
            feature_group_size=cfg["feature_group_size"],
            is_classifier=is_classifier
        )

        state_dict, missing = convert(jax_params(jax_model), torch_model)
        self.assertEqual(len(missing), 0)
        torch_model.load_state_dict(state_dict, strict=True)
        torch_model.eval()

        # 5. Prepare random input data
        b, t, h = 3, 5, 8
        np.random.seed(123)
        x_np = np.random.normal(size=(b, t, h)).astype(np.float32)
        
        if is_classifier:
          y_np = np.random.randint(0, cfg["max_classes"], size=(b, t)).astype(np.float32)
        else:
          y_np = np.random.normal(size=(b, t)).astype(np.float32)
          
        train_size_np = np.array([2, 3, 4], dtype=np.int32)
        d_np = np.array([5, 6, 7], dtype=np.int32) # active feature counts (d < h)
        cat_mask_np = np.zeros((b, h), dtype=bool)
        cat_mask_np[0, :3] = True
        cat_mask_np[1, :4] = True

        # JAX Inputs
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        train_size_jax = jnp.array(train_size_np)
        d_jax = jnp.array(d_np)
        cat_mask_jax = jnp.array(cat_mask_np)

        # PyTorch Inputs
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        train_size_torch = torch.from_numpy(train_size_np)
        d_torch = torch.from_numpy(d_np)
        cat_mask_torch = torch.from_numpy(cat_mask_np)

        # 6. Forward passes
        # JAX
        jax_out = jax_model(x_jax, y_jax, train_size_jax, cat_mask=cat_mask_jax, d=d_jax)
        jax_out_np = np.asarray(jax_out)

        # PyTorch
        with torch.no_grad():
          torch_out = torch_model(x_torch, y_torch, train_size_torch, cat_mask=cat_mask_torch, d=d_torch)
          torch_out_np = torch_out.numpy()

        # 7. Compare JAX vs PyTorch outputs
        diff = np.abs(jax_out_np - torch_out_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Assert max difference is less than 1e-4
        self.assertLess(
            max_diff,
            1e-4,
            f"Fidelity discrepancy found: max diff = {max_diff}, mean diff = {mean_diff} for is_classifier={is_classifier}"
        )


if __name__ == "__main__":
  unittest.main()

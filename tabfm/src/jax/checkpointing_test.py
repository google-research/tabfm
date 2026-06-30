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

import chex
from flax import nnx
import jax.numpy as jnp
import optax
from tabfm.src.jax import checkpointing
from tabfm.src.jax import model as tabfm_model

from absl.testing import absltest


class DummyDatasetIter:

  def __init__(self, data=0):
    self.data = data

  def __next__(self):
    val = self.data
    self.data += 1
    return {"data": jnp.array(val)}

  def get_state(self):
    return {"data": self.data}

  def set_state(self, state):
    self.data = state["data"]


class CheckpointingTest(unittest.IsolatedAsyncioTestCase, absltest.TestCase):

  def _create_dummy_state(self, rngs):
    model = tabfm_model.TabFM(
        loss="cross_entropy",
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        rngs=rngs,
    )
    tx = optax.adam(1e-3)
    return nnx.ModelAndOptimizer(model, tx)

  async def test_checkpointing(self):
    rngs = nnx.Rngs(0)
    state = self._create_dummy_state(rngs)

    # Create dummy dataset iterator with get_state / set_state
    dataset_iter = DummyDatasetIter()
    self.assertEqual(next(dataset_iter)["data"], 0)  # consume 0

    # Save checkpoint
    checkpoint_dir = self.create_tempdir().full_path
    checkpoint_manager = checkpointing.create_checkpoint_manager(
        checkpoint_dir, max_checkpoints=1
    )

    checkpointing.save_checkpoint_state(checkpoint_manager, 1, state, dataset_iter)
    self.assertTrue((checkpoint_manager.directory / '1').exists())

    # Restore checkpoint
    rngs_new = nnx.Rngs(1) # use different rngs to ensure state is overwritten
    state_to_restore = self._create_dummy_state(rngs_new)
    dataset_iter_to_restore = DummyDatasetIter()  # new iterator starting from 0

    restored_params, restored_iter = checkpointing.restore_checkpoint_state(
        checkpoint_manager,
        1,
        nnx.eval_shape(lambda: nnx.state(state_to_restore)),
        dataset_iter_to_restore,
    )
    nnx.update(state_to_restore, restored_params)

    # Check that restored state params are same as original
    chex.assert_trees_all_equal(nnx.state(state), nnx.state(state_to_restore))

    # Check that iterator is restored to correct position
    self.assertEqual(next(restored_iter)["data"], 1)


if __name__ == "__main__":
  absltest.main()

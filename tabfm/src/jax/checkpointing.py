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

import logging
import os
from typing import Any
from flax import nnx
import orbax.checkpoint as ocp


def create_checkpoint_manager(
    checkpoint_dir: str, max_checkpoints: int | None = None, read_only: bool = False
) -> ocp.CheckpointManager:
  if read_only:
    options = ocp.CheckpointManagerOptions(read_only=True)
    return ocp.CheckpointManager(
        os.path.join(checkpoint_dir, "orbax"), options=options
    )
  else:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints,
        create=True,
        async_options=ocp.options.AsyncOptions(timeout_secs=1800),
    )
    return ocp.CheckpointManager(
        os.path.join(checkpoint_dir, "orbax"),
        options=options,
    )


def save_checkpoint_state(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
    state: nnx.ModelAndOptimizer,
    dataset_iter: Any | None,
):
  """Saves the training state using Orbax CheckpointManager."""
  logging.info("Saving checkpoint at step %d...", step)
  # The manager saves the step and handles the directory structure.
  _, params = nnx.split(state)
  save_args = {"params": ocp.args.StandardSave(params)}
  if dataset_iter is not None:
    if hasattr(dataset_iter, "get_state"):
      save_args["dataset_iter_state"] = ocp.args.JsonSave(
          dataset_iter.get_state()
      )
    else:
      logging.warning(
          "dataset_iter provided but has no get_state method. Iterator state"
          " will not be saved."
      )
  checkpoint_manager.save(step, args=ocp.args.Composite(**save_args))
  # It's good practice to wait for the save to complete.
  checkpoint_manager.wait_until_finished()
  logging.info("Checkpoint saved.")


def restore_checkpoint_state(
    checkpoint_manager: ocp.CheckpointManager,
    restore_step: int,
    abstract_params: Any,
    dataset_iter: Any | None = None,
):
  """Restores the training state from an Orbax checkpoint."""
  logging.info("Restoring checkpoint from step %d...", restore_step)

  restore_args = {
      "params": ocp.args.StandardRestore(abstract_params, strict=False),  # pytype: disable=wrong-keyword-args
  }
  if dataset_iter is not None:
    if hasattr(dataset_iter, "set_state"):
      restore_args["dataset_iter_state"] = ocp.args.JsonRestore()
    else:
      logging.warning(
          "dataset_iter provided but has no set_state method. Iterator state"
          " cannot be restored natively."
      )

  restored = checkpoint_manager.restore(
      restore_step, args=ocp.args.Composite(**restore_args)
  )

  if dataset_iter is not None:
    if hasattr(dataset_iter, "set_state") and "dataset_iter_state" in restored:
      dataset_iter.set_state(restored["dataset_iter_state"])
      restored_dataset_iter = dataset_iter
      logging.info("Restored state and iterator from step %d.", restore_step)
    else:
      restored_dataset_iter = dataset_iter
      logging.info("Restored state from step %d.", restore_step)
    return restored["params"], restored_dataset_iter
  else:
    logging.info("Restored state from step %d.", restore_step)
    return restored["params"], None


def load_model(checkpoint_dir: str, step: int | None = None) -> Any:
  """Loads a TabFM model from an Orbax checkpoint directory.

  This function expects:
  1. A 'config.json' file in `checkpoint_dir` containing the model architecture
     parameters (e.g., embed_dim, col_num_blocks, etc.).
  2. An Orbax checkpoint under `checkpoint_dir/orbax`.

  Args:
    checkpoint_dir: Path to the directory containing config.json and the orbax/
      folder.
    step: The checkpoint step to restore. If None, auto-detects the latest step.

  Returns:
    An instantiated TabFM model with restored parameters.
  """
  import json  # pylint: disable=g-import-not-at-top
  from tabfm.src.model import TabFM  # pylint: disable=g-import-not-at-top

  # 1. Load architecture config
  config_path = os.path.join(checkpoint_dir, "config.json")
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Model configuration not found at {config_path}")

  with open(config_path, "r") as f:
    model_config = json.load(f)

  # 2. Instantiate TabFM with dummy RNGs (overwritten by checkpoint)
  rngs = nnx.Rngs(0)
  model = TabFM(rngs=rngs, **model_config)

  # 3. Create CheckpointManager
  checkpoint_manager = create_checkpoint_manager(checkpoint_dir, read_only=True)

  # 4. Determine step to restore
  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      raise ValueError(f"No checkpoints found in {checkpoint_dir}/orbax")

  # 5. Split state to get abstract params for restoration
  state = nnx.state(model)

  # 6. Restore parameters
  restored = checkpoint_manager.restore(
      step,
      args=ocp.args.Composite(
          params=ocp.args.StandardRestore(state, strict=False)
      ),
  )

  # 7. Update model with restored state
  nnx.update(model, restored["params"])

  return model

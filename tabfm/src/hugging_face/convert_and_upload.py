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

"""Utility to convert JAX TabFM checkpoints to PyTorch and upload to Hugging Face."""

import os
import logging
from typing import Any, Optional, Tuple, Literal
from absl import app
from absl import flags
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Float, Int

from tabfm.src.jax import tabfm_v1_0_0
from tabfm.src.pytorch import model as MT
from tabfm.src.pytorch.tabfm_v1_0_0 import TabFM_HF
from tabfm.src.hugging_face.torch_convert import jax_params, convert

# Architecture config of the v1.0.0 checkpoint.
V1_0_0_CONFIG = dict(
    embed_dim=256,
    max_classes=10,
    col_num_blocks=3,
    col_nhead=4,
    col_num_inds=256,
    row_num_blocks=3,
    row_nhead=8,
    row_num_cls=8,
    icl_num_blocks=24,
    icl_nhead=8,
    ff_factor=4,
    feature_group_size=3,
)

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "model_type",
    "all",
    ["classification", "regression", "all"],
    "Type of model to convert.",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Local directory to save the converted PyTorch checkpoint.",
    required=True,
)
flags.DEFINE_string(
    "repo_id",
    None,
    "Hugging Face repository ID (e.g. 'google/tabfm-1.0.0-pytorch'). "
    "If not provided, the conversion is run locally and not uploaded.",
)
flags.DEFINE_string(
    "token",
    None,
    "Hugging Face write token. Required if repo_id is provided.",
)
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Local path to JAX checkpoint directory. If None, JAX weights will be "
    "downloaded from Hugging Face.",
)


def convert_model(
    model_type: Literal["classification", "regression"],
    checkpoint_path: Optional[str] = None,
) -> Tuple[TabFM_HF, float]:
  """Converts JAX checkpoint of model_type to PyTorch TabFM and runs parity verification."""
  logging.info("Loading JAX %s model...", model_type)
  is_classifier = (model_type == "classification")
  
  # Load JAX model
  jm = tabfm_v1_0_0.load(
      model_type=model_type,
      checkpoint_path=checkpoint_path,
      col_attention_impl="jax", row_attention_impl="jax", icl_attention_impl="jax", dtype=jnp.float32,
  )
  jp = jax_params(jm)
  
  # Retrieve the decoder hidden dimension from the loaded JAX model weights
  decoder_hidden = jp["icl_predictor.decoder.layers.0.kernel"].shape[1]
  
  logging.info("Instantiating PyTorch model...")
  torch_model = TabFM_HF(
      decoder_hidden=decoder_hidden,
      is_classifier=is_classifier,
      **V1_0_0_CONFIG,
  )
  
  logging.info("Converting parameters...")
  state_dict, missing = convert(jp, torch_model)
  if missing:
    raise ValueError(f"Missing parameters during conversion: {missing}")
    
  torch_model.load_state_dict(state_dict, strict=True)
  torch_model.eval()
  
  # Run a parity check on dummy inputs to ensure the conversion is correct.
  logging.info("Verifying parity on dummy inputs...")
  max_diff = verify_parity(jm, torch_model, is_classifier)
  logging.info("Parity check max difference: %e", max_diff)
  if max_diff >= 1e-4:
    raise ValueError(f"Parity check failed with max difference: {max_diff}")
    
  return torch_model, max_diff


def verify_parity(
    jax_model: Any,
    torch_model: MT.TabFM,
    is_classifier: bool,
) -> float:
  """Runs JAX and PyTorch models on identical random inputs and returns max absolute difference."""
  B, T, H = 1, 12, 5
  rng = np.random.default_rng(0)
  x: Float[np.ndarray, "B T H"] = rng.random((B, T, H)).astype(np.float32)
  
  if is_classifier:
    y: Int[np.ndarray, "B T"] = rng.integers(0, 3, (B, T)).astype(np.int32)
  else:
    y: Float[np.ndarray, "B T"] = rng.standard_normal((B, T)).astype(np.float32)
    
  ts: Int[np.ndarray, "B"] = np.array([7], dtype=np.int32)
  
  # Run JAX
  jout = np.asarray(jax_model(jnp.asarray(x), jnp.asarray(y), train_size=jnp.asarray(ts)))
  
  # Run PyTorch
  with torch.no_grad():
    tout = torch_model(torch.tensor(x), torch.tensor(y), torch.tensor(ts)).numpy()
    
  max_diff = float(np.max(np.abs(jout - tout)))
  return max_diff


def save_checkpoint(
    model: TabFM_HF,
    output_dir: str,
    model_type: str,
) -> str:
  """Saves model weights and config to output_dir using save_pretrained."""
  model_dir = os.path.join(output_dir, model_type)
  os.makedirs(model_dir, exist_ok=True)
  logging.info("Saving model to %s...", model_dir)
  model.save_pretrained(model_dir)
  logging.info("Saved model to %s", model_dir)
  return model_dir


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
    
  # Set JAX to run on CPU to avoid device initialization errors on machines
  # without proper JAX CUDA driver setups.
  os.environ["JAX_PLATFORMS"] = "cpu"
  
  model_types = (
      ["classification", "regression"]
      if FLAGS.model_type == "all"
      else [FLAGS.model_type]
  )
  
  local_dirs = {}
  for mtype in model_types:
    logging.info("=== Starting conversion for %s ===", mtype)
    torch_model, max_diff = convert_model(mtype, FLAGS.checkpoint_path)
    logging.info("Conversion successful! Parity max diff = %e", max_diff)
    
    saved_dir = save_checkpoint(torch_model, FLAGS.output_dir, mtype)
    local_dirs[mtype] = saved_dir
    
  if FLAGS.repo_id:
    if not FLAGS.token:
      raise ValueError("Hugging Face token is required when repo_id is provided.")

    from huggingface_hub import HfApi  # pylint: disable=g-import-not-at-top
    api = HfApi(token=FLAGS.token)

    for mtype, sdir in local_dirs.items():
      logging.info("Uploading %s folder to %s...", mtype, FLAGS.repo_id)
      api.upload_folder(
          folder_path=sdir,
          repo_id=FLAGS.repo_id,
          path_in_repo=mtype,
          repo_type="model",
      )
    logging.info("Successfully uploaded PyTorch checkpoints to Hugging Face!")


if __name__ == "__main__":
  app.run(main)

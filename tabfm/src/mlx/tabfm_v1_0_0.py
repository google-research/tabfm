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

"""Loads pre-trained TabFM v1.0.0 weights into the MLX backend.

The MLX port shares its module/parameter naming with the PyTorch port, so it
loads the PyTorch safetensors checkpoint (google/tabfm-1.0.0-pytorch) directly
via ``mx.load`` -- no separate MLX weight release or key remapping is needed.
"""

import json
import os
import threading
from typing import Any, Dict, Optional

from absl import logging
import mlx.core as mx
from huggingface_hub import snapshot_download

from tabfm.src.mlx.model import TabFM

# The MLX backend reuses the PyTorch weight release: parameter names and
# layouts (Linear as [out, in]) are identical, so the safetensors file loads
# one-to-one.
HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "model.safetensors"

_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, TabFM] = {}


def _model_kwargs_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
  """Translates a checkpoint config.json into TabFM constructor kwargs."""
  kwargs = dict(config)
  if "task" in kwargs:
    kwargs["is_classifier"] = kwargs.pop("task") == "classification"
  for key in ("model_type", "version", "framework"):
    kwargs.pop(key, None)
  return kwargs


def _load_from_dir(local_dir: str, model_type: str) -> TabFM:
  """Instantiates a TabFM from a directory with config.json + safetensors."""
  weights_path = os.path.join(local_dir, _WEIGHTS_NAME)
  if not os.path.exists(weights_path):
    raise FileNotFoundError(
        f"No {_WEIGHTS_NAME} found in {local_dir}. The MLX backend loads "
        "safetensors checkpoints (as released in "
        f"https://huggingface.co/{HF_REPO_ID})."
    )

  cfg_path = os.path.join(local_dir, _CONFIG_NAME)
  if os.path.exists(cfg_path):
    with open(cfg_path) as f:
      model_kwargs = _model_kwargs_from_config(json.load(f))
  else:
    # no config.json: pass is_classifier explicitly
    logging.warning("No config.json found in %s", local_dir)
    model_kwargs = {"is_classifier": model_type == "classification"}

  model = TabFM(**model_kwargs)
  # strict=True: the checkpoint keys (parameters + buffers such as
  # `rope.freqs` and `fourier_frequencies`) must map one-to-one onto the
  # MLX module tree.
  model.load_weights(list(mx.load(weights_path).items()), strict=True)
  return model


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    *,
    dtype: Any = mx.bfloat16,
    use_cache: bool = True,
) -> TabFM:
  """Loads the MLX TabFM v1.0.0 model with pre-trained weights.

  The checkpoint is stored in float32, but the model is designed to run in
  bfloat16 (matching the JAX release's ``dtype=jnp.bfloat16`` compute default),
  with a few internal fp32 upcasts. ``dtype`` casts the model accordingly; pass
  ``None`` to keep the float32 weights.

  ``dtype`` is provided for float32 debugging / quality comparison; the model
  is designed for bfloat16 and this option may be removed in a future release.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory with the checkpoint (either the directory
      containing the ``model_type`` subfolder, or that subfolder itself). If
      None, downloads from Hugging Face (google/tabfm-1.0.0-pytorch).
    dtype: Compute dtype to cast the model to after loading. Defaults to
      bfloat16; pass None to keep the float32 weights. MLX arrays live in
      unified memory, so there is no device argument.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    An eval-mode MLX TabFM model with pre-trained weights loaded.
  """
  if model_type not in ("classification", "regression"):
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        "Must be 'classification' or 'regression'."
    )

  cache_key = (model_type, checkpoint_path, str(dtype))
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()
  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    if checkpoint_path is None:
      logging.info(
          "Downloading TabFM v1.0.0 %s weights from Hugging Face...",
          model_type,
      )
      base_path = snapshot_download(
          repo_id=HF_REPO_ID,
          allow_patterns=[f"{model_type}/**"],
      )
      local_dir = os.path.join(base_path, model_type)
    else:
      local_dir = checkpoint_path
      if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local checkpoint path not found: {local_dir}")
      sub = os.path.join(local_dir, model_type)
      if os.path.isdir(sub):
        local_dir = sub

    model = _load_from_dir(local_dir, model_type)

    if dtype is not None:
      model.set_dtype(dtype)  # engage the bf16 compute design (see docstring)
    # Materialize the (lazy) checkpoint read + dtype cast now, so load errors
    # surface here rather than inside the first forward pass, and the cached
    # model is safe to share across threads.
    mx.eval(model.parameters())
    model.eval()

    if use_cache:
      _LOAD_CACHE[cache_key] = model
    return model
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

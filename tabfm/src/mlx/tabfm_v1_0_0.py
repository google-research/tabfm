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

"""Loads the MLX TabFM v1.0.0 model with pre-trained weights.

Mirrors tabfm.src.pytorch.tabfm_v1_0_0.load(): the checkpoint is stored in
float32, but the model is designed to run in bfloat16, with a few internal
fp32 upcasts. Converted weights are cached as .npz under ~/.cache/tabfm_mlx
so the 6 GB safetensors are only translated once.

NOTE: the pretrained weights are distributed under the separate
tabfm-non-commercial-v1.0 license (non-commercial, non-production use only).
"""

import json
import os
import threading
from typing import Any, Dict, Optional

from absl import logging
from huggingface_hub import snapshot_download

import mlx.core as mx

from tabfm.src.mlx.convert import convert_safetensors
from tabfm.src.mlx.model import TabFM

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, TabFM] = {}


def _convert_cache_path(
    repo_id: str, model_type: str, dtype: Optional[mx.Dtype]
) -> str:
  """Cache file for one (checkpoint, model_type, storage dtype) triple.

  dtype is part of the name: the converted npz stores the cast weights, so
  loading the same checkpoint at another dtype is a different artifact.
  Keying on the triple keeps both (e.g. a bfloat16 run and a float32 parity
  run) instead of re-converting ~6 GB on every switch.
  """
  cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tabfm_mlx")
  os.makedirs(cache_dir, exist_ok=True)
  safe = repo_id.replace("/", "__")
  dtype_tag = "float32" if dtype is None else str(dtype).rsplit(".", 1)[-1]
  return os.path.join(cache_dir, f"{safe}__{model_type}__{dtype_tag}.npz")


def _load_weights_cached(
    safetensors_path: str, npz_path: str, dtype: Optional[mx.Dtype]
) -> Dict[str, mx.array]:
  """Loads weights, converting safetensors->npz once (staleness-checked)."""
  meta_path = npz_path + ".meta.json"
  st = os.stat(safetensors_path)
  meta = {
      "path": safetensors_path,
      "size": st.st_size,
      "mtime": st.st_mtime,
      "dtype": None if dtype is None else str(dtype),
  }
  use_cached = False
  if os.path.exists(npz_path) and os.path.exists(meta_path):
    try:
      with open(meta_path) as f:
        use_cached = json.load(f) == meta
    except (OSError, ValueError):
      use_cached = False
  if not use_cached:
    logging.info("Converting %s to MLX weights (one-time)...", safetensors_path)
    convert_safetensors(safetensors_path, npz_path, dtype=dtype)
    with open(meta_path, "w") as f:
      json.dump(meta, f)
  weights = mx.load(npz_path)
  if isinstance(weights, (list, tuple)):  # savez dict round-trip guard
    weights = dict(weights)
  if dtype is not None:
    weights = {
        k: v.astype(dtype) if v.dtype != dtype else v
        for k, v in weights.items()
    }
  return weights


def _apply_config(cfg: Dict[str, Any], model_kwargs: Dict[str, Any]) -> None:
  # Translates the HF config.json keys into TabFM constructor kwargs, mirroring
  # the PyTorch loader (drops hub-only keys, derives is_classifier from task).
  if "is_classifier" not in model_kwargs and "task" in cfg:
    model_kwargs["is_classifier"] = cfg.pop("task") == "classification"
  for key in ("model_type", "version", "framework", "task"):
    cfg.pop(key, None)
  for k, v in cfg.items():
    if k not in model_kwargs:
      model_kwargs[k] = v


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    *,
    dtype: Any = mx.bfloat16,
    use_cache: bool = True,
) -> TabFM:
  """Loads the MLX TabFM v1.0.0 model with pre-trained weights.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory or weights file. If None, downloads from
      Hugging Face (google/tabfm-1.0.0-pytorch).
    dtype: Compute dtype to cast the model to after loading. Defaults to
      bfloat16; pass None to keep the float32 weights (~1.2x slower, and the
      reference the bfloat16 path approximates). float16 is rejected: its
      65504 range overflows to NaN here.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    A TabFM MLX model with pre-trained weights loaded.
  """
  if model_type not in ("classification", "regression"):
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        "Must be 'classification' or 'regression'."
    )

  if dtype is mx.float16:
    raise ValueError(
        "float16 overflows this model: activations exceed its 65504 range and "
        "the forward pass returns NaN. Use bfloat16 (the design dtype, same "
        "speed) or None for float32."
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
      base = snapshot_download(
          repo_id=HF_REPO_ID,
          allow_patterns=[f"{model_type}/**", "config.json"],
      )
      local_dir = os.path.join(base, model_type)
      safetensors_path = os.path.join(local_dir, "model.safetensors")
    else:
      if os.path.isfile(checkpoint_path):
        # A direct file path to model.safetensors.
        safetensors_path = checkpoint_path
        local_dir = os.path.dirname(checkpoint_path)
      elif os.path.isdir(checkpoint_path):
        local_dir = checkpoint_path
        sub = os.path.join(local_dir, model_type)
        if os.path.isdir(sub):
          local_dir = sub
        safetensors_path = os.path.join(local_dir, "model.safetensors")
      else:
        raise FileNotFoundError(
            f"Local checkpoint path not found: {checkpoint_path}"
        )

    cfg_path = os.path.join(local_dir, "config.json")
    model_kwargs: Dict[str, Any] = {}
    if os.path.exists(cfg_path):
      with open(cfg_path) as f:
        _apply_config(json.load(f), model_kwargs)
    else:
      logging.warning("No config.json found in %s", local_dir)
    if "is_classifier" not in model_kwargs:
      model_kwargs["is_classifier"] = model_type == "classification"

    model = TabFM(**model_kwargs)

    npz_path = _convert_cache_path(
        HF_REPO_ID if checkpoint_path is None else local_dir, model_type, dtype
    )
    weights = _load_weights_cached(safetensors_path, npz_path, dtype)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    if use_cache:
      _LOAD_CACHE[cache_key] = model
    return model
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

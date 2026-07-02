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

import json
import os
import threading
from typing import Any, Dict, Optional
from absl import logging
from huggingface_hub import PyTorchModelHubMixin, constants, snapshot_download

from tabfm.src.pytorch.model import TabFM

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, "TabFM_HF"] = {}


class TabFM_HF(
    TabFM,
    PyTorchModelHubMixin,
    repo_url="https://github.com/google-research/tabfm",
    license="other",
):
  """PyTorch TabFM model with HuggingFace Hub support.

  Subclasses TabFM directly (rather than wrapping it) and mixes in
  PyTorchModelHubMixin, keeping the Hugging Face specific loading logic out
  of the plain model class.
  """

  @classmethod
  def _from_pretrained(
      cls,
      *,
      model_id,
      revision,
      cache_dir,
      force_download,
      local_files_only,
      token,
      map_location="cpu",
      strict=True,
      **model_kwargs,
  ):
    subfolder = model_kwargs.pop("subfolder", None)

    def _apply_config(cfg):
      if "is_classifier" not in model_kwargs and "task" in cfg:
        model_kwargs["is_classifier"] = cfg.pop("task") == "classification"
      for key in ("model_type", "version", "framework"):
        cfg.pop(key, None)
      for k, v in cfg.items():
        if k not in model_kwargs:
          model_kwargs[k] = v

    # translate config keys already merged into model_kwargs by from_pretrained()
    _apply_config(model_kwargs)

    if subfolder is None:
      local_id = model_id
    elif os.path.isdir(model_id):
      local_id = os.path.join(model_id, subfolder)
    else:
      base_path = snapshot_download(
          repo_id=model_id,
          revision=revision,
          cache_dir=cache_dir,
          force_download=force_download,
          local_files_only=local_files_only,
          token=token,
          allow_patterns=[f"{subfolder}/**"],
      )
      local_id = os.path.join(base_path, subfolder)

    if subfolder is not None:
      cfg_path = os.path.join(local_id, constants.CONFIG_NAME)
      if os.path.exists(cfg_path):
        with open(cfg_path) as f:
          _apply_config(json.load(f))
      else:
        logging.warning("No config.json found in %s", local_id)

    return super()._from_pretrained(
        model_id=local_id,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        map_location=map_location,
        strict=strict,
        **model_kwargs,
    )


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> "TabFM_HF":
  """Loads the PyTorch TabFM v1.0.0 model with pre-trained weights.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory or weights file. If None, downloads from
      Hugging Face (google/tabfm-1.0.0-pytorch).
    device: Target device (e.g. 'cuda', 'cpu'). Defaults to 'cpu'.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    An eval-mode TabFM_HF model with pre-trained weights loaded.
  """
  if model_type not in ("classification", "regression"):
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        "Must be 'classification' or 'regression'."
    )

  cache_key = (model_type, checkpoint_path, device)
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()
  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    if checkpoint_path is None:
      logging.info(
          "Downloading TabFM v1.0.0 PyTorch %s weights from Hugging Face...",
          model_type,
      )
      model = TabFM_HF.from_pretrained(HF_REPO_ID, subfolder=model_type)
    else:
      local_dir = checkpoint_path
      if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local checkpoint path not found: {local_dir}")
      sub = os.path.join(local_dir, model_type)
      if os.path.isdir(sub):
        local_dir = sub

      if os.path.exists(os.path.join(local_dir, "config.json")):
        model = TabFM_HF.from_pretrained(local_dir)
      else:
        # no config.json: pass is_classifier explicitly
        model = TabFM_HF.from_pretrained(
            local_dir,
            is_classifier=(model_type == "classification"),
        )

    if device is not None:
      model = model.to(device)
    model.eval()

    if use_cache:
      _LOAD_CACHE[cache_key] = model
    return model
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

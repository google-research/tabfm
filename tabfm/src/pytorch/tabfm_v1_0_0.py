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

from dataclasses import dataclass
import os
import threading
from typing import Any, Dict, Optional
from absl import logging
import torch

from tabfm.src.pytorch.model import TabFM

HF_REPO_ID = "google/tabfm-1.0.0-pytorch"

@dataclass(frozen=True)
class Config:
  max_classes: int = 10
  embed_dim: int = 256
  col_num_blocks: int = 3
  col_nhead: int = 4
  col_num_inds: int = 256
  row_num_blocks: int = 3
  row_nhead: int = 8
  row_num_cls: int = 8
  icl_num_blocks: int = 24
  icl_nhead: int = 8
  ff_factor: int = 4
  feature_group_size: int = 3
  num_freq: int = 32
  is_classifier: bool = True

  def to_dict(self) -> Dict[str, Any]:
    return {
        "max_classes": self.max_classes,
        "embed_dim": self.embed_dim,
        "col_num_blocks": self.col_num_blocks,
        "col_nhead": self.col_nhead,
        "col_num_inds": self.col_num_inds,
        "row_num_blocks": self.row_num_blocks,
        "row_nhead": self.row_nhead,
        "row_num_cls": self.row_num_cls,
        "icl_num_blocks": self.icl_num_blocks,
        "icl_nhead": self.icl_nhead,
        "ff_factor": self.ff_factor,
        "feature_group_size": self.feature_group_size,
        "num_freq": self.num_freq,
        "is_classifier": self.is_classifier,
    }

@dataclass(frozen=True)
class ClassificationConfig(Config):
  is_classifier: bool = True

@dataclass(frozen=True)
class RegressionConfig(Config):
  is_classifier: bool = False

_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, TabFM] = {}

def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
    use_cache: bool = True,
) -> TabFM:
  """Loads the PyTorch TabFM v1.0.0 model with pre-trained weights."""
  cache_key = (model_type, checkpoint_path, device)
  
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()

  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    if model_type == "classification":
      config = ClassificationConfig()
    elif model_type == "regression":
      config = RegressionConfig()
    else:
      raise ValueError(f"Unsupported model_type: {model_type}.")

    model = TabFM(**config.to_dict())

    if checkpoint_path is None:
      try:
        from huggingface_hub import snapshot_download
        logging.info("Downloading TabFM v1.0.0 PyTorch %s weights...", model_type)
        base_path = snapshot_download(repo_id=HF_REPO_ID)
        checkpoint_file = os.path.join(base_path, model_type, "pytorch_model.bin")
      except ImportError as e:
        raise ImportError("huggingface_hub is required to download weights.") from e
    else:
      checkpoint_file = checkpoint_path
      if os.path.isdir(checkpoint_file):
        for sub in [os.path.join(checkpoint_file, model_type, "pytorch_model.bin"),
                    os.path.join(checkpoint_file, "pytorch_model.bin")]:
          if os.path.exists(sub):
            checkpoint_file = sub
            break

    if not os.path.exists(checkpoint_file):
      raise FileNotFoundError(f"Weights not found at: {checkpoint_file}")

    logging.info("Loading PyTorch state dict from %s...", checkpoint_file)
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    if device is not None:
      model = model.to(device)

    model.eval()
    
    if use_cache:
      _LOAD_CACHE[cache_key] = model
    return model
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

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

"""TabFM v1.0.0 Model Release.

This module provides simple access to the TabFM v1.0.0 model architecture and
allows downloading/restoring pre-trained weights from Hugging Face or loading
them from a local path.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional
from absl import logging
from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from tabfm.src import checkpointing
from tabfm.src.model import TabFM, YEmbeddingScheme

# Hugging Face repository ID for TabFM v1.0.0
HF_REPO_ID = "google/tabfm-v1-0-0"


@dataclass(frozen=True)
class Config:
  """Hardcoded architecture configuration for TabFM v1.0.0."""

  loss: str = "cross_entropy"
  max_classes: int = 10
  embed_dim: int = 256
  col_num_blocks: int = 3
  col_nhead: int = 4
  col_num_inds: int = 256
  row_num_blocks: int = 3
  row_nhead: int = 8
  row_num_cls: int = 8
  row_rope_base: float = 100000.0
  icl_num_blocks: int = 24
  icl_nhead: int = 8
  ff_factor: int = 4
  activation: str = "swiglu"
  feature_group: bool = True
  feature_group_size: int = 3
  use_fourier_features: bool = True
  fourier_features_num_frequencies: int = 32
  fourier_features_sigma: float = 1.0
  cache_icl_input_only: bool = False
  y_embedding_scheme: YEmbeddingScheme = (
      YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING
  )
  use_bias: bool = True

  def to_dict(self) -> Dict[str, Any]:
    return {
        "loss": self.loss,
        "max_classes": self.max_classes,
        "embed_dim": self.embed_dim,
        "col_num_blocks": self.col_num_blocks,
        "col_nhead": self.col_nhead,
        "col_num_inds": self.col_num_inds,
        "row_num_blocks": self.row_num_blocks,
        "row_nhead": self.row_nhead,
        "row_num_cls": self.row_num_cls,
        "row_rope_base": self.row_rope_base,
        "icl_num_blocks": self.icl_num_blocks,
        "icl_nhead": self.icl_nhead,
        "ff_factor": self.ff_factor,
        "activation": self.activation,
        "feature_group": self.feature_group,
        "feature_group_size": self.feature_group_size,
        "use_fourier_features": self.use_fourier_features,
        "fourier_features_num_frequencies": (
            self.fourier_features_num_frequencies
        ),
        "fourier_features_sigma": self.fourier_features_sigma,
        "cache_icl_input_only": self.cache_icl_input_only,
        "y_embedding_scheme": self.y_embedding_scheme,
        "use_bias": self.use_bias,
    }


@dataclass(frozen=True)
class ClassificationConfig(Config):
  """Architecture configuration for TabFM v1.0.0 classification model."""

  loss: str = "cross_entropy"


@dataclass(frozen=True)
class RegressionConfig(Config):
  """Architecture configuration for TabFM v1.0.0 regression model."""

  loss: str = "rmse"


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    step: Optional[int] = None,
    *,
    dtype: Any = jnp.bfloat16,
) -> TabFM:
  """Loads the TabFM v1.0.0 model with pre-trained weights.

  If `checkpoint_path` is not provided, it will attempt to download the weights
  from Hugging Face (google/tabfm-v1-0-0). If provided, it will load from the
  specified local directory containing the Orbax checkpoint.

  Args:
    model_type: Type of model to load ('classification' or 'regression').
    checkpoint_path: Local directory containing the 'orbax/' checkpoint, or None
      to download from Hugging Face.
    step: The checkpoint step to restore (for local loading).
    dtype: Calculations dtype for JAX.

  Returns:
    An initialized TabFM model with restored weights.
  """
  # 1. Instantiate model with hardcoded config based on model_type
  if model_type == "classification":
    config = ClassificationConfig()
  elif model_type == "regression":
    config = RegressionConfig()
  else:
    raise ValueError(
        f"Unsupported model_type: {model_type}. Must be 'classification' or"
        " 'regression'."
    )

  rngs = nnx.Rngs(0)
  model = TabFM(rngs=rngs, dtype=dtype, **config.to_dict())

  # 2. Get checkpoint directory
  if checkpoint_path is None:
    # Download from Hugging Face
    try:
      from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

      logging.info(
          "Downloading TabFM v1.0.0 %s weights from Hugging Face...", model_type
      )
      base_path = snapshot_download(repo_id=HF_REPO_ID)
      checkpoint_path = os.path.join(base_path, model_type)
    except ImportError as e:
      raise ImportError(
          "huggingface_hub is required to download weights. "
          "Install it using 'pip install huggingface_hub' or provide a "
          "local checkpoint_path."
      ) from e
  else:
    # If local root checkpoint path is provided, try appending model_type
    if not os.path.exists(os.path.join(checkpoint_path, "orbax")):
      potential_path = os.path.join(checkpoint_path, model_type)
      if os.path.exists(os.path.join(potential_path, "orbax")):
        checkpoint_path = potential_path

  # 3. Restore parameters from local/downloaded path
  checkpoint_manager = checkpointing.create_checkpoint_manager(
      checkpoint_path, read_only=True
  )
  if step is None:
    step = checkpoint_manager.latest_step()
    if step is None:
      raise ValueError(f"No checkpoints found in {checkpoint_path}/orbax")

  state = nnx.state(model)
  restored = checkpoint_manager.restore(
      step,
      args=ocp.args.Composite(
          params=ocp.args.StandardRestore(state, strict=False)
      ),
  )
  nnx.update(model, restored["params"])

  return model

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
import threading
from typing import Any, Dict, Optional
from absl import logging
from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from huggingface_hub import ModelHubMixin, snapshot_download
from tabfm.src.jax import checkpointing
from tabfm.src.jax.model import TabFM, YEmbeddingScheme

# Hugging Face repository ID for TabFM v1.0.0
HF_REPO_ID = "google/tabfm-1.0.0-jax"


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
  use_bias: bool = False

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


def _restore_from_dir(
    model: TabFM,
    checkpoint_path: str,
    model_type: str,
    step: Optional[int],
) -> TabFM:
  """Restores Orbax checkpoint into model in place and returns model."""
  if not os.path.exists(os.path.join(checkpoint_path, "orbax")):
    potential_path = os.path.join(checkpoint_path, model_type)
    if os.path.exists(os.path.join(potential_path, "orbax")):
      checkpoint_path = potential_path

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


class TabFM_HF(TabFM, ModelHubMixin):
  """JAX TabFM model with HuggingFace Hub support.

  Subclasses TabFM directly rather than wrapping it, since NNX has special
  handling for nnx.Module subclasses that a wrapper breaks under JIT.
  Provides from_pretrained(), save_pretrained(), and push_to_hub() via
  ModelHubMixin.
  """

  def __init__(self, *args, model_type: str = "classification", **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, "_model_type", model_type)

  def __setattr__(self, name: str, value: Any):
    if name in ("_hub_mixin_config", "_model_type"):
      # Bypass NNX checking entirely and write directly to Python's object dict.
      object.__setattr__(self, name, value)
    else:
      super().__setattr__(name, value)

  @classmethod
  def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: Optional[str],
      cache_dir,
      force_download: bool,
      local_files_only: bool,
      token,
      model_type: str = "classification",
      step: Optional[int] = None,
      col_attention_impl: str = "flash",
      row_attention_impl: str = "jax",
      icl_attention_impl: str = "flash",
      dtype: Any = jnp.bfloat16,
      **kwargs,
  ) -> "TabFM_HF":
    from tabfm.src.jax.model import AttentionImplementation  # pylint: disable=g-import-not-at-top

    if model_type == "classification":
      config = ClassificationConfig()
    elif model_type == "regression":
      config = RegressionConfig()
    else:
      raise ValueError(
          f"Unsupported model_type: {model_type!r}. "
          "Must be 'classification' or 'regression'."
      )

    config_dict = config.to_dict()
    config_dict["col_attention_impl"] = AttentionImplementation(col_attention_impl)
    config_dict["row_attention_impl"] = AttentionImplementation(row_attention_impl)
    config_dict["icl_attention_impl"] = AttentionImplementation(icl_attention_impl)

    model = cls(rngs=nnx.Rngs(0), dtype=dtype, model_type=model_type, **config_dict)

    if os.path.isdir(model_id):
      checkpoint_path = model_id
    else:
      logging.info(
          "Downloading TabFM v1.0.0 JAX %s weights from Hugging Face...",
          model_type,
      )
      base_path = snapshot_download(
          repo_id=model_id,
          revision=revision,
          cache_dir=cache_dir,
          force_download=force_download,
          local_files_only=local_files_only,
          token=token,
          allow_patterns=[f"{model_type}/**"],
      )
      checkpoint_path = os.path.join(base_path, model_type)

    return _restore_from_dir(model, checkpoint_path, model_type, step)

  def _save_pretrained(self, save_directory):
    """Save Orbax checkpoint to save_directory/<model_type>/orbax/."""
    out_path = os.path.join(save_directory, self._model_type)
    os.makedirs(out_path, exist_ok=True)
    manager = checkpointing.create_checkpoint_manager(out_path, read_only=False)
    state = nnx.state(self)
    manager.save(0, args=ocp.args.Composite(params=ocp.args.StandardSave(state)))
    manager.wait_until_finished()


# Process-wide memo of restored models (shared by load() and TabFM_HF).
_LOAD_CACHE_LOCK = threading.Lock()
_LOAD_CACHE: Dict[Any, "TabFM_HF"] = {}


def load(
    model_type: str = "classification",
    checkpoint_path: Optional[str] = None,
    step: Optional[int] = None,
    *,
    col_attention_impl: str = "flash",
    row_attention_impl: str = "jax",
    icl_attention_impl: str = "flash",
    dtype: Any = jnp.bfloat16,
    use_cache: bool = True,
) -> "TabFM_HF":
  """Loads the TabFM v1.0.0 JAX model with pre-trained weights.

  If `checkpoint_path` is not provided, downloads weights from Hugging Face
  (google/tabfm-1.0.0-jax), fetching only the requested model_type subfolder.

  Args:
    model_type: 'classification' or 'regression'.
    checkpoint_path: Local directory containing the 'orbax/' checkpoint, or
      None to download from Hugging Face.
    step: Checkpoint step to restore (for local loading).
    col_attention_impl: Attention impl for column-attention layers ('jax' or
      'flash').
    row_attention_impl: Attention impl for row-attention layers.
    icl_attention_impl: Attention impl for ICL layers.
    dtype: JAX compute dtype.
    use_cache: Reuse a process-wide cached model for identical settings.

  Returns:
    The restored TabFM_HF model.
  """
  cache_key = (
      model_type,
      checkpoint_path,
      step,
      col_attention_impl,
      row_attention_impl,
      icl_attention_impl,
      str(dtype),
  )
  if use_cache:
    _LOAD_CACHE_LOCK.acquire()
  try:
    if use_cache and cache_key in _LOAD_CACHE:
      return _LOAD_CACHE[cache_key]

    result = TabFM_HF.from_pretrained(
        HF_REPO_ID if checkpoint_path is None else checkpoint_path,
        model_type=model_type,
        step=step,
        col_attention_impl=col_attention_impl,
        row_attention_impl=row_attention_impl,
        icl_attention_impl=icl_attention_impl,
        dtype=dtype,
    )

    if use_cache:
      _LOAD_CACHE[cache_key] = result
    return result
  finally:
    if use_cache:
      _LOAD_CACHE_LOCK.release()

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

"""LoRA fine-tuning for the MLX TabFM backend.

Adds trainable low-rank adapters (https://arxiv.org/abs/2106.09685) on top of
the frozen pre-trained model. The base weights are never updated: training
optimizes only the adapter matrices, and ``merge_lora`` can fold them back
into plain ``nn.Linear`` layers afterwards for zero-overhead inference.

Training mirrors how TabFM performs inference (in-context learning): each
step samples an episode from the fine-tuning dataset, splits it into context
rows (labels visible to the model) and target rows, and takes the loss on the
target rows only. Inputs are the *numeric* arrays the model consumes -- run
them through the same preprocessing used at inference time (e.g.
``TransformToNumerical`` / ``PreprocessingPipeline``) before training so the
adapters see the distribution the wrapper will produce at predict time.

Typical usage:

  model = tabfm_v1_0_0_mlx.load()
  lora.apply_lora(model, rank=8)             # freezes base, adds adapters
  lora.train_lora(model, X_train, y_train)   # trains adapters only
  lora.merge_lora(model)                     # folds adapters into the base
"""

import math
from typing import Any, Callable, List, Optional, Tuple

from absl import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

# Attention projections adapted by default. The ICL predictor carries most of
# the parameters (24 blocks at 8x the base width in v1.0.0) and is the module
# that maps context labels to predictions, so it is where adaptation pays off
# first; pass scope="all" to also adapt the column/row towers.
_DEFAULT_TARGET_KEYS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
  """A frozen linear layer with a trainable low-rank residual.

  Computes ``linear(x) + (alpha / rank) * (x @ A) @ B`` where only ``A`` and
  ``B`` are trainable. ``B`` starts at zero, so the wrapped layer is exactly
  equivalent to the original at initialization.

  The adapters are kept in float32 regardless of the base compute dtype
  (bfloat16 by default): adapter gradients are small and bf16 accumulation
  noise can exceed them.
  """

  def __init__(self, linear: nn.Module, rank: int = 8, alpha: float = 16.0):
    super().__init__()
    out_dim, in_dim = linear.weight.shape
    self.linear = linear
    self.scale = alpha / rank
    bound = 1.0 / math.sqrt(in_dim)
    self.lora_a = mx.random.uniform(
        low=-bound, high=bound, shape=(in_dim, rank), dtype=mx.float32
    )
    self.lora_b = mx.zeros((rank, out_dim), dtype=mx.float32)

  def __call__(self, x):
    y = self.linear(x)
    # Adapter math in float32 (x promotes on matmul), cast back at the end.
    z = (x.astype(mx.float32) @ self.lora_a) @ self.lora_b
    return y + (self.scale * z).astype(y.dtype)


def _iter_linear_parents(module: Any, prefix: str = ""):
  """Yields (parent, attribute_name, path, child) for every nn.Linear child."""
  if isinstance(module, nn.Module):
    items = list(module.children().items())
  elif isinstance(module, (list, tuple)):
    items = list(enumerate(module))
  elif isinstance(module, dict):
    items = list(module.items())
  else:
    return
  for name, child in items:
    path = f"{prefix}.{name}" if prefix else str(name)
    if isinstance(child, nn.Linear):
      yield module, name, path, child
    else:
      yield from _iter_linear_parents(child, path)


def _set_child(parent: Any, name: Any, value: Any) -> None:
  if isinstance(parent, nn.Module):
    setattr(parent, str(name), value)
  else:  # list entry (Encoder/SetTransformer blocks) or dict entry
    parent[name] = value


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_keys: Tuple[str, ...] = _DEFAULT_TARGET_KEYS,
    scope: str = "icl",
) -> int:
  """Freezes the model and wraps target linear layers with LoRA adapters.

  Args:
    model: The MLX TabFM model to adapt (modified in place).
    rank: LoRA rank of the adapter matrices.
    alpha: LoRA scaling numerator; the residual is scaled by ``alpha / rank``.
    target_keys: Attribute names of the linear layers to adapt (matched
      against the last path component, e.g. ``("q_proj", "v_proj")``).
    scope: ``"icl"`` to adapt only the ICL predictor (default), ``"all"`` to
      adapt every matching layer in the model.

  Returns:
    The number of linear layers wrapped.
  """
  if scope == "icl":
    root: nn.Module = model.icl_predictor
  elif scope == "all":
    root = model
  else:
    raise ValueError(f"Unsupported scope: {scope!r}. Use 'icl' or 'all'.")

  # Freeze first: modules added afterwards (the adapters) stay trainable, so
  # trainable_parameters() is exactly the LoRA weights.
  model.freeze()

  replaced = 0
  for parent, name, path, child in list(_iter_linear_parents(root)):
    if str(name) in target_keys or path.split(".")[-1] in target_keys:
      _set_child(parent, name, LoRALinear(child, rank=rank, alpha=alpha))
      replaced += 1
  if replaced == 0:
    raise ValueError(
        f"No linear layers matched target_keys={target_keys!r} in scope"
        f" {scope!r}."
    )
  return replaced


def merge_lora(model: nn.Module) -> int:
  """Folds every LoRA adapter back into its wrapped linear layer.

  After merging, the model contains plain ``nn.Linear`` layers whose weights
  are ``W + (alpha / rank) * (A @ B)^T`` -- numerically identical outputs with
  no adapter overhead. Quantized layers cannot be merged.

  Args:
    model: Model previously adapted with ``apply_lora`` (modified in place).

  Returns:
    The number of adapters merged.
  """
  merged = 0

  def _walk(module):
    nonlocal merged
    if isinstance(module, nn.Module):
      items = list(module.children().items())
    elif isinstance(module, (list, tuple)):
      items = list(enumerate(module))
    elif isinstance(module, dict):
      items = list(module.items())
    else:
      return
    for name, child in items:
      if isinstance(child, LoRALinear):
        lin = child.linear
        delta = (child.scale * (child.lora_a @ child.lora_b)).T
        lin.weight = (lin.weight.astype(mx.float32) + delta).astype(
            lin.weight.dtype
        )
        _set_child(module, name, lin)
        merged += 1
      else:
        _walk(child)

  _walk(model)
  mx.eval(model.parameters())
  return merged


def save_adapters(model: nn.Module, path: str) -> None:
  """Saves only the trainable (LoRA) parameters as a safetensors file."""
  mx.save_safetensors(path, dict(tree_flatten(model.trainable_parameters())))


def load_adapters(model: nn.Module, path: str) -> None:
  """Loads adapter weights saved by ``save_adapters`` into an adapted model."""
  model.load_weights(list(mx.load(path).items()), strict=False)


def _episode(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    batch_rows: int,
    context_fraction: float,
) -> Tuple[mx.array, mx.array, mx.array, np.ndarray, int]:
  """Samples one in-context episode: a row batch split into context/targets."""
  n = X.shape[0]
  take = min(batch_rows, n)
  idx = rng.permutation(n)[:take]
  train_size = max(1, min(take - 1, int(take * context_fraction)))
  x_ep = X[idx][None].astype(np.float32)  # [1, T, H]
  y_ep = y[idx].astype(np.float32)
  # Mask target labels with the -100 sentinel. The model only *uses* context
  # labels (positions < train_size), so this is belt-and-braces to make label
  # leakage structurally impossible; it also mirrors the predict-time padding.
  y_in = y_ep.copy()
  y_in[train_size:] = -100.0
  return (
      mx.array(x_ep),
      mx.array(y_in[None]),
      mx.array(np.array([train_size], dtype=np.int32)),
      y_ep[train_size:],
      train_size,
  )


def fit_lora(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    norm_method: Optional[str] = None,
    **train_kwargs: Any,
) -> List[float]:
  """Fits a TabFM estimator's preprocessing, then trains the LoRA adapters.

  ``TabFMClassifier.fit`` / ``TabFMRegressor.fit`` never update model weights:
  they fit the label/feature encoders and the ensemble generator, and cache
  the preprocessed training matrix as the in-context set. This helper reuses
  exactly that machinery so the adapters are trained on the same numeric
  distribution the wrapper feeds the model at predict time, then trains via
  ``train_lora``. Call ``apply_lora(estimator.model, ...)`` first.

  Args:
    estimator: A ``TabFMClassifier`` or ``TabFMRegressor`` wrapping an MLX
      model that has been adapted with ``apply_lora``.
    X: Raw features (DataFrame or array), as you would pass to ``fit``.
    y: Raw targets, as you would pass to ``fit``.
    norm_method: Which fitted normalization view to train on. Defaults to the
      estimator's first norm method (typically ``"none"``).
    **train_kwargs: Forwarded to ``train_lora`` (steps, learning_rate, ...).

  Returns:
    The per-step loss history from ``train_lora``.
  """
  estimator.fit(X, y)
  gen = estimator.ensemble_generator_
  nm = norm_method or gen.norm_methods_[0]
  if nm not in gen.preprocessors_:
    raise ValueError(
        f"norm_method {nm!r} not fitted; available: {list(gen.preprocessors_)}"
    )
  X_np = np.asarray(gen.preprocessors_[nm].X_transformed_, dtype=np.float32)
  y_np = np.asarray(gen.y_)
  if estimator.model.is_classifier:
    train_kwargs.setdefault("num_classes", estimator.n_classes_)
  return train_lora(estimator.model, X_np, y_np, **train_kwargs)


def train_lora(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    steps: int = 200,
    batch_rows: int = 64,
    context_fraction: float = 0.7,
    learning_rate: float = 1e-4,
    num_classes: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
) -> List[float]:
  """Trains the LoRA adapters; the frozen base weights are never updated.

  Args:
    model: Model previously adapted with ``apply_lora``.
    X: Numeric feature matrix of shape (n_samples, n_features), already
      preprocessed the same way the sklearn wrapper preprocesses inputs.
    y: Targets of shape (n_samples,) -- integer class ids in
      ``[0, num_classes)`` for classification, floats for regression.
    steps: Number of optimizer steps (one sampled episode per step).
    batch_rows: Rows per episode (context + targets).
    context_fraction: Fraction of each episode used as in-context examples.
    learning_rate: Adam learning rate for the adapters.
    num_classes: Number of classes (classification only). Defaults to
      ``max(y) + 1``; must be <= the model's ``max_classes``.
    seed: Seed for episode sampling.
    verbose: Log the loss every 10 steps.

  Returns:
    The per-step loss history.
  """
  is_classifier = model.is_classifier
  if is_classifier:
    if num_classes is None:
      num_classes = int(np.max(y)) + 1
    if num_classes > model.max_classes:
      raise ValueError(
          f"num_classes={num_classes} exceeds the model's max_classes="
          f"{model.max_classes}."
      )

  trainable = tree_flatten(model.trainable_parameters())
  if not trainable:
    raise ValueError("No trainable parameters; call apply_lora first.")

  def loss_fn(m, x_ep, y_in, train_size, y_target):
    out = m(x_ep, y_in, train_size)  # [1, T, K]
    target = out[0, int(train_size[0].item()):, :]
    if is_classifier:
      logits = target[:, :num_classes]
      return nn.losses.cross_entropy(
          logits, y_target.astype(mx.int32), reduction="mean"
      )
    return nn.losses.mse_loss(
        target[:, 0], y_target.astype(target.dtype), reduction="mean"
    )

  value_and_grad = nn.value_and_grad(model, loss_fn)
  optimizer = optim.Adam(learning_rate=learning_rate)
  rng = np.random.default_rng(seed)

  losses = []
  for step in range(steps):
    x_ep, y_in, train_size, y_target_np, _ = _episode(
        rng, X, y, batch_rows, context_fraction
    )
    loss, grads = value_and_grad(
        model, x_ep, y_in, train_size, mx.array(y_target_np)
    )
    optimizer.update(model, grads)
    mx.eval(model.trainable_parameters(), optimizer.state, loss)
    losses.append(float(loss.item()))
    if verbose and step % 10 == 0:
      logging.info("lora step %d: loss %.4f", step, losses[-1])
  return losses

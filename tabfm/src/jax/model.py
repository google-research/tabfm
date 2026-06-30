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

"""Tabular Foundation Model (TabFM) architecture implementation.

We use the following notation for tensor axes dimensions:

B: Batch size
T: Number of rows (sequence length)
H: Number of columns (features)
E: Embedding size
I: Number of induced tokens
C: Number of CLS tokens
K: Number of classes
D: Head dimension
N: Number of attention heads
G: Feature group size
F: Number of fourier frequencies
M: Number of ensemble estimators
L: Generic linear/feature dimension
Y: Number of layers (a.k.a blocks)
"""

from absl import logging
import chex
import math
from math import pi
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import enum
import typing

import einops
rearrange = einops.rearrange
repeat = einops.repeat
from flax import nnx
from flax.nnx import Module
import jax
import jax.numpy as jnp
import jaxtyping as jt; import typeguard; import numpy as np; jt.typed = jt.jaxtyped(typechecker=typeguard.typechecked)

Array = jax.Array | np.ndarray
DType = Any
PRNGKey = jax.Array
Float = jt.Float

from . import memory_efficient_attention



class AttentionImplementation(str, enum.Enum):
  JAX = 'jax'
  # vmaps jax.nn.dot_product_attention over the head dimension to save memory
  JAX_VMAP_ON_HEAD_DIM = 'jax_vmap_on_head_dim'
  FLASH = 'flash'
  NONE = 'none'


# Rotary Embedding


@jt.typed
def exists(val: Any) -> bool:
  """Checks if a value is not None."""
  return val is not None


@jt.typed
def default(val: Any, d: Any) -> Any:
  """Returns 'val' if it exists, otherwise returns 'd'."""
  return val if exists(val) else d


@jt.typed
def rotate_half(x: jt.Float[jax.Array | np.ndarray, '*B D_r']) -> jt.Float[jax.Array | np.ndarray, '*B D_r']:
  """Rotates half of the input tensor's dimensions for rotary positional embeddings.

  This operation splits the last dimension into two halves and swaps them,
  negating one half.

  Args:
    x: Input tensor to rotate.

  Returns:
    Rotated tensor.
  """
  # Reshape the input to separate the rotation dimension
  x = rearrange(x, '... (d r) -> ... d r', r=2)
  # Split the last dimension into two parts (x1 and x2 correspond to the original x[..., :d] and x[..., d:])
  x1, x2 = x[..., 0], x[..., 1]
  # Stack them back with x2 negated and in the first position
  x = jnp.stack((-x2, x1), axis=-1)
  # Reshape back to the original format
  return rearrange(x, '... d r -> ... (d r)')


@jt.typed
def apply_rotary_emb(
    freqs: jax.Array | np.ndarray,
    t: jax.Array | np.ndarray,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
) -> jax.Array | np.ndarray:
  """Applies rotary positional embeddings to a tensor 't'.

  Args:
    freqs: Frequencies tensor for the rotation.
    t: Input tensor to apply rotary embedding to, typically query or key.
    start_index: The starting index in the embedding dimension to apply the
      rotation. Useful if only a part of the embedding needs rotation.
    scale: Scaling factor to apply after rotation (used in XPOS).
    seq_dim: The dimension corresponding to the sequence length in the input
      tensor 't'.

  Returns:
    Tensor with rotary embeddings applied.
  """
  dtype = t.dtype

  # Adjust frequencies if the input tensor has 3 dimensions (e.g., (batch, seq_len, dim))
  # and we need to match the sequence length.
  if t.ndim == 3:
    # Get the current sequence length from the specified sequence dimension
    current_seq_len = t.shape[seq_dim]
    # Slice freqs to match the current sequence length. Assumes freqs are pre-computed
    # for a maximum possible sequence length and we take the tail.
    freqs = freqs[freqs.shape[0] - current_seq_len :]

  rot_dim = freqs.shape[-1]  # Dimension of the part to be rotated
  end_index = start_index + rot_dim

  assert rot_dim <= t.shape[-1], (
      f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in'
      f' all the positions {rot_dim}'
  )

  # Split 't' into three parts:
  # 1. 't_left': The part of the tensor before 'start_index' (not rotated).
  # 2. 't_middle': The part of the tensor from 'start_index' to 'end_index' (will be rotated).
  # 3. 't_right': The part of the tensor after 'end_index' (not rotated).
  t_left = t[..., :start_index]
  t_middle = t[..., start_index:end_index]
  t_right = t[..., end_index:]

  # Apply the rotary embedding transformation:
  # (t_middle * cos(freqs) * scale) + (rotate_half(t_middle) * sin(freqs) * scale)
  t_transformed = (t_middle * jnp.cos(freqs) * scale) + (
      rotate_half(t_middle) * jnp.sin(freqs) * scale
  )

  # Concatenate the three parts back together to form the output tensor
  out = jnp.concatenate((t_left, t_transformed, t_right), axis=-1)

  # Ensure the output tensor has the original dtype
  return out.astype(dtype)


class PerDimScale(nnx.Module):
  """Per-dimension scaling."""

  __data__ = ('per_dim_scale',)

  @jt.typed
  def __init__(self, num_dims: int, *, rngs: Any):
    del rngs
    self.num_dims = num_dims
    self.per_dim_scale = nnx.Param(jnp.zeros(shape=(num_dims,)))

  @jt.typed
  def __call__(self, x: jt.Float[jax.Array | np.ndarray, '*B L']) -> jt.Float[jax.Array | np.ndarray, '*B L']:
    """Applies per-dimension scaling.

    Args:
      x: Input tensor.

    Returns:
      Scaled tensor.
    """
    return x * (
        1.442695041
        / jnp.sqrt(self.num_dims)
        * jax.nn.softplus(self.per_dim_scale)
    ).astype(x.dtype)


class RotaryEmbedding(nnx.Module):
  """RotaryEmbedding is a module that implements rotary positional embeddings for

  use in transformer models. Rotary embeddings encode positional information
  in a way that allows continuous rotation of embeddings, enhancing the model's
  ability to capture long-range dependencies and positional relations.
  """

  def __init__(
      self,
      dim: int,
      custom_freqs: Optional[jax.Array | np.ndarray] = None,
      freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
      theta: float = 10000,
      max_freq: int = 10,
      num_freqs: int = 1,
      interpolate_factor: float = 1.0,
      theta_rescale_factor: float = 1.0,
      seq_before_head_dim: bool = True,
      *,
      rngs: Any,  # Required for Flax nnx modules, even if not directly used for random init here.
      dtype: DType = jnp.bfloat16,
  ):
    """Initializes the RotaryEmbedding module.

    Parameters
    ----------
    dim (int): The dimension of the embeddings to apply RoPE to.
    custom_freqs (jax.Array | np.ndarray | None): Custom frequency tensor. If None,
                                    frequencies are generated based on
                                    `freqs_for`.
    freqs_for (Literal["lang", "pixel", "constant"]): Specifies the type of
    frequencies.
                                                    'lang' (language), 'pixel'
                                                    (image), or 'constant'.
    theta (float): Base scaling factor for language frequencies.
    max_freq (int): Maximum frequency for pixel-based embeddings.
    num_freqs (int): Number of frequencies for 'constant' type.
    interpolate_factor (float): Factor to interpolate sequence length for longer
    sequences.
    theta_rescale_factor (float): Rescaling factor for theta for longer
    sequences (NTK-aware scaling).
    seq_before_head_dim (bool): If True, the sequence dimension is before the
    head dimension (e.g., (B, S, H, D)).
                                Otherwise, it's typically (B, H, S, D).
    rngs (nnx.Rngs): Random number generators for NNX.
    dtype (DType): Data type for calculations.
    """
    self.dim = dim
    self.freqs_for = freqs_for
    self.interpolate_factor = interpolate_factor
    self.dtype = dtype

    assert dim >=2, f'dim must be at least 2. Got {dim}'
    # Apply theta rescaling based on NTK-aware scaling for longer sequence lengths
    theta *= theta_rescale_factor ** (dim / (dim - 2))

    # Initialize frequencies based on 'freqs_for'
    if exists(custom_freqs):
      freqs_init = custom_freqs
    elif freqs_for == 'lang':
      # Language-based frequencies (standard RoPE formulation)
      freqs_init = 1.0 / (
          theta
          ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim)
      )
    elif freqs_for == 'pixel':
      # Pixel-based frequencies (often used for images)
      freqs_init = jnp.linspace(1.0, max_freq / 2, dim // 2) * pi
    elif freqs_for == 'constant':
      # Constant frequencies
      freqs_init = jnp.ones(num_freqs).astype(dtype)
    else:
      raise ValueError(f'Unknown freqs_for type: {freqs_for}')

    # Define 'freqs' as an nnx.Param if learnable, otherwise as a regular nnx.Variable
    assert freqs_init is not None  # For pytype.
    self.freqs = nnx.Variable(freqs_init.astype(self.dtype))

    # Determine the default sequence dimension based on `seq_before_head_dim`
    self.default_seq_dim = -3 if seq_before_head_dim else -2

  @jt.typed
  def get_seq_pos(
      self, seq_len: Any, dtype: Any, offset: int = 0
  ) -> jt.Float[jax.Array | np.ndarray, 'T']:
    """Computes the sequence positions for rotary embeddings.

    Args:
      seq_len: Length of the sequence.
      dtype: Data type for the output tensor.
      offset: Offset for the sequence positions.

    Returns:
      Sequence positions tensor.
    """
    return (jnp.arange(seq_len, dtype=dtype) + offset) / self.interpolate_factor

  @jt.typed
  def rotate_queries_or_keys(
      self,
      t: jax.Array | np.ndarray,
      seq_dim: Optional[int] = None,
      offset: int = 0,
      scale: Optional[float] = None,
  ) -> jax.Array | np.ndarray:
    """Applies rotary embeddings to a single tensor (either queries or keys).

    Args:
      t: Input tensor to apply rotary embedding to.
      seq_dim: The dimension corresponding to the sequence length.
      offset: Offset for the sequence positions.
      scale: Scaling factor to apply after rotation.

    Returns:
      Tensor with rotary embeddings applied.
    """
    seq_dim = default(seq_dim, self.default_seq_dim)

    dtype, seq_len = t.dtype, t.shape[seq_dim]

    # Get sequence positions
    seq = self.get_seq_pos(seq_len, dtype=dtype, offset=offset)

    # Call the module's __call__ method to get frequencies
    freqs = self(seq, seq_len=seq_len, offset=offset)

    # If the sequence dimension is -3 (e.g., (batch, seq, head, dim)),
    # rearrange frequencies to match the broadcastable shape.
    if seq_dim == -3:
      freqs = rearrange(freqs, 'n d -> n 1 d')

    # Apply the rotary embedding transformation
    return apply_rotary_emb(
        freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim
    )

  @jt.typed
  def __call__(
      self,
      t: jt.Float[jax.Array | np.ndarray, '*B T'],
      seq_len: Any = None,
      offset: int = 0,
  ) -> jt.Float[jax.Array | np.ndarray, '*B T D_rot']:
    """The main forward pass for generating rotary embedding frequencies.

    Args:
      t: Sequence positions tensor.
      seq_len: Length of the sequence.
      offset: Offset for the sequence positions.

    Returns:
      Frequencies tensor.
    """
    # Get the base frequencies (either learned or pre-computed)
    freqs = self.freqs

    # Calculate frequencies by multiplying positions (t) with base frequencies
    freqs = jnp.einsum('..., f -> ... f', t.astype(freqs.dtype), freqs)
    # Repeat frequencies to match the 'r=2' pattern for complex number rotation
    freqs = repeat(freqs, '... n -> ... (n r)', r=2)

    return freqs


# Attention


# ----------------------------------------------------------------------------
# 1. Helper Functions
#
# Utility functions used by the NNX modules.
# ----------------------------------------------------------------------------


@jt.typed
def get_activation(activation: Union[str, Callable]) -> Callable:
  """Get activation function class from a string name.

  Translated to return JAX activation functions.
  """
  if callable(activation):
    return activation

  activation_map = {
      'relu': nnx.relu,
      'leaky_relu': nnx.leaky_relu,
      'gelu': nnx.gelu,
      'tanh': nnx.tanh,
  }

  if activation not in activation_map:
    raise ValueError(
        f'Unknown activation: {activation}. '
        f'Supported: {list(activation_map.keys())}'
    )
  return activation_map[activation]


# ----------------------------------------------------------------------------
# 2. Translated NNX Modules
#
# flax.nnx.Module implementations.
# ----------------------------------------------------------------------------


class OneHotAndLinear(nnx.Module):
  """Combines one-hot encoding and linear projection in a single efficient

  operation. Flax/NNX implementation.

  Parameters
  ----------
  num_classes : int
      Number of distinct categories for one-hot encoding.
  embed_dim : int
      Output embedding dimension.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      num_classes: int,
      embed_dim: int,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.num_classes = num_classes
    self.embed_dim = embed_dim
    self.projection = nnx.Linear(
        self.num_classes, self.embed_dim, rngs=rngs, dtype=dtype
    )
    self.dtype = dtype

  @jt.typed
  def __call__(self, src: jt.Shaped[jax.Array | np.ndarray, '... T']) -> jt.Float[jax.Array | np.ndarray, '... T E']:
    """Transforms integer indices to dense embeddings.

    Args:
      src: Integer tensor containing category indices.

    Returns:
      Embedded representation.
    """
    # The dtype of the one-hot vector must match the kernel's dtype.
    one_hot = jax.nn.one_hot(
        src, self.num_classes, dtype=self.projection.kernel.dtype
    )
    return self.projection(one_hot)


class MLP(nnx.Module):
  """Multi-layer perceptron with configurable architecture.

  Flax/NNX implementation.

  Parameters
  ----------
  in_dim : int
      Input feature dimension.
  out_dim : Optional[int], default=None
      Output dimension. If None, uses the last hidden dimension.
  hidden_dims : List[int], default=[256, 256, 256]
      Dimensions of hidden layers.
  activation : str, default='gelu'
      Activation function: 'relu', 'gelu', 'leaky_relu', or 'tanh'.
  use_bias : bool, default=True
      Whether to include bias terms in linear layers.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      in_dim: int,
      out_dim: Optional[int] = None,
      hidden_dims: Tuple[int, ...] = (256, 256, 256),
      activation: str = 'gelu',
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.layers = nnx.List()
    act_fn = get_activation(activation)
    self.dtype = dtype

    prev_dim = in_dim
    for hidden_dim in hidden_dims:
      self.layers.append(
          nnx.Linear(
              prev_dim, hidden_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
          )
      )
      self.layers.append(act_fn)
      prev_dim = hidden_dim

    if out_dim is not None:
      self.layers.append(
          nnx.Linear(
              prev_dim, out_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
          )
      )

  @jt.typed
  def __call__(self, x: jt.Float[jax.Array | np.ndarray, '... L_in']) -> jt.Float[jax.Array | np.ndarray, '... L_out']:
    """Forward pass through the MLP.

    Args:
      x: Input tensor.

    Returns:
      Output tensor.
    """
    for layer in self.layers:
      x = layer(x)
    return x


@jt.typed
def _logn(n: int, dtype: jnp.dtype) -> jax.Array | np.ndarray:
  """Compute :math:`\\log(n)` safely, avoiding fp16 overflow for large ``n``."""
  return jnp.array(math.log(max(n, 1)), dtype=dtype)


@jt.typed
def _extract_kv_from_cache(
    cached_kv: Tuple[jt.Float[jax.Array | np.ndarray, 'B T N D'],
                     jt.Float[jax.Array | np.ndarray, 'B T N D']]
) -> Tuple[jt.Float[jax.Array | np.ndarray, 'B T N D'],
           jt.Float[jax.Array | np.ndarray, 'B T N D']]:
  """Extracts key and value tensors from the cache.

  Args:
    cached_kv: Tuple of (key, value) tensors.

  Returns:
    Tuple of (key, value) tensors.
  """
  assert isinstance(cached_kv, tuple), (
      'cached_kv must be a tuple of (k, v).',
      f'cached_kv type: {type(cached_kv)}',
  )
  return cached_kv


@jt.typed
def _encode_kv_into_cache(
    k: jt.Float[jax.Array | np.ndarray, 'B T N D'],
    v: jt.Float[jax.Array | np.ndarray, 'B T N D']
) -> Tuple[jt.Float[jax.Array | np.ndarray, 'B T N D'],
           jt.Float[jax.Array | np.ndarray, 'B T N D']]:
  """Encodes key and value tensors into the cache.

  Args:
    k: Key tensor.
    v: Value tensor.

  Returns:
    Tuple of (key, value) tensors.
  """
  return (k, v)


class MultiheadAttention(nnx.Module):
  """Enhanced multi-head attention with rotary positional embedding support.

  Flax/NNX implementation.

  The logic from the original `multi_head_attention_forward` is integrated
  directly into the `__call__` method for better encapsulation.

  Parameters
  ----------
  embed_dim : int
      Model dimension.
  num_heads : int
      Number of attention heads.
  use_bias : bool, default=True
      Whether to use bias in projection layers.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      num_heads: int,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.attention_impl = attention_impl
    self.dtype = dtype


    assert (
        self.head_dim * num_heads == self.embed_dim
    ), 'embed_dim must be divisible by num_heads'

    kwargs = {}
    if zero_out_proj_init:
      kwargs['kernel_init'] = nnx.initializers.zeros
      if use_bias:
        kwargs['bias_init'] = nnx.initializers.zeros

    self.out_proj = nnx.Linear(
        embed_dim,
        embed_dim,
        use_bias=use_bias,
        rngs=rngs,
        dtype=self.dtype,
        **kwargs,
    )
    # self.out_proj.kernel[...] = jnp.zeros_like(self.out_proj.kernel[...])
    # if self.out_proj.bias is not None:
    #     self.out_proj.bias[...] = jnp.zeros_like(self.out_proj.bias[...])
    self.q_proj = nnx.Linear(
        self.embed_dim, self.embed_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
    )
    self.k_proj = nnx.Linear(
        self.embed_dim, self.embed_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
    )
    self.v_proj = nnx.Linear(
        self.embed_dim, self.embed_dim, use_bias=use_bias, rngs=rngs, dtype=self.dtype
    )
    self.query_ln = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=self.dtype)
    self.key_ln = nnx.RMSNorm(self.head_dim, rngs=rngs, dtype=self.dtype)
    self.per_dim_scale = PerDimScale(num_dims=self.head_dim, rngs=rngs)

  @jt.typed
  def __call__(
      self,
      query: jt.Float[jax.Array | np.ndarray, 'B T E'],
      key: jt.Float[jax.Array | np.ndarray, 'B T_src E'] | None,
      value: jt.Float[jax.Array | np.ndarray, 'B T_src E'] | None,
      attn_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B #N #T T_src']] = None,
      rope: Optional[RotaryEmbedding] = None,
      cached_kv: Optional[
          Tuple[
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],
          ]
      ] = None,
      return_kv: bool = False,
  ) -> Union[
      jt.Float[jax.Array | np.ndarray, 'B T E'],  # Output tensor.
      Tuple[
          jt.Float[jax.Array | np.ndarray, 'B T E'],  # Output tensor.
          Tuple[
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],  # New key cache
              jt.Float[jax.Array | np.ndarray, 'B T_src N D'],  # New value cache
          ],
      ],
  ]:
    """Computes multi-head attention.

    Args:
      query: Query tensor.
      key: Key tensor.
      value: Value tensor.
      attn_mask: Boolean mask for attention.
      rope: Rotary positional embedding instance.
      cached_kv: Cached key and value tensors. If provided, then key and value
        must be None since we are in decode mode and will extract k, v from
        cached_kv.
      return_kv: If True, returns the computed key and value tensors.

    Returns:
      Attention output tensor or (output, (k, v)).
    """
    batch_size, tgt_len, _ = query.shape
    dtype = query.dtype

    is_prefill = return_kv
    is_decode = cached_kv is not None
    is_train = (not return_kv) and (cached_kv is None)
    assert int(is_prefill) + int(is_decode) + int(is_train) == 1, (
        'Exactly one of is_prefill, is_decode, and is_train can be True.'
        + f' Got {is_prefill=}, {is_decode=}, {is_train=}.'
    )

    if attn_mask is not None:
      assert attn_mask.shape[0] == batch_size, (
          f'attn_mask batch size must match query batch size: {attn_mask.shape[0]} != {batch_size}'
      )
    if self.attention_impl == AttentionImplementation.NONE:
      # For None attention, kv cache is empty.
      if return_kv:
        return query, (jnp.array([]), jnp.array([]))
      else:
        return query

    # 1. Project Q
    q = self.q_proj(query)
    q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim)

    # 2. Handle K, V
    if cached_kv is not None:
      assert key is None, f'key must be None if cached_kv is not None {key=}'
      assert value is None, f'value must be None if cached_kv is not None {value=}.'
      k, v = _extract_kv_from_cache(cached_kv)
      src_len = k.shape[-3]  # (Batch, Seq, Head, Dim)
    else:
      assert key is not None, f'key must not be None if cached_kv is None.'
      assert value is not None, f'value must not be None if cached_kv is None.'
      src_len = key.shape[-2]
      k = self.k_proj(key)
      v = self.v_proj(value)
      k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim)
      v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim)

    # 3. Apply RoPE if provided
    if rope is not None:
      q = rope.rotate_queries_or_keys(q)
      # Only rotate K if it was just computed (not cached).
      # We assume cached_kv is already rotated.
      if cached_kv is None:
        k = rope.rotate_queries_or_keys(k)

    q = self.query_ln(q)
    # Only normalize K if it was just computed
    if cached_kv is None:
      k = self.key_ln(k)
    q = self.per_dim_scale(q)


    new_kv = _encode_kv_into_cache(k, v)
    if attn_mask is not None:
      assert attn_mask.ndim == 4, 'attn_mask must be 4D'
      assert (
          attn_mask.shape[-3] == 1
      ), 'attn_mask with ndim=4 must have shape[-3]==1 when using JAX attention'

    if self.attention_impl == AttentionImplementation.FLASH:
      attention_bias = None
      if attn_mask is not None:
        # This assumes attn_mask is not all zeros otherwise this will result
        # in attending to everything equally.
        attention_bias = jnp.where(attn_mask, 0.0, -1e30)
      attn_output = memory_efficient_attention.dot_product_attention_multihead(
          query=q,
          key=k,
          value=v,
          bias=attention_bias,
          dtype=dtype,
          query_chunk_size=128 if tgt_len >= 128 else tgt_len,
          key_chunk_size=128 if src_len >= 128 else src_len,
      )
    elif self.attention_impl == AttentionImplementation.JAX:
      attn_output = jax.nn.dot_product_attention(
          query=q,
          key=k,
          value=v,
          mask=attn_mask,
          scale=1.0,
      )
    elif self.attention_impl == AttentionImplementation.JAX_VMAP_ON_HEAD_DIM:
      if attn_mask is not None:
        attn_mask = attn_mask[:, 0, :, :]
      q_h = jnp.swapaxes(q, -2, 0)
      k_h = jnp.swapaxes(k, -2, 0)
      v_h = jnp.swapaxes(v, -2, 0)

      # Define the function to apply to each head.
      @jax.remat
      def _attention_fn(inputs):
        queries, keys, values = inputs
        return jax.nn.dot_product_attention(
            query=queries,
            key=keys,
            value=values,
            mask=attn_mask,
            scale=1.0,
        )

      # Map the attention function over the head dimension.
      attn_output_h = jax.lax.map(_attention_fn, (q_h, k_h, v_h))

      # Move the head dimension back to its original position.
      attn_output = jnp.swapaxes(attn_output_h, 0, -2)
    else:
      raise ValueError(
          'Unsupported attention implementation: %s'
          % self.attention_impl
      )

    # 6. Reshape and final projection
    flat_output = attn_output.reshape(batch_size, tgt_len, self.embed_dim)
    out = self.out_proj(flat_output)

    if return_kv:
      return out, new_kv
    return out


class MultiheadAttentionBlock(nnx.Module):
  """Attention block supporting RoPE.

  Flax/NNX implementation of TransformerEncoderLayer.

  Parameters
  ----------
  d_model : int
      Model dimension.
  nhead : int
      Number of attention heads.
  dim_feedforward : int
     Dimension of the feedforward network.
  activation : str or unary callable, default="gelu"
      The activation function for the feedforward network.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      activation: str | Callable = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.attn = MultiheadAttention(
        d_model,
        nhead,
        rngs=rngs,
        dtype=dtype,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
    )
    self.pre_attn_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)
    self.post_attn_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)
    self.pre_ff_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)
    self.post_ff_ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=dtype)

    kwargs = {}
    if zero_out_proj_init:
      kwargs['kernel_init'] = nnx.initializers.zeros
      kwargs['bias_init'] = nnx.initializers.zeros

    self.activation_name = (
        activation
        if isinstance(activation, str)
        else getattr(activation, '__name__', None)
    )

    if self.activation_name == 'swiglu':
      self.linear1_gate = nnx.Linear(
          d_model,
          dim_feedforward,
          rngs=rngs,
          dtype=dtype,
          use_bias=use_bias,
          **kwargs,
      )
      self.activation = nnx.silu
    else:
      self.activation = get_activation(activation)

    self.linear1 = nnx.Linear(
        d_model,
        dim_feedforward,
        rngs=rngs,
        dtype=dtype,
        use_bias=use_bias,
        **kwargs,
    )
    self.linear2 = nnx.Linear(
        dim_feedforward,
        d_model,
        rngs=rngs,
        dtype=dtype,
        use_bias=use_bias,
        **kwargs,
    )

    self.dtype = dtype

  def _ff_block(self, x: Array) -> Array:
    x = self.pre_ff_ln(x)
    if getattr(self, 'activation_name', None) == 'swiglu':
      x_gate = self.linear1_gate(x)
      x_val = self.linear1(x)
      x = self.activation(x_gate) * x_val
    else:
      x = self.linear1(x)
      x = self.activation(x)
    x = self.linear2(x)
    x = self.post_ff_ln(x)
    return x

  @jt.typed
  def __call__(
      self,
      q: jt.Float[jax.Array | np.ndarray, 'B T E'],
      k: Optional[jt.Float[jax.Array | np.ndarray, 'B T_src E']] = None,
      v: Optional[jt.Float[jax.Array | np.ndarray, 'B T_src E']] = None,
      attn_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B #N #T T_src']] = None,
      rope: Optional[RotaryEmbedding] = None,
      *,
      cached_kv: Optional[Tuple[jt.Float[jax.Array | np.ndarray, 'B T_src N D'],
                                jt.Float[jax.Array | np.ndarray, 'B T_src N D']]] = None,
      return_kv: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T E'],   # output
             Tuple[jt.Float[jax.Array | np.ndarray, 'B T E'],  # output
                   Tuple[jt.Float[jax.Array | np.ndarray, 'B T_src N D'],  # key cache
                         jt.Float[jax.Array | np.ndarray, 'B T_src N D']]]  # value cache
             ]:
    """Forward pass through the attention block.

    Args:
      q: Query tensor.
      k: Key tensor.
      v: Value tensor.
      attn_mask: Attention mask.
      rope: Rotary positional embedding.
      cached_kv: Cached key/value tensors.
      return_kv: Whether to return updated KV cache.

    Returns:
      Output tensor or (output, new_kv).
    """
    x = q
    q_n = self.pre_attn_ln(q)
    if cached_kv is not None:
      assert k is None, f'k must be None when cached_kv is not None, {k=}'
      assert v is None, f'v must be None when cached_kv is not None, {v=}'
      k_n = None
      v_n = None
    else:
      k = q if k is None else k
      v = q if v is None else v
      k_n = self.pre_attn_ln(k)
      v_n = self.pre_attn_ln(v)

    # Pre-norm: norm -> sublayer -> residual
    attn_res = self.attn(
        query=q_n,
        key=k_n,
        value=v_n,
        attn_mask=attn_mask,
        rope=rope,
        cached_kv=cached_kv,
        return_kv=return_kv,
    )
    if return_kv:
      attn_out, new_kv = attn_res
    else:
      attn_out = attn_res

    attn_out = self.post_attn_ln(attn_out)

    x = x + attn_out
    ff_out = self._ff_block(x)
    x = x + ff_out
    if return_kv:
      return x, new_kv
    return x


class InducedSelfAttentionBlock(nnx.Module):
  """Induced Self-Attention for efficient O(n) attention.

  Flax/NNX implementation.

  Parameters
  ----------
  d_model, nhead, dim_feedforward, activation:
      Parameters for the underlying MultiheadAttentionBlocks.
  num_inds : int
      Number of learnable inducing points.

  rngs : nnx.Rngs
      RNGs for parameter initialization.
  """

  @jt.typed
  def __init__(
      self,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      num_inds: int,
      activation: Union[str, Callable] = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):

    self.num_inds = num_inds
    self.dtype = dtype

    # Learnable inducing points, initialized with truncated normal
    self.ind_vectors = nnx.Param(
        jax.random.truncated_normal(
            rngs.params(), -2, 2, (num_inds, d_model)
        ).astype(self.dtype)
        * 0.02
    )

    # Two-stage attention mechanism
    block_args = {
        'd_model': d_model,
        'nhead': nhead,
        'dim_feedforward': dim_feedforward,
        'activation': activation,
        'rngs': rngs,
        'attention_impl': attention_impl,
        'zero_out_proj_init': zero_out_proj_init,
        'use_bias': use_bias,
        'dtype': dtype,
    }
    self.mab1 = MultiheadAttentionBlock(**block_args)
    self.mab2 = MultiheadAttentionBlock(**block_args)

  @jt.typed
  def _induced_attention(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      *,
      cached_inducing_repr: Optional[jt.Float[jax.Array | np.ndarray, 'B I E']] = None,
      return_inducing_repr: bool = False,
  ) -> Union[Array, Tuple[Array, Array]]:
    """Helper to run the two-stage attention with static masking."""
    *batch_shape, seq_len, d_model = src.shape

    hidden = None
    if cached_inducing_repr is not None:
      hidden = cached_inducing_repr
    else:
      # Broadcast inducing vectors to match batch shape
      ind_vectors = jnp.broadcast_to(
          self.ind_vectors[...], (*batch_shape, self.num_inds, d_model)
      )

      mask = None
      if train_size is not None:
        # train_size is (B,)
        # True means attend
        # mask shape: (B, seq_len)
        mask = jnp.arange(seq_len)[None, :] < train_size[:, None]
        # Reshape for attention: (B, 1, 1, seq_len) to broadcast over heads and query length (num_inds)
        mask = mask[:, None, None, :]

      # Stage 1: Input projects to inducing points.
      hidden = self.mab1(
          q=ind_vectors,
          k=src,
          v=src,
          attn_mask=mask,  # Pass boolean mask
      )

    # Stage 2: Inducing points project back to input (unchanged).
    out = self.mab2(
        q=src,
        k=hidden,
        v=hidden,
    )
    if return_inducing_repr:
      return typing.cast(Array, out), typing.cast(Array, hidden)
    return typing.cast(Array, out)

  @jt.typed
  def __call__(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      *,
      cached_inducing_repr: Optional[jt.Float[jax.Array | np.ndarray, 'B I E']] = None,
      return_inducing_repr: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T E'],
             Tuple[jt.Float[jax.Array | np.ndarray, 'B T E'],   # output
                   jt.Float[jax.Array | np.ndarray, 'B I E']    # cached_inducing_repr
                  ]
            ]:
    """Apply induced self-attention.

    Args:
      src: Input tensor.
      train_size: If provided, limits inducing points' attention.
      cached_inducing_repr: Cached inducing points representation.
      return_inducing_repr: Whether to return the inducing points representation.

    Returns:
      Output tensor or (output, inducing_repr).
    """
    return self._induced_attention(
        src,
        train_size,
        cached_inducing_repr=cached_inducing_repr,
        return_inducing_repr=return_inducing_repr,
    )


# Encoders


class Encoder(Module):
  """A stack of multi-head attention blocks.

  Attributes:
      num_blocks (int): Number of blocks in the encoder.
      d_model (int): The dimension of the model.
      nhead (int): The number of attention heads.
      dim_feedforward (int): The dimension of the feed-forward part of the
        block.
      activation (str): The activation function ('relu' or 'gelu').
      use_rope (bool): If True, use rotary positional embeddings.
      rope_base (float): The base for RoPE.
  """

  @jt.typed
  def __init__(
      self,
      num_blocks: int,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      activation: str = 'gelu',
      use_rope: bool = False,
      rope_base: float = 10000,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Optional[Any] = None,
      dtype: DType = jnp.bfloat16,
      cache_icl_input_only: bool = False,
  ):
    if d_model % nhead != 0:
      raise ValueError(
          f'd_model ({d_model}) must be divisible by nhead ({nhead})'
      )

    activations = {'relu': jax.nn.relu, 'gelu': jax.nn.gelu, 'swiglu': 'swiglu'}
    if activation not in activations:
      raise ValueError(f'Activation must be one of {list(activations.keys())}')

    act_fn = activations[activation]

    @nnx.split_rngs(splits=num_blocks)
    @nnx.vmap(axis_size=num_blocks)
    def create_block(rngs):
      return MultiheadAttentionBlock(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          activation=act_fn,
          attention_impl=attention_impl,
          zero_out_proj_init=zero_out_proj_init,
          use_bias=use_bias,
          rngs=rngs,
          dtype=dtype,
      )

    self.blocks = create_block(rngs)
    self.dtype = dtype
    self.cache_icl_input_only = cache_icl_input_only
    self.rope = (
        RotaryEmbedding(
            dim=d_model // nhead, theta=rope_base, rngs=rngs, dtype=self.dtype
        )
        if use_rope
        else None
    )

  @jt.typed
  def __call__(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      attn_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B #N #T T_prefill']] = None,
      *,
      cached_kv: (
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill E'] |         # For cache_icl_input_only
          Tuple[jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],    # Key cache
                jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D']] |  # Value cache
          None
      )=None,
      return_kv: bool = False,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T E'] |
        Tuple[
          jt.Float[jax.Array | np.ndarray, 'B T E'],            # Output
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill E'] |         # For cache_icl_input_only
          Tuple[jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],    # Key cache
                jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D']]    # Value cache
        ]):
    """Forward pass through the stacked encoder blocks.

    Args:
      src: Input tensor.
      attn_mask: Optional attention mask.
      cached_kv: Optional cached KV tensors for decoding.
      return_kv: If True, returns the updated KV cache.

    Returns:
      Output tensor or (output, kvs).
    """
    if cached_kv is not None and return_kv:
      raise ValueError('Cannot both use cached_kv and return_kv in Encoder.')

    if cached_kv is not None:
      if self.cache_icl_input_only:
        # cached_kv is the stacked pre-block inputs: shape (num_layers, B, T, d_model).
        @nnx.scan(
            in_axes=(0, nnx.Carry, 0),  # block, carry, layer_input
            out_axes=nnx.Carry,
        )
        @nnx.remat
        def scan_fn_cached_input(
            block: MultiheadAttentionBlock,
            carry: jax.Array | np.ndarray,
            layer_input: jax.Array | np.ndarray,
        ):
          out = block(
              q=carry,
              k=layer_input,
              v=layer_input,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=None,
              return_kv=False,
          )
          return out

        final_out = scan_fn_cached_input(self.blocks, src, cached_kv)
        return final_out
      else:
        # cached_kv is a stack of per-layer (K, V) caches.
        @nnx.scan(
            in_axes=(0, nnx.Carry, 0),  # block, carry, cached_kv
            out_axes=nnx.Carry,
        )
        @nnx.remat
        def scan_fn_cached(
            block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray, layer_kv
        ):
          out = block(
              q=carry,
              k=None,
              v=None,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=layer_kv,
              return_kv=False,
          )
          return out

        final_out = scan_fn_cached(self.blocks, src, cached_kv)
        return final_out

    if return_kv:
      if self.cache_icl_input_only:
        # Return per-layer inputs to the block.
        @nnx.scan(
            in_axes=(0, nnx.Carry),  # block, carry
            out_axes=(nnx.Carry, 0),  # carry, layer_input
        )
        @nnx.remat
        def scan_fn_return_input(
            block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray
        ):
          out = block(
              q=carry,
              k=carry,
              v=carry,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=None,
              return_kv=False,
          )
          return out, carry  # carry is the input to this block

        final_out, kvs = scan_fn_return_input(self.blocks, src)
        return final_out, kvs
      else:
        @nnx.scan(
            in_axes=(0, nnx.Carry),  # block, carry
            out_axes=(nnx.Carry, 0),  # carry, layer_kv
        )
        @nnx.remat
        def scan_fn_return_kv(block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray):
          out, new_kv = block(
              q=carry,
              k=carry,
              v=carry,
              attn_mask=attn_mask,
              rope=self.rope,
              cached_kv=None,
              return_kv=True,
          )
          return out, new_kv

        final_out, kvs = scan_fn_return_kv(self.blocks, src)
        return final_out, kvs

    @nnx.scan(
        in_axes=(0, nnx.Carry),  # block, carry
        out_axes=nnx.Carry,
    )
    @nnx.remat
    def scan_fn(block: MultiheadAttentionBlock, carry: jax.Array | np.ndarray):
      out = block(
          q=carry,
          k=carry,
          v=carry,
          attn_mask=attn_mask,
          rope=self.rope,
      )
      return out

    final_out = scan_fn(self.blocks, src)
    return final_out


class SetTransformer(Module):
  """A stack of induced self-attention blocks for permutation-invariant processing.

  Attributes:
      num_blocks (int): The number of induced self-attention blocks.
      d_model (int): The dimension of the model.
      nhead (int): The number of attention heads.
      dim_feedforward (int): The dimension of the feed-forward network.
      num_inds (int): The number of inducing points.
      activation (str): The activation function ('relu' or 'gelu').
  """

  @jt.typed
  def __init__(
      self,
      num_blocks: int,
      d_model: int,
      nhead: int,
      dim_feedforward: int,
      num_inds: int = 13,
      activation: str = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Optional[Any] = None,
      dtype: DType = jnp.bfloat16,
  ):
    self.dtype = dtype
    if d_model % nhead != 0:
      raise ValueError(
          f'd_model ({d_model}) must be divisible by nhead ({nhead})'
      )

    activations = {'relu': jax.nn.relu, 'gelu': jax.nn.gelu, 'swiglu': 'swiglu'}
    if activation not in activations:
      raise ValueError(f'Activation must be one of {list(activations.keys())}')

    act_fn = activations[activation]
    @nnx.split_rngs(splits=num_blocks)
    @nnx.vmap(axis_size=num_blocks)
    def create_block(rngs):
      return InducedSelfAttentionBlock(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          num_inds=num_inds,
          activation=act_fn,
          attention_impl=attention_impl,
          zero_out_proj_init=zero_out_proj_init,
          rngs=rngs,
          use_bias=use_bias,
          dtype=dtype,
      )

    self.blocks = create_block(rngs)

  @jt.typed
  def __call__(
      self,
      src: jt.Float[jax.Array | np.ndarray, 'B T E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      *,
      cached_inducing_repr: Optional[jt.Float[jax.Array | np.ndarray, 'Y B I E']] = None,
      return_inducing_repr: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T E'],    # Output
             # For cached_inducing_repr:
             Tuple[
                 # Output
                 jt.Float[jax.Array | np.ndarray, 'B T E'],
                 # Cache
                 jt.Float[jax.Array | np.ndarray, 'Y B I E']]]:
    """Applies the Set Transformer to the input.

    Args:
      src: Input tensor.
      train_size: Optional training size for masking.
      cached_inducing_repr: Optional cached inducing points representation.
      return_inducing_repr: If True, returns the inducing points representation.

    Returns:
      Output tensor or (output, inducing_repr).
    """
    if cached_inducing_repr is not None and return_inducing_repr:
      raise ValueError(
          'Cannot both use cached_inducing_repr and return_inducing_repr.'
      )

    if cached_inducing_repr is not None:

      @nnx.scan(
          in_axes=(0, nnx.Carry, 0),
          out_axes=nnx.Carry,
      )
      @nnx.remat
      def scan_fn_cached(
          block: InducedSelfAttentionBlock, carry: jax.Array | np.ndarray, layer_repr
      ):
        out = block(
            carry,
            train_size=train_size,
            cached_inducing_repr=layer_repr,
        )
        return out

      final_out = scan_fn_cached(self.blocks, src, cached_inducing_repr)
      return final_out

    elif return_inducing_repr:

      @nnx.scan(
          in_axes=(0, nnx.Carry),
          out_axes=(nnx.Carry, 0),
      )
      @nnx.remat
      def scan_fn_return(block: InducedSelfAttentionBlock, carry: jax.Array | np.ndarray):
        out, repr = block(
            carry,
            train_size=train_size,
            return_inducing_repr=True,
        )
        return out, repr

      final_out, stacked_repr = scan_fn_return(self.blocks, src)
      return final_out, stacked_repr

    else:

      @nnx.scan(
          in_axes=(0, nnx.Carry),  # block, carry, rngs
          out_axes=nnx.Carry,
      )
      @nnx.remat
      def scan_fn(block: InducedSelfAttentionBlock, carry: jax.Array | np.ndarray):
        out = block(
            carry, train_size=train_size
        )
        return out

      final_out = scan_fn(self.blocks, src)
      return final_out


# Column-wise embedding


class YEmbeddingScheme(enum.Enum):
  """Defines how target values (y) are embedded and combined with features (X)."""

  NONE = 'none'
  """y is not used in the cell embedding stage."""

  ADD_Y_TO_X_POST_EMBEDDING = 'add_y_to_x_post_embedding'
  """y is embedded to a vector of size embed_dim and added to the cell embedding *after* the cell embedding projection."""


class CellEmbedder(nnx.Module):
  """Embeds individual cells.

  Parameters
  ----------
  embed_dim : int
      Embedding dimension.
  y_embedding_scheme : YEmbeddingScheme
      The scheme to use for embedding y values.
  max_classes : int
      Number of classes for y embedding lookup table.
  y_col_embedder_encoder_nhid : int
      Number of hidden units for the y encoder MLP.
  is_classifier : bool
      Whether the model is a classifier.
  feature_group : bool, default=False
      Whether to group features with overlap using shifts (equivalent to 'same').
  feature_group_size : int, default=3
      The number of features in each group when feature_group is True.
  fourier_features_num_frequencies : int, default=32
      Number of frequencies for Fourier features.
  fourier_features_sigma : float, default=1.0
      Sigma for initializing Fourier feature frequencies.
  rngs : nnx.Rngs
      RNGs for parameter initialization.
  dtype : DType
      Data type for calculations.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      y_embedding_scheme: YEmbeddingScheme,
      max_classes: int,
      y_col_embedder_encoder_nhid: int,
      *,
      is_classifier: bool,
      feature_group: Union[bool, str] = False,
      feature_group_size: int = 3,
      use_fourier_features: bool = True,
      fourier_features_num_frequencies: int = 32,
      fourier_features_sigma: float = 1.0,
      rngs: Optional[nnx.Rngs] = None,
      dtype: DType = jnp.bfloat16,
  ) -> None:
    self.embed_dim = embed_dim
    self.dtype = dtype
    self.y_embedding_scheme = y_embedding_scheme
    if isinstance(feature_group, str):
      if feature_group.lower() in ['none', 'false', '']:
        feature_group = False
      elif feature_group.lower() in ['same', 'true']:
        feature_group = True
      else:
        raise ValueError(f'Invalid feature_group string: {feature_group}')

    self.feature_group = feature_group
    self.feature_group_size = feature_group_size
    self.use_fourier_features = use_fourier_features
    self.fourier_features_num_frequencies = fourier_features_num_frequencies
    self.fourier_features_sigma = fourier_features_sigma

    self.is_classifier = is_classifier

    in_dim = feature_group_size if self.feature_group else 1

    if self.use_fourier_features:
      # --- New path: Fourier feature projection ---
      # Projects raw input values through learned frequency banks before the
      # linear embedding, giving the model a richer spectral representation.
      linear_in_dim = self.fourier_features_num_frequencies * 2
      if rngs is not None:
        freq_key = rngs.params()
      else:
        freq_key = jax.random.key(42)
      self.fourier_frequencies = nnx.Param(
          jax.random.normal(
              freq_key, (in_dim, self.fourier_features_num_frequencies)
          )
          * self.fourier_features_sigma
      )
      # Separate frequency bank for categorical features so num and cat paths
      # can independently optimize their frequency scales without gradient
      # conflict. Numerics want metric-preserving (moderate ω); cats want
      # high ω to decorrelate adjacent integer-encoded category indices.
      if rngs is not None:
        freq_key_cat = rngs.params()
      else:
        freq_key_cat = jax.random.key(43)
      self.fourier_frequencies_cat = nnx.Param(
          jax.random.normal(
              freq_key_cat, (in_dim, self.fourier_features_num_frequencies)
          )
          * self.fourier_features_sigma
      )
      # Numerical input projection
      self.in_linear = nnx.Linear(
          linear_in_dim,
          embed_dim,
          rngs=rngs,
          dtype=self.dtype,
          bias_init=nnx.initializers.normal(stddev=1e-6),
      )
      # Separate categorical projection: each feature type gets its own learned
      # linear head, giving strictly more expressive power than a shared proj.
      self.in_linear_cat = nnx.Linear(
          linear_in_dim,
          embed_dim,
          rngs=rngs,
          dtype=self.dtype,
          bias_init=nnx.initializers.normal(stddev=1e-6),
      )
    else:
      # --- Old (legacy) path: direct linear projection ---
      # Used when loading checkpoints trained before Fourier features were
      # introduced. Produces the same parameter tree as the old model:
      #   in_linear: (in_dim -> embed_dim)
      # No fourier_frequencies, fourier_frequencies_cat, or in_linear_cat.
      self.in_linear = nnx.Linear(
          in_dim,
          embed_dim,
          rngs=rngs,
          dtype=self.dtype,
          bias_init=nnx.initializers.normal(stddev=1e-6),
      )

    if self.y_embedding_scheme == YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING:
      if self.is_classifier:
        self.y_embedder_lookup = nnx.Embed(
            num_embeddings=max_classes,
            features=embed_dim,
            rngs=rngs,
            dtype=self.dtype,
        )
      else:
        self.y_embedder_lookup = MLP(
            in_dim=1,
            out_dim=embed_dim,
            hidden_dims=(y_col_embedder_encoder_nhid,),
            activation='gelu',
            rngs=rngs,
            dtype=self.dtype,
        )

  @jt.typed
  def feature_grouping(
      self,
      X: jt.Shaped[jax.Array | np.ndarray, 'B T H'],
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None
  ) -> jt.Shaped[jax.Array | np.ndarray, 'B T H G']:
    """Groups features with overlap using shifts.

    Args:
      X: Input feature tensor.
      d: The number of active features for each batch element.

    Returns:
      Grouped feature tensor.
    """
    if not self.feature_group:
      return jnp.expand_dims(X, -1)

    B, T, H = X.shape
    size = self.feature_group_size

    idxs = jnp.arange(H, dtype=jnp.int32)
    stacked = []
    for i in range(size):
      # i=0 -> offset=0, i=1 -> offset=1, i=2 -> offset=3, i=3 -> offset=7 ...
      offset = (2**i) - 1
      if d is not None:
        # Guard against mod-by-zero: XLA silently returns x for x%0.
        d_safe = jnp.maximum(d, 1)
        idx = (idxs[None, None, :] + offset) % d_safe[:, None, None]
        # Extract dynamic indices across the batch gracefully
        stacked.append(jnp.take_along_axis(X, idx, axis=-1))
      else:
        idx = (idxs[None, None, :] + offset) % H
        stacked.append(jnp.take_along_axis(X, idx, axis=-1))
    return jnp.stack(stacked, axis=-1)

  @jt.typed
  def __call__(
      self,
      features: jt.Float[jax.Array | np.ndarray, 'B T H'],
      y: jt.Shaped[Array, 'B T'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      cat_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B H']] = None,
  ) -> jt.Float[jax.Array | np.ndarray, 'B T H E']:
    """Embeds the input features and target values.

    Args:
      features: Input feature tensor.
      y: Target value tensor.
      train_size: The number of training samples for each batch element.
      d: The number of active features for each batch element.
      cat_mask: Boolean mask indicating categorical features.

    Returns:
      Cell embeddings.
    """
    B, T, H = features.shape
    E = self.embed_dim

    features_grouped = self.feature_grouping(features, d=d)

    HC = features_grouped.shape[-2]

    if self.use_fourier_features:
      # --- New path: Fourier feature projection ---
      features_expanded_raw = features_grouped  # (B, T, HC, in_dim)

      # features_expanded shape: (B, T, HC, in_dim)
      # fourier_frequencies shape: (in_dim, num_frequencies)
      x_proj = jnp.einsum(
          '...i,if->...if', features_expanded_raw, self.fourier_frequencies.value
      )
      # Always keep fourier_feats in per-slot shape: (B, T, HC, G, num_freq*2).
      # When feature_group=False, feature_grouping returns (B, T, HC, 1), so G=1
      # (a singleton). Summing over axis=-2 later collapses it correctly in both
      # cases, removing the need to branch on self.feature_group here.
      features_expanded = jnp.concatenate(
          [jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1
      )  # (B, T, HC, G, num_freq*2)

      # Route through numerical or categorical projection per slot, then sum over G.
      num_out = self.in_linear(features_expanded)  # (B, T, HC, G, E)

      if cat_mask is not None:
        # Group cat_mask exactly like features (X) were grouped.
        # Expand to (B, 1, HC) so feature_grouping sees shape (B, T, HC).
        cat_mask_grouped = self.feature_grouping(cat_mask[:, None, :], d=d)  # (B, 1, HC, G)
        # Broadcast to (B, 1, HC, G, 1) to match (B, T, HC, G, E)
        cat_mask_per_slot = cat_mask_grouped[:, :, :, :, None]

        # Cat-specific Fourier bank: project raw values through ω_cat.
        x_proj_cat = jnp.einsum(
            '...i,if->...if',
            features_expanded_raw,
            self.fourier_frequencies_cat.value,
        )  # (B, T, HC, G, num_freq)
        fe_cat_fourier = jnp.concatenate(
            [jnp.sin(x_proj_cat), jnp.cos(x_proj_cat)], axis=-1
        )  # (B, T, HC, G, num_freq*2)
        cat_out = self.in_linear_cat(fe_cat_fourier)  # (B, T, HC, G, E)

        selected = jnp.where(cat_mask_per_slot, cat_out, num_out)
        cell_embeddings = selected.sum(axis=-2)  # (B, T, HC, E)
      else:
        # No cat_mask: treat all as numerical and sum across slots (or singleton).
        cell_embeddings = num_out.sum(axis=-2)  # (B, T, HC, E)
    else:
      # --- Old (legacy) path: direct linear projection ---
      # Matches the parameter tree of checkpoints trained before Fourier
      # features were introduced. in_linear projects (in_dim -> embed_dim)
      # directly from raw scalar/grouped values.
      cell_embeddings = self.in_linear(features_grouped)  # (B, T, HC, E)

    if self.y_embedding_scheme == YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING:
      if self.is_classifier:
        # Embed y using a lookup table.
        y_embedded: Float[Array, '... T E'] = self.y_embedder_lookup(
            y.astype(jnp.int32)
        )
      else:
        # Embed y as a continuous feature.
        # TODO: Try using Fourier features for the continuous y embedding (as
        # done for X in feature_grouping) instead of a plain MLP, so the model
        # gets the same frequency-rich representation for target values.
        y_embedded: Float[Array, '... T E'] = self.y_embedder_lookup(y[..., None])

      if train_size is not None:
        # Create a mask for the training samples.
        train_mask = jnp.arange(T)[None, :] < train_size[:, None]  # (B, T)
        train_mask = train_mask[..., None, None]  # (B, T, 1, 1)
        # Add the embedded y only to the training samples.
        cell_embeddings = jnp.where(
            train_mask,
            cell_embeddings + y_embedded[:, :, None, :],
            cell_embeddings,
        )
      else:
        # If train_size is not provided, add to all samples.
        cell_embeddings = cell_embeddings + y_embedded[:, :, None, :]

    if d is not None:
      # (B, 1, HC, 1)
      mask = jnp.arange(HC)[None, None, :, None] < d[:, None, None, None]
      cell_embeddings = jnp.where(mask, cell_embeddings, 0.0)

    chex.assert_shape(cell_embeddings, (B, T, HC, E))
    return cell_embeddings


class ColEmbedding(nnx.Module):
  """Distribution-aware column-wise embedding in Flax.

  This module maps each scalar cell in a column to a high-dimensional embedding
  while capturing statistical regularities within the column. Unlike traditional
  approaches that use separate embedding layers per column, it employs a shared
  set transformer to process all features.

  ColEmbedding operates as follows:
  1. Each scalar cell is first linearly projected into the embedding dimension.
  2. The set transformer generates distribution-aware weights and biases for
  each column.
  3. The final column embeddings are computed as: column * weights + biases.

  Parameters
  ----------
  embed_dim : int
      Embedding dimension.
  num_blocks : int
      Number of induced self-attention blocks in the set transformer.
  nhead : int
      Number of attention heads of the set transformer.
  dim_feedforward : int
      Dimension of the feedforward network of the set transformer.
  num_inds : int
      Number of inducing points used in self-attention blocks of the set
      transformer.
  activation : str or callable, default="gelu"
      The activation function used in the feedforward network, can be
      either a string ("relu" or "gelu") or a unary callable.
  reserve_cls_tokens : int, default=4
      Number of slots to reserve for CLS tokens to avoid concatenation.
  rngs : nnx.Rngs, optional
      The random number generators for the module.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      num_blocks: int,
      nhead: int,
      dim_feedforward: int,
      num_inds: int,
      activation: str | Callable = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Optional[nnx.Rngs] = None,
      dtype: DType = jnp.bfloat16,
  ) -> None:
    self.embed_dim = embed_dim
    self.dtype = dtype

    self.tf_col = SetTransformer(
        num_blocks=num_blocks,
        d_model=embed_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_inds=num_inds,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )

    self.out_w = nnx.Linear(
        embed_dim,
        embed_dim,
        rngs=rngs,
        dtype=self.dtype,
        bias_init=nnx.initializers.normal(stddev=1e-6),
    )
    self.ln_w = nnx.RMSNorm(embed_dim, rngs=rngs, dtype=self.dtype)

  @jt.typed
  def __call__(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H E'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      feature_shuffles: Optional[List[List[int]]] = None,
      *,
      cached_repr: Optional[jt.Float[jax.Array | np.ndarray, 'Y B*H I E']] = None,
      return_repr: bool = False,
  ) -> Union[jt.Float[jax.Array | np.ndarray, 'B T H E'],  # Output
             Tuple[jt.Float[jax.Array | np.ndarray, 'B T H E'], # Output
                   jt.Float[jax.Array | np.ndarray, 'Y B*H I E']  # cached_repr
                  ]]:
    """Transform input table into embeddings.

    Args:
      X: Input tensor.
      train_size: Position to split the input into training and test data.
      feature_shuffles: A list of feature shuffle patterns for each table.
      cached_repr: Transformer attention cache.
      return_repr: If True, return the new representations computed in this call.

    Returns:
      Embeddings or (embeddings, new_repr).
    """
    assert not (cached_repr is not None and return_repr), (
        'Cannot have both cached_repr not None and return_repr set to True.'
    )
    B, T, HC, E = X.shape
    padded_train_size = T
    if cached_repr is not None:
      assert train_size is None, (
          'train_size must be None when cached_repr is not None.'
      )

    assert padded_train_size is not None
    train_size_expanded = None
    if train_size is not None:
      if train_size.ndim == 2:
        train_size = jnp.squeeze(train_size, axis=-1)
      chex.assert_shape(train_size, (B,))
      chex.assert_shape(padded_train_size, ())
      # Expand train_size to match the flattened batch dimension (B * HC)
      # Repeat for each feature column: (B,) -> (B * HC,)
      train_size_expanded = jnp.repeat(train_size, HC)

    X = X.transpose((0, 2, 1, 3))
    src = X.reshape(B * HC, T, E)

    representations: Array
    new_repr: Optional[Array] = None

    tf_col_st = typing.cast(SetTransformer, self.tf_col)
    if cached_repr is not None:    # Decode.
      representations = typing.cast(
          Array,
          tf_col_st(
              src,
              # train_size is not used in SetTransformer during decode.
              train_size=None,
              cached_inducing_repr=cached_repr,
          ),
      )
    elif return_repr:             # Prefill.
      representations, new_repr = tf_col_st(
          src,
          train_size=train_size_expanded,
          return_inducing_repr=True,
      )
    else:                        # Train.
      representations = typing.cast(
          Array,
          tf_col_st(
              src,
              train_size=train_size_expanded,
          ),
      )

    # 3. Output Projection & Normalization
    embeddings = self.ln_w(self.out_w(representations))  # (B * HC, T, E)

    # Reshape back to original table structure
    # (B * HC, T, E) -> (B, HC, T, E)
    embeddings = embeddings.reshape((B, HC, T, self.embed_dim))

    # Transpose to (B, T, HC, E)
    final_embeddings = embeddings.transpose((0, 2, 1, 3))

    if return_repr:
      return final_embeddings, typing.cast(Array, new_repr)
    return final_embeddings

# Row-wise embedding

class RowInteraction(nnx.Module):
  """Context-aware row-wise interaction, rewritten in Flax NNX.

  This module captures interactions between features within each row using a
  transformer
  encoder. It prepends learnable class tokens to the feature embeddings and uses
  these tokens to aggregate information.
  """

  @jt.typed
  def __init__(
      self,
      embed_dim: int,
      num_blocks: int,
      nhead: int,
      dim_feedforward: int,
      num_cls: int = 4,
      rope_base: float = 100000,
      activation: str | Callable = 'gelu',
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      output_full_sequence: bool = False,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    self.embed_dim = embed_dim
    self.num_cls = num_cls
    self.dtype = dtype
    self.output_full_sequence = output_full_sequence

    self.tf_row = Encoder(
        num_blocks=num_blocks,
        d_model=embed_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        activation=activation,
        use_rope=True,
        rope_base=rope_base,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
        rngs=rngs,
        dtype=self.dtype,
    )

    self.out_ln = nnx.RMSNorm(embed_dim, rngs=rngs, dtype=self.dtype)

  @jt.typed
  def __call__(
      self,
      embeddings: jt.Float[jax.Array | np.ndarray, 'B T H E'],
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T H E']
        | jt.Float[jax.Array | np.ndarray, 'B T C_TIMES_E']): # if output_full_sequence is False.
    """Captures interactions between features within each row.

    Args:
      embeddings: Input feature embeddings.
      d: The number of active features for each batch element.

    Returns:
      Context-aware representations.
    """
    B, T, HC, E = embeddings.shape
    embeddings_reshaped = embeddings.reshape(B * T, HC, E)

    # Create mask to prevent attention to padding features if `d` is provided
    attn_mask = None
    if d is not None:
      if d.ndim == 2:
        d = jnp.squeeze(d, axis=-1)
      # `d` is (B,), representing number of features for each table in the batch
      d_padded = d + self.num_cls
      indices = jnp.arange(HC)
      # Mask is True for positions to be attended to. Shape: (B, HC)
      mask_per_table = indices < d_padded[:, None]
      # Expand and reshape to match the transformer input: (B*T, HC)
      attn_mask = jnp.repeat(mask_per_table, T, axis=0)
      # Expand to (B*T, 1, HC) for attention mask
      attn_mask = attn_mask[:, None, None, :]

    # Process through the transformer
    outputs = self.tf_row(
        embeddings_reshaped,
        attn_mask=attn_mask,
    )
    if self.output_full_sequence:
      outputs = self.out_ln(outputs)

      # Reshape flattened symbols back to (B, T, HC, E)
      representations = outputs.reshape(B, T, HC, -1)
    else:
      # Extract, normalize, and flatten CLS token outputs
      cls_outputs = outputs[:, : self.num_cls, :]
      cls_outputs = self.out_ln(cls_outputs)

      # Reshape flattened CLS outputs back to (B, T, C*E)
      representations = cls_outputs.reshape(B, T, -1)

    return representations


@nnx.dataclass
class ICLearningCache:
  layer_caches: (
      jt.Float[jax.Array | np.ndarray, 'Y B T_prefill E']  # For cache_icl_input_only
      | Tuple[
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],  # Key cache
          jt.Float[jax.Array | np.ndarray, 'Y B T_prefill N D'],  # Value cache
      ]
  )
  prefill_train_size: jt.Int[jax.Array | np.ndarray, 'B']


class ICLearning(nnx.Module):
  """Dataset-wise in-context learning.

  This module is rewritten in Flax NNX and implements in-context learning that:
  1. Takes row representations and training labels as input.
  2. Conditions the model on training examples.
  3. Makes predictions for test examples based on learned patterns.

  parameters
  ----------
  max_classes : int
      Number of classes that the model supports.
  d_model : int
      Model dimension.
  num_blocks : int
      Number of blocks used in the ICL encoder.
  nhead : int
      Number of attention heads of the ICL encoder.
  dim_feedforward : int
      Dimension of the feedforward network of the ICL encoder.
  activation : str, default="gelu"
      The activation function used in the feedforward network.
  rngs : nnx.Rngs, optional
      RNGs for initialization.
  """
  @jt.typed
  def __init__(
      self,
      loss: str,
      max_classes: int,
      d_model: int,
      num_blocks: int,
      nhead: int,
      dim_feedforward: int,
      activation: str = 'gelu',
      zero_out_proj_init: bool = False,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      use_bias: bool = True,
      cache_icl_input_only: bool = False,
      *,
      rngs: Any | None = None,
      dtype: DType = jnp.bfloat16,
  ):
    super().__init__()
    self.attention_impl = attention_impl
    self.max_classes = max_classes
    self.dtype = dtype
    self.loss = loss
    self.is_classifier = self.loss == 'cross_entropy'
    self.cache_icl_input_only = cache_icl_input_only

    self.tf_icl = Encoder(
        num_blocks=num_blocks,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        activation=activation,
        use_rope=False,
        attention_impl=attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
        cache_icl_input_only=cache_icl_input_only,
        rngs=rngs,
        dtype=self.dtype,
    )
    self.ln = nnx.RMSNorm(d_model, rngs=rngs, dtype=self.dtype)
    if self.is_classifier:
      self.y_encoder = OneHotAndLinear(
          max_classes, d_model, rngs=rngs, dtype=self.dtype
      )
      self.decoder = MLP(
          in_dim=d_model,
          out_dim=max_classes,
          hidden_dims=(d_model * 2,),
          rngs=rngs,
          dtype=self.dtype,
      )
    else:
      self.y_encoder = MLP(
          in_dim=1,
          out_dim=d_model,
          hidden_dims=(d_model * 2,),
          rngs=rngs,
          dtype=self.dtype,
      )
      if self.loss == 'rmse' or self.loss == 'mse':
        out_dim = 1
      else:
        raise ValueError(f'Unsupported regression loss: {self.loss}')
      self.decoder = MLP(
          in_dim=d_model,
          out_dim=out_dim,
          hidden_dims=(d_model * 2,),
          rngs=rngs,
          dtype=self.dtype,
      )

  @jt.typed
  def _prefill_sequence_length_from_cache(
      self, cache: ICLearningCache
  ) -> Any:
    """Returns the sequence length of the prefill cache."""
    assert cache.layer_caches is not None, 'Layer caches must be non-empty.'
    if self.cache_icl_input_only:
      # Shape (Y, B, T, E)
      return cache.layer_caches.shape[2]
    else:
      # Tuple of (K, V), each shape (Y, B, T, N, D)
      k, _ = cache.layer_caches
      return k.shape[2]

  @jt.typed
  def __call__(
      self,
      R: jt.Float[jax.Array | np.ndarray, 'B T E'],
      y: jt.Shaped[jax.Array | np.ndarray, 'B T'],
      train_size: Optional[jt.Int[jax.Array | np.ndarray, 'B']],
      *,
      cache: Optional[ICLearningCache] = None,
      return_cache: bool = False,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E']
        | Tuple[
            jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E'],
            ICLearningCache
          ]):
    """Forward pass for ICLearning."""
    is_prefill = return_cache
    is_decode = cache is not None
    is_train = not is_prefill and not is_decode
    assert int(is_prefill) + int(is_decode) + int(is_train) == 1, (
        'Exactly one of is_prefill, is_decode, and is_train can be True.'
        + f' Got {is_prefill=}, {is_decode=}, {is_train=}.'
    )

    y = y.astype(self.dtype)
    if self.is_classifier:
      y_encoded = self.y_encoder(y.astype(jnp.int32))
    else:
      y_encoded = self.y_encoder(y[..., None])

    B, sequence_length, _ = R.shape

    if is_train or is_prefill:
      assert train_size is not None
      if train_size.ndim == 2:
        train_size = jnp.squeeze(train_size, axis=-1)
      train_mask = (
          jnp.arange(sequence_length)[None, :] < train_size[:, None]
      )
      full_attn_mask = train_mask[:, None, None, :]
      y_encoded = y_encoded * train_mask.reshape(B, sequence_length, 1)
      R = R + y_encoded
    else:
      assert is_decode
      prefill_train_size = cache.prefill_train_size
      prefill_sequence_length = self._prefill_sequence_length_from_cache(cache)
      train_mask = (
          jnp.arange(prefill_sequence_length)[None, :] < prefill_train_size[:, None]
      )
      full_attn_mask = train_mask[:, None, None, :]

    new_cache = None
    if is_train:
      result = self.tf_icl(R, attn_mask=full_attn_mask)
    elif is_prefill:
      result, new_layer_caches = self.tf_icl(
          R, attn_mask=full_attn_mask, return_kv=True
      )
      new_cache = ICLearningCache(
          layer_caches=new_layer_caches,
          prefill_train_size=train_size
      )
    else:
      assert is_decode
      assert train_size is None
      result = self.tf_icl(
          R, attn_mask=full_attn_mask, cached_kv=cache.layer_caches
      )



    result = self.ln(result)
    result = self.decoder(result)
    if return_cache:
      assert new_cache is not None
      return result, new_cache
    return result

# TabFM

class TabFM(nnx.Module):
  """A Tabular Foundation Model (TabFM), rewritten in Flax NNX.

  TabFM is a transformer-based architecture for in-context learning on tabular
  data to make
  predictions without fine-tuning. It processes tabular data through three
  sequential stages:

  1. Column-wise embedding creates distribution-aware embeddings.
  2. Row-wise interaction captures interactions between features within each
  row.
  3. Dataset-wise in-context learning learns patterns from labeled examples and
  makes predictions.

  Datasets with more than `max_classes` classes are not supported.

  Parameters
  ----------
  max_classes : int, default=10
      Number of classes that the model supports.
  embed_dim : int, default=128
      Model dimension used in the column/row embedding transformers. For the
      in-context
      learning transformer, the dimension is this value multiplied by the number
      of CLS tokens.
  col_num_blocks : int, default=3
      Number of induced self-attention blocks in the column embedding
      transformer.
  col_nhead : int, default=4
      Number of attention heads in the column embedding transformer.
  col_num_inds : int, default=128
      Number of inducing points in the column embedding transformer.
  row_num_blocks : int, default=3
      Number of attention blocks in the row interaction transformer.
  row_nhead : int, default=8
      Number of attention heads in the row interaction transformer.
  row_num_cls : int, default=4
      Number of learnable CLS tokens used to aggregate feature information per
      row.
  row_rope_base : float, default=100000
      Base scaling factor for rotary position encoding in the row interaction
      transformer.
  icl_num_blocks : int, default=12
      Number of transformer blocks in the in-context learning transformer.
  icl_nhead : int, default=4
      Number of attention heads in the in-context learning transformer.
  ff_factor : int, default=2
      Expansion factor for feedforward networks across all components.
  activation : str or callable, default="gelu"
      Activation function used throughout the model.
  y_embedding_scheme : YEmbeddingScheme, default=YEmbeddingScheme.NONE
      The scheme to use for embedding y values.
  y_col_embedder_encoder_nhid: int, default=6
      Number of hidden units for the encoder in the y
      column embedding MLP. Only used when y_embedding_scheme is
      ADD_Y_TO_X_POST_EMBEDDING with a regression target.
  feature_group : bool, default=False
      Whether to group features for cell embedding. If 'none', features are
      embedded independently. If 'same', groups features with overlap using
      shifts.
  feature_group_size : int, default=3
      The number of features in each group when feature_group is not False.
  col_attention_impl : AttentionImplementation, default=AttentionImplementation.JAX
      Which attention implementation to use for column embedding.
  row_attention_impl : AttentionImplementation, default=AttentionImplementation.JAX
      Which attention implementation to use for row interaction.
  icl_attention_impl : AttentionImplementation, default=AttentionImplementation.JAX
      Which attention implementation to use for attention layers in ICL.
  use_fourier_features : bool, default=True
      If True (default), CellEmbedder uses learned Fourier frequency banks to
      project raw feature values before the linear embedding. Set to False when
      loading checkpoints trained before this feature was introduced; doing so
      reconstructs the original parameter tree (single ``in_linear`` mapping
      ``in_dim -> embed_dim`` directly), allowing Orbax to restore the state
      without key-shape mismatches.
  rngs : nnx.Rngs
      The random number generators for initialization.
  """

  @jt.typed
  def __init__(
      self,
      loss: str = 'cross_entropy',
      max_classes: int = 10,
      embed_dim: int = 128,
      col_num_blocks: int = 3,
      col_nhead: int = 4,
      col_num_inds: int = 128,
      row_num_blocks: int = 3,
      row_nhead: int = 8,
      row_num_cls: int = 4,
      row_rope_base: float = 100000,
      icl_num_blocks: int = 12,
      icl_nhead: int = 4,
      col_attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      row_attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      icl_attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      ff_factor: int = 2,
      activation: str | Callable = 'gelu',
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
      y_col_embedder_encoder_nhid: int = 6,
      cache_icl_input_only: bool = False,
      zero_out_proj_init: bool = False,
      use_bias: bool = True,
      feature_group: Union[bool, str] = False,
      feature_group_size: int = 3,
      use_fourier_features: bool = True,
      fourier_features_num_frequencies: int = 32,
      fourier_features_sigma: float = 1.0,
      *,
      rngs: Any,
      dtype: DType = jnp.bfloat16,
  ):
    super().__init__()
    self.max_classes = max_classes
    self.embed_dim = embed_dim
    self.row_num_cls = row_num_cls
    self.dtype = dtype
    self.loss = loss
    self.is_classifier = self.loss == 'cross_entropy'

    self.cls_tokens = nnx.Param(
        nnx.initializers.truncated_normal(stddev=0.02)(
            rngs.params(), (row_num_cls, embed_dim)
        ).astype(self.dtype)
    )

    # Stage 1: Column-wise Embedding
    self.cell_embedder = CellEmbedder(
        embed_dim=embed_dim,
        y_embedding_scheme=y_embedding_scheme,
        max_classes=max_classes,
        y_col_embedder_encoder_nhid=y_col_embedder_encoder_nhid,
        is_classifier=self.is_classifier,
        feature_group=feature_group,
        feature_group_size=feature_group_size,
        use_fourier_features=use_fourier_features,
        fourier_features_num_frequencies=fourier_features_num_frequencies,
        fourier_features_sigma=fourier_features_sigma,
        rngs=rngs,
        dtype=self.dtype,
    )
    self.col_embedder = ColEmbedding(
        embed_dim=embed_dim,
        num_blocks=col_num_blocks,
        nhead=col_nhead,
        num_inds=col_num_inds,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        attention_impl=col_attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )
    self.col_embedder_2 = ColEmbedding(
        embed_dim=embed_dim,
        num_blocks=col_num_blocks,
        nhead=col_nhead,
        num_inds=col_num_inds,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        attention_impl=col_attention_impl,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )

    # Stage 2: Row-wise Interaction
    self.row_interactor = RowInteraction(
        embed_dim=embed_dim,
        num_blocks=row_num_blocks,
        nhead=row_nhead,
        num_cls=row_num_cls,
        rope_base=row_rope_base,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        attention_impl=row_attention_impl,
        rngs=rngs,
        dtype=self.dtype,
        output_full_sequence=True,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )
    self.row_interactor_2 = RowInteraction(
        embed_dim=embed_dim,
        num_blocks=row_num_blocks,
        nhead=row_nhead,
        num_cls=row_num_cls,
        rope_base=row_rope_base,
        dim_feedforward=embed_dim * ff_factor,
        activation=activation,
        attention_impl=row_attention_impl,
        rngs=rngs,
        dtype=self.dtype,
        output_full_sequence=False,
        zero_out_proj_init=zero_out_proj_init,
        use_bias=use_bias,
    )
    # Stage 3: Dataset-wise In-Context Learning
    icl_dim = embed_dim * row_num_cls  # CLS tokens are concatenated for ICL
    self.icl_predictor = ICLearning(
        loss=loss,
        max_classes=max_classes,
        d_model=icl_dim,
        num_blocks=icl_num_blocks,
        nhead=icl_nhead,
        dim_feedforward=icl_dim * ff_factor,
        activation=activation,
        rngs=rngs,
        dtype=self.dtype,
        zero_out_proj_init=zero_out_proj_init,
        attention_impl=icl_attention_impl,
        use_bias=use_bias,
        cache_icl_input_only=cache_icl_input_only,
    )
    self.y_embedding_scheme = y_embedding_scheme

  @jt.typed
  def __call__(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H'],
      y: jt.Shaped[Array, 'B T'],
      train_size: jt.Int[jax.Array | np.ndarray, 'B'] | jt.Int[jax.Array | np.ndarray, 'B 1'],
      d: jt.Int[jax.Array | np.ndarray, 'B'] | jt.Int[jax.Array | np.ndarray, 'B 1'] | None = None,
      cat_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B H']] = None,
      softmax_temperature: float = 0.9,
      num_classes: Optional[int] = None,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T E']      # Regression
        | jt.Float[jax.Array | np.ndarray, 'B T K']):  # Classification
    """Processes tabular data through nested encoders and ICL predictor.

    Args:
      X: Input feature tensor.
      y: Target value tensor.
      train_size: Number of training samples per batch element.
      d: Number of features per dataset.
      cat_mask: Boolean mask indicating categorical features.
      softmax_temperature: Temperature for softmax (inference only).
      num_classes: Number of classes in the dataset.

    Returns:
      Logits or probabilities for the sequence.
    """
    # Cast input to bfloat16 at the beginning of the model call
    X = jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)

    _, T, _ = X.shape
    if train_size.ndim == 2:
      train_size = jnp.squeeze(train_size, axis=-1)
    if d is not None:
      if d.ndim == 2:
        d = jnp.squeeze(d, axis=-1)
    valid_mask = X != -100.0

    train_size = train_size.astype(jnp.int32)

    # Stage 1: Cell-wise embedding
    embeddings = self.cell_embedder(
        X, y, train_size=train_size, d=d, cat_mask=cat_mask
    )  # (B, T, HC, E)


    # Column-wise embedding
    embeddings = typing.cast(
        Array,
        self.col_embedder(
            embeddings,
            train_size=train_size,
        ),
    )
    # Prepend CLS tokens
    B1, T1, _, _ = embeddings.shape
    cls_tokens_expanded = jnp.broadcast_to(
        self.cls_tokens[...], (B1, T1, self.row_num_cls, self.embed_dim)
    )

    embeddings = jnp.concatenate(
        [cls_tokens_expanded, embeddings], axis=-2
    )

    embeddings = self.row_interactor(
        embeddings,
        d=d,
    )


    embeddings = typing.cast(
        Array,
        self.col_embedder_2(
            embeddings,
            train_size=train_size,
        ),
    )

    # Stage 2: Row-wise interaction
    representations = self.row_interactor_2(
        embeddings,
        d=d,
    )

    if (
        self.is_classifier
        and num_classes is not None
        and num_classes > self.icl_predictor.max_classes
    ):
      raise ValueError(
          f'Number of classes ({num_classes}) exceeds the maximum supported '
          f'({self.icl_predictor.max_classes}).'
      )
    out = self.icl_predictor(
        representations,
        y,
        train_size=train_size,
    )

    return out

  @jt.typed
  def prefill(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H'],
      y: jt.Shaped[jax.Array | np.ndarray, 'B T'],
      d: jt.Int[jax.Array | np.ndarray, 'B'] = None,
      cat_mask: jt.Bool[jax.Array | np.ndarray, 'B H'] | None = None,
  ) -> (
      Tuple[(jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E']),
             Dict[str, Any] # cache
      ]):
    """Prefills the model with training data and returns the cache.

    Args:
      X: Input feature tensor.
      y: Target value tensor.
      d: Number of features per dataset.
      cat_mask: Boolean mask indicating categorical features.

    Returns:
      Tuple of (logits, updated_cache).
    """
    X = jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)
    y = y.astype(self.dtype)

    _, T, _ = X.shape

    # Pad T to be a multiple of 128 for efficiency.
    # TODO: Do this only when use_flash_attention is True.
    block_size = 128
    pad_len = ((T - 1) // block_size + 1) * block_size - T
    paddings_X = [(0, 0), (0, pad_len), (0, 0)]
    X = jnp.pad(X, paddings_X, constant_values=-100.0)
    paddings_y = [(0, 0), (0, pad_len)]
    y = jnp.pad(y, paddings_y, constant_values=-100.0)
    total_padded_length = X.shape[1]

    # For prefill, all data is training data.
    # We calculate train_size based on valid y values to handle external padding.
    # TODO: Pass 'train_size' as an argument to prefill so the '-100'
    # value can be used for regression tasks.
    assert y.ndim == 2, 'y should have shape (B, T)'
    is_valid = jnp.not_equal(y, -100.0)
    train_size = jnp.sum(is_valid, axis=-1, dtype=jnp.int32).astype(jnp.int32)

    # Stage 0: Cell-wise embedding
    cell_embeddings = self.cell_embedder(
        X, y, train_size=train_size, d=d, cat_mask=cat_mask
    )


    # Stage 1: Column-wise embedding
    res = typing.cast(
        Tuple[jnp.ndarray, Array],
        self.col_embedder(
            cell_embeddings,
            train_size=train_size,
            return_repr=True,
        ),
    )
    embeddings, cache_col1 = res


    # Prepend CLS tokens to embeddings before row interactor
    B1, T1, _, _ = embeddings.shape
    cls_tokens_expanded = jnp.broadcast_to(
        self.cls_tokens[...], (B1, T1, self.row_num_cls, self.embed_dim)
    )
    embeddings = jnp.concatenate([cls_tokens_expanded, embeddings], axis=-2)

    embeddings = self.row_interactor(
        embeddings, d=d
    )

    res2 = typing.cast(
        Tuple[jnp.ndarray, Array],
        self.col_embedder_2(
            embeddings,
            train_size=train_size,
            return_repr=True,
        ),
    )
    embeddings, cache_col2 = res2

    representations = self.row_interactor_2(
        embeddings, d=d
    )

    # Stage 3: ICL
    logits_icl, cache_icl = self.icl_predictor(
        representations,
        y,
        train_size=train_size,
        return_cache=True,
    )

    cache = {
        'col1': cache_col1,
        'col2': cache_col2,
        'icl': cache_icl,
    }
    # Unpad to the original 'T'.
    return logits_icl[:, :T, :], cache

  @jt.typed
  def decode(
      self,
      X: jt.Float[jax.Array | np.ndarray, 'B T H'],
      cache: Dict[str, Any],
      d: Optional[jt.Int[jax.Array | np.ndarray, 'B']] = None,
      cat_mask: Optional[jt.Bool[jax.Array | np.ndarray, 'B H']] = None,
      softmax_temperature: float = 0.9,
      num_classes: Optional[int] = None,
  ) -> (jt.Float[jax.Array | np.ndarray, 'B T K'] | jt.Float[jax.Array | np.ndarray, 'B T E']):
    """Generates predictions for test data using cached KVs.

    Args:
      X: Input feature tensor.
      cache: KV cache from prefill.
      d: Number of features per dataset.
      cat_mask: Boolean mask indicating categorical features.
      softmax_temperature: Temperature for softmax.
      num_classes: Number of classes in the dataset.

    Returns:
      Logits for the test data.
    """
    X = jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)

    B, T, _ = X.shape

    # Pad T to be a multiple of 128
    # TODO: Do this only when use_flash_attention is True.
    block_size = 128
    pad_len = ((T - 1) // block_size + 1) * block_size - T
    paddings_X = [(0, 0), (0, pad_len), (0, 0)]
    X = jnp.pad(X, paddings_X, constant_values=-100.0)

    # y is not used for features in decode (test rows).
    y = jnp.full((B, X.shape[1]), -100.0, dtype=self.dtype)

    # Stage 0: Cell-wise embedding for test rows
    cell_embeddings = self.cell_embedder(
        X, y,
        # train_size is 0 for the current batch (all are test rows relative to the cache)
        train_size=jnp.zeros((B,),
                             dtype=jnp.int32),
        d=d, cat_mask=cat_mask
    )


    # Stage 1: Column-wise embedding with cache
    embeddings = typing.cast(
        Array,
        self.col_embedder(
            cell_embeddings,
            train_size=None,
            cached_repr=cache['col1'],
        ),
    )


    # Prepend CLS tokens
    B1, T1, _, _ = embeddings.shape
    cls_tokens_expanded = jnp.broadcast_to(
        self.cls_tokens[...], (B1, T1, self.row_num_cls, self.embed_dim)
    )
    embeddings = jnp.concatenate([cls_tokens_expanded, embeddings], axis=-2)

    embeddings = self.row_interactor(
        embeddings, d=d
    )

    embeddings = typing.cast(
        Array,
        self.col_embedder_2(
            embeddings,
            train_size=None,
            cached_repr=cache['col2'],
        ),
    )

    # Stage 2: Row-wise interaction
    representations = self.row_interactor_2(
        embeddings, d=d
    )

    # Stage 3: ICL with cache
    out = self.icl_predictor(
        representations,
        y,
        train_size=None,
        cache=cache['icl'],
    )

    # Unpad
    out = out[:, :T, ...]
    return out

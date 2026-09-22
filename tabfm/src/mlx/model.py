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

"""MLX port of the TabFM architecture (forward path).

Faithful port of tabfm.src.pytorch.model: module/param names are identical so
weight conversion from the PyTorch checkpoint is mechanical (see convert.py)
and numerical parity can be verified layer by layer (see model_test.py).

Differences from the PyTorch version (all numerically exact):
  - No activation-chunking paths (row/col/ffn chunk sizes stay None). Chunking
    in the PyTorch version is exact; MLX's lazy evaluation + unified memory
    make it unnecessary for the same task sizes in v1.
  - prefill()/decode() do not pad the sequence to a multiple of 128. That
    padding exists for TPU sharding in JAX; on MLX padded rows are pure
    waste (every stage is per-position, per-row independent, or masked),
    so it is skipped. Values match the padded PyTorch path (gated by
    model_test.py's cross-backend checks).
  - RMSNorm uses the fused mx.fast.rms_norm kernel rather than a manual
    float32 chain. Exact in float32; in bfloat16 it is as close to the
    float32 reference as the manual version (see RMSNorm).
  - No `.to(device)` / `.eval()` concepts (unified memory, no train mode).
  - The KV cache defaults to full precision; int8 quantization helpers are
    ported but opt-in (see ICLearningCache.quantize()).

Requires: mlx (Apple silicon), numpy. No PyTorch/JAX dependency.
"""

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _silu(x):
  return x * mx.sigmoid(x)


def _gelu_tanh(x):
  # jax.nn.gelu defaults to the tanh approximation -> match it.
  return nn.gelu_approx(x)


def _softplus(x):
  # Stable softplus in float32 (mlx has no mx.softplus).
  return mx.logaddexp(mx.zeros(()), x)


def get_activation(name):
  return {
      "relu": lambda x: mx.maximum(x, 0),
      "gelu": _gelu_tanh,
      "silu": _silu,
  }[name]


class RMSNorm(nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = mx.ones((dim,))
    self.eps = eps

  def __call__(self, x):
    # Fused kernel: it reduces in float32 internally, so float32 inputs match
    # the JAX/Flax reference (which keeps x*rsqrt in float32) to 3e-07. In
    # bfloat16 it rounds x*rsqrt before applying the weight, unlike the manual
    # float32 chain, but that is immaterial: against the float32 reference on
    # the real checkpoint the two agree to 4.5e-03 and 4.4e-03 respectively,
    # with identical predicted classes. The norms are ~23% of inference time
    # and this removes essentially all of it (see real_checkpoint_parity_test
    # for the float32 gate).
    return mx.fast.rms_norm(x, self.weight, self.eps)


def rope_interleaved(x, base):
  """Interleaved RoPE over the T axis of [B, T, N, Dh] (lucidrains convention)."""
  dh, t = x.shape[-1], x.shape[1]
  inv = 1.0 / (base ** (mx.arange(0, dh, 2).astype(mx.float32) / dh))
  f = mx.arange(t).astype(mx.float32)[:, None] * inv[None, :]
  cos = mx.repeat(mx.cos(f), 2, axis=-1)[None, :, None, :].astype(x.dtype)
  sin = mx.repeat(mx.sin(f), 2, axis=-1)[None, :, None, :].astype(x.dtype)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
  return x * cos + rot * sin


class RoPE(nn.Module):
  """One RoPE per Encoder, holding the inverse-frequency buffer loaded FROM the
  checkpoint (JAX stores `rope.freqs`, computed in bf16 at train time --
  recomputing it in fp32 differs by ~1e-3 and that error grows with sequence
  length)."""

  def __init__(self, dim, base):
    super().__init__()
    inv = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    self.freqs = inv  # init = formula; overwritten on load

  def rotate(self, x):  # x: [B, T, N, Dh], rotate over the T axis
    t = x.shape[1]
    f = (
        mx.arange(t).astype(mx.float32)[:, None]
        * self.freqs.astype(mx.float32)[None, :]
    )
    cos = mx.repeat(mx.cos(f), 2, axis=-1)[None, :, None, :].astype(x.dtype)
    sin = mx.repeat(mx.sin(f), 2, axis=-1)[None, :, None, :].astype(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos + rot * sin


class MultiheadAttention(nn.Module):

  def __init__(self, d_model, nhead, rope_base=None):
    super().__init__()
    self.nhead, self.hd = nhead, d_model // nhead
    self.rope_base = rope_base  # None => no RoPE
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)
    self.query_ln = RMSNorm(self.hd)
    self.key_ln = RMSNorm(self.hd)
    self.per_dim_scale = mx.zeros((self.hd,))

  def __call__(
      self,
      query,
      key,
      value,
      attn_mask=None,
      rope=None,
      cached_kv=None,
      return_kv=False,
  ):
    """Computes multi-head attention, optionally with a K/V cache.

    At most one of cached_kv, return_kv is set. cached_kv is a (k, v) tuple of
    already-projected (and, where used, rotated and key-normalized) tensors of
    shape [B, T_src, N, D] from a prior call; key and value are then None and
    their projections are skipped. return_kv returns the freshly computed
    (k, v) in that layout for a later call to reuse.
    """
    b, tq, d = query.shape
    q = self.q_proj(query).reshape(b, tq, self.nhead, self.hd)

    if cached_kv is not None:
      assert (
          key is None and value is None
      ), "key/value must be None when cached_kv is provided."
      cached_k, cached_v = cached_kv
      # Cached K/V may be int8-quantized; dequantize to compute dtype before use.
      k = (
          cached_k.dequantize(q.dtype)
          if isinstance(cached_k, QuantizedTensor)
          else cached_k
      )
      v = (
          cached_v.dequantize(q.dtype)
          if isinstance(cached_v, QuantizedTensor)
          else cached_v
      )
    else:
      assert (
          key is not None and value is not None
      ), "key/value must not be None when cached_kv is absent."
      k = self.k_proj(key).reshape(b, key.shape[1], self.nhead, self.hd)
      v = self.v_proj(value).reshape(b, value.shape[1], self.nhead, self.hd)

    if self.rope_base is not None:
      # Cached K is already post-RoPE, so only rotate freshly-computed K.
      q = (
          rope.rotate(q)
          if rope is not None
          else rope_interleaved(q, self.rope_base)
      )
      if cached_kv is None:
        k = (
            rope.rotate(k)
            if rope is not None
            else rope_interleaved(k, self.rope_base)
        )

    q = self.query_ln(q)
    if cached_kv is None:
      k = self.key_ln(k)
    # per-dim scale in float32 (softplus), then cast to compute dtype -- matches JAX PerDimScale.
    scale = (
        1.442695041
        / math.sqrt(self.hd)
        * _softplus(self.per_dim_scale.astype(mx.float32))
    )
    q = q * scale.astype(q.dtype)

    new_k, new_v = k, v  # cache format: [B, T_src, N, D], pre-transpose.

    q = mx.transpose(q, (0, 2, 1, 3))
    k = mx.transpose(k, (0, 2, 1, 3))
    v = mx.transpose(v, (0, 2, 1, 3))  # [B,N,T,D]
    # bf16 SDPA (the fused kernel does the softmax in float32 internally).
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=attn_mask)
    out = self.out_proj(mx.transpose(o, (0, 2, 1, 3)).reshape(b, tq, d))
    if return_kv:
      return out, (new_k, new_v)
    return out


class MultiheadAttentionBlock(nn.Module):

  def __init__(
      self, d_model, nhead, dim_ff, activation="swiglu", rope_base=None
  ):
    super().__init__()
    self.attn = MultiheadAttention(d_model, nhead, rope_base)
    self.pre_attn_ln = RMSNorm(d_model)
    self.post_attn_ln = RMSNorm(d_model)
    self.pre_ff_ln = RMSNorm(d_model)
    self.post_ff_ln = RMSNorm(d_model)
    self.swiglu = activation == "swiglu"
    self.linear1 = nn.Linear(d_model, dim_ff)
    if self.swiglu:
      self.linear1_gate = nn.Linear(d_model, dim_ff)
      self.act = _silu
    else:
      self.act = get_activation(activation)
    self.linear2 = nn.Linear(dim_ff, d_model)
    self.ffn_chunk_size = None  # kept for API parity; chunking not needed in v1

  def _ff(self, x):
    xn = self.pre_ff_ln(x)
    if self.swiglu:
      x = self.act(self.linear1_gate(xn)) * self.linear1(xn)
    else:
      x = self.act(self.linear1(xn))
    return self.post_ff_ln(self.linear2(x))

  def __call__(
      self,
      q,
      k=None,
      v=None,
      attn_mask=None,
      rope=None,
      cached_kv=None,
      return_kv=False,
  ):
    q_n = self.pre_attn_ln(q)
    if cached_kv is not None:
      assert (
          k is None and v is None
      ), "k/v must be None when cached_kv is provided."
      k_n, v_n = None, None
    else:
      k = q if k is None else k
      v = q if v is None else v
      k_n = self.pre_attn_ln(k)
      v_n = self.pre_attn_ln(v)
    attn_res = self.attn(
        q_n,
        k_n,
        v_n,
        attn_mask,
        rope=rope,
        cached_kv=cached_kv,
        return_kv=return_kv,
    )
    if return_kv:
      attn_out, new_kv = attn_res
    else:
      attn_out = attn_res
    a = self.post_attn_ln(attn_out)
    x = q + a
    x = x + self._ff(x)
    if return_kv:
      return x, new_kv
    return x


class InducedSelfAttentionBlock(nn.Module):

  def __init__(self, d_model, nhead, dim_ff, num_inds, activation="swiglu"):
    super().__init__()
    self.ind_vectors = mx.zeros((num_inds, d_model))
    self.mab1 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)
    self.mab2 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)

  def __call__(
      self, src, attn_mask=None, cached_hidden=None, return_hidden=False
  ):
    """Applies induced self-attention, optionally reusing a cached hidden.

    If cached_hidden (the mab1 output from a prior call) is given, mab1 is
    skipped and only mab2 runs; return_hidden returns the freshly computed
    hidden for a later call to reuse.
    """
    if cached_hidden is not None:
      hidden = cached_hidden
    else:
      ind = mx.broadcast_to(
          self.ind_vectors, (src.shape[0],) + self.ind_vectors.shape
      )
      hidden = self.mab1(ind, src, src, attn_mask=attn_mask)
    out = self.mab2(src, hidden, hidden)
    if return_hidden:
      return out, hidden
    return out


class Encoder(nn.Module):

  def __init__(
      self,
      num_blocks,
      d_model,
      nhead,
      dim_ff,
      activation="swiglu",
      rope_base=100000.0,
  ):
    super().__init__()
    # One RoPE per Encoder (mirrors JAX `tf_row.rope.freqs`), shared by all blocks.
    self.rope = (
        RoPE(d_model // nhead, rope_base) if rope_base is not None else None
    )
    self.blocks = [
        MultiheadAttentionBlock(d_model, nhead, dim_ff, activation, rope_base)
        for _ in range(num_blocks)
    ]

  def __call__(self, x, attn_mask=None, cached_kv=None, return_kv=False):
    """Runs the stacked attention blocks, optionally with a per-block K/V cache.

    cached_kv, if given, is a per-block list of (k, v) that each block uses in
    place of its k/v projections; return_kv collects the freshly computed
    per-block (k, v). At most one of the two is set.
    """
    if cached_kv is not None:
      assert not return_kv, "Cannot both use cached_kv and return_kv."
      for blk, kv in zip(self.blocks, cached_kv):
        x = blk(x, attn_mask=attn_mask, rope=self.rope, cached_kv=kv)
      return x
    if return_kv:
      kvs = []
      for blk in self.blocks:
        x, kv = blk(x, attn_mask=attn_mask, rope=self.rope, return_kv=True)
        kvs.append(kv)
      return x, kvs
    for blk in self.blocks:
      x = blk(x, attn_mask=attn_mask, rope=self.rope)
    return x


class SetTransformer(nn.Module):

  def __init__(
      self, num_blocks, d_model, nhead, dim_ff, num_inds, activation="swiglu"
  ):
    super().__init__()
    self.blocks = [
        InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, activation)
        for _ in range(num_blocks)
    ]

  def __call__(
      self, src, attn_mask=None, cached_hidden=None, return_hidden=False
  ):
    """Runs the stacked induced-attention blocks.

    cached_hidden, if given, is a per-block list of cached mab1 outputs (see
    InducedSelfAttentionBlock.forward()); return_hidden=True collects the
    freshly computed per-block hidden reprs for later reuse.
    """
    if cached_hidden is not None:
      assert (
          not return_hidden
      ), "Cannot both use cached_hidden and return_hidden."
      for blk, h in zip(self.blocks, cached_hidden):
        src = blk(src, cached_hidden=h)
      return src
    if return_hidden:
      hiddens = []
      for blk in self.blocks:
        src, h = blk(src, attn_mask=attn_mask, return_hidden=True)
        hiddens.append(h)
      return src, hiddens
    for blk in self.blocks:
      src = blk(src, attn_mask=attn_mask)
    return src


class MLP(nn.Module):

  def __init__(
      self, in_dim, hidden_dims: List[int], out_dim, activation="gelu"
  ):
    super().__init__()
    self.act = get_activation(activation)
    dims = [in_dim] + list(hidden_dims)
    self.layers = []  # only Linears; activation applied between
    for i in range(len(hidden_dims)):
      self.layers.append(nn.Linear(dims[i], dims[i + 1]))
    self.layers.append(nn.Linear(dims[-1], out_dim))

  def __call__(self, x):
    for i, lin in enumerate(self.layers):
      x = lin(x)
      if i < len(self.layers) - 1:
        x = self.act(x)
    return x


class OneHotAndLinear(nn.Module):

  def __init__(self, num_classes, embed_dim):
    super().__init__()
    self.num_classes = num_classes
    self.projection = nn.Linear(num_classes, embed_dim)

  def __call__(self, y):  # y: [B, T] int
    y_long = y.astype(mx.int32)
    nc = self.num_classes
    y_mapped = mx.where((y_long >= 0) & (y_long < nc), y_long, nc)
    oh = (mx.arange(nc + 1) == y_mapped[..., None]).astype(
        self.projection.weight.dtype
    )
    oh_sliced = oh[..., :nc]
    return self.projection(oh_sliced)


class CellEmbedder(nn.Module):

  def __init__(
      self,
      embed_dim,
      max_classes,
      feature_group_size=3,
      num_freq=32,
      is_classifier=True,
  ):
    super().__init__()
    self.embed_dim = embed_dim
    self.fgs = feature_group_size
    self.is_classifier = is_classifier
    in_dim = feature_group_size
    self.fourier_frequencies = mx.zeros((in_dim, num_freq))
    self.fourier_frequencies_cat = mx.zeros((in_dim, num_freq))
    self.in_linear = nn.Linear(num_freq * 2, embed_dim)
    self.in_linear_cat = nn.Linear(num_freq * 2, embed_dim)
    if is_classifier:  # classification: embedding lookup over class ids
      self.y_embedder_lookup = nn.Embedding(max_classes, embed_dim)
    else:  # regression: MLP over the scalar target (y_col_embedder_encoder_nhid=6)
      self.y_embedder_lookup = MLP(1, [6], embed_dim, activation="gelu")
    self.row_chunk_size = None  # kept for API parity; chunking not needed in v1

  def _group(self, x, d=None):  # x: [B,T,H] -> [B,T,H,G]
    h = x.shape[-1]
    idxs = mx.arange(h)
    stacked = []
    if d is not None:
      # Per-batch wrap-around over each member's ACTIVE feature count d (not the
      # padded width h). Mirrors the JAX `% d_safe` path so zero-padded slots are
      # filled with wrapped real features rather than mixing padding into groups.
      d_safe = mx.clip(d.astype(mx.int32), 1, 2**31 - 1)  # [B]
      for i in range(self.fgs):
        offset = (2**i) - 1
        idx = (idxs[None, :] + offset) % d_safe[:, None]  # [B, H]
        idx = mx.broadcast_to(
            idx[:, None, :], (x.shape[0], x.shape[1], h)
        )  # [B,T,H]
        stacked.append(mx.take_along_axis(x, idx, axis=-1))
    else:
      for i in range(self.fgs):
        offset = (2**i) - 1
        stacked.append(mx.take(x, (idxs + offset) % h, axis=-1))
    return mx.stack(stacked, axis=-1)

  def _cell(
      self, x, cat_mask, d=None
  ):  # [B,t,H] -> [B,t,HC,E] (Fourier expansion + sum over G)
    g = mx.expand_dims(self._group(x, d=d), -1).astype(
        mx.float32
    )  # float32 Fourier
    dt = x.dtype
    ff = self.fourier_frequencies.astype(mx.float32)
    ffc = self.fourier_frequencies_cat.astype(mx.float32)
    num_out = self.in_linear(
        mx.concatenate([mx.sin(g * ff), mx.cos(g * ff)], axis=-1).astype(dt)
    )
    if cat_mask is not None:
      cat_out = self.in_linear_cat(
          mx.concatenate([mx.sin(g * ffc), mx.cos(g * ffc)], axis=-1).astype(dt)
      )
      cm = mx.expand_dims(
          self._group(mx.expand_dims(cat_mask, 1).astype(mx.float32), d=d), -1
      ).astype(mx.bool_)
      return mx.where(cm, cat_out, num_out).sum(-2)
    return num_out.sum(-2)

  def __call__(self, x, y, train_size, cat_mask=None, d=None):
    cell = self._cell(x, cat_mask, d=d)
    if self.is_classifier:
      y_clean = mx.clip(
          y.astype(mx.int32), 0, self.y_embedder_lookup.weight.shape[0] - 1
      )
      y_emb = self.y_embedder_lookup(y_clean)  # [B,T,E]
    else:
      y_emb = self.y_embedder_lookup(
          mx.expand_dims(y, -1).astype(cell.dtype)
      )  # scalar -> [B,T,E]
    t = x.shape[1]
    tm = (mx.arange(t)[None, :] < train_size[:, None])[..., None, None]
    out = mx.where(tm, cell + y_emb[:, :, None, :], cell)
    if d is not None:
      # Zero the padded feature columns (cols >= d): the % d wrap above fills them
      # with real features for valid indexing, but they must not enter attention.
      hc = out.shape[2]
      colmask = (mx.arange(hc)[None, :] < d[:, None])[
          :, None, :, None
      ]  # [B,1,HC,1]
      out = mx.where(colmask, out, mx.zeros_like(out))
    return out


class ColEmbedding(nn.Module):

  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_inds):
    super().__init__()
    self.tf_col = SetTransformer(num_blocks, d_model, nhead, dim_ff, num_inds)
    self.out_w = nn.Linear(d_model, d_model)
    self.ln_w = RMSNorm(d_model)
    self.col_chunk_size = None  # kept for API parity; chunking not needed in v1

  def _stage(self, src, mask=None, cached_hidden=None, return_hidden=False):
    out = self.tf_col(
        src,
        attn_mask=mask,
        cached_hidden=cached_hidden,
        return_hidden=return_hidden,
    )
    if return_hidden:
      out, hidden = out
      return self.ln_w(self.out_w(out)), hidden
    return self.ln_w(self.out_w(out))

  def __call__(self, x, train_size, *, cached_repr=None, return_repr=False):
    """Transform input table into column-wise embeddings.

    train_size is None exactly when cached_repr is given (decode): cached_repr
    supplies the induced-point hidden, so mab1 and its mask are skipped.
    cached_repr and return_repr are mutually exclusive.
    """
    assert not (
        cached_repr is not None and return_repr
    ), "Cannot have both cached_repr not None and return_repr True."
    assert (cached_repr is not None) == (
        train_size is None
    ), "train_size must be None iff cached_repr is given."
    b, t, hc, e = x.shape
    src = x.transpose(0, 2, 1, 3).reshape(b * hc, t, e)  # [B*HC, T, E]

    if cached_repr is not None:  # Decode: reuse cached induced-point hidden.
      out = self._stage(src, None, cached_hidden=cached_repr)
      return out.reshape(b, hc, t, e).transpose(0, 2, 1, 3)

    ts = mx.repeat(train_size, hc)  # [B*HC]
    mask = (mx.arange(t)[None, :] < ts[:, None])[:, None, None, :]

    if return_repr:  # Prefill: also return the induced-point hidden per block.
      out, hidden = self._stage(src, mask, return_hidden=True)
      out = out.reshape(b, hc, t, e).transpose(0, 2, 1, 3)
      return out, hidden

    out = self._stage(src, mask)
    return out.reshape(b, hc, t, e).transpose(0, 2, 1, 3)


class RowInteraction(nn.Module):

  def __init__(
      self,
      d_model,
      num_blocks,
      nhead,
      dim_ff,
      num_cls,
      rope_base=100000.0,
      output_full=True,
  ):
    super().__init__()
    self.tf_row = Encoder(
        num_blocks, d_model, nhead, dim_ff, rope_base=rope_base
    )
    self.out_ln = RMSNorm(d_model)
    self.num_cls = num_cls
    self.output_full = output_full
    self.row_chunk_size = None  # kept for API parity; chunking not needed in v1

  def _stage(self, src, mask=None):
    out = self.tf_row(src, attn_mask=mask)
    return self.out_ln(out if self.output_full else out[:, : self.num_cls, :])

  def __call__(self, x, d=None):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.reshape(b * t, hc, e)
    # Mask cross-column attention to the valid columns (CLS + d real features);
    # padded columns (>= d + num_cls) must not be attended to. Matches JAX.
    mask = None
    if d is not None:
      d_padded = d.astype(mx.int32) + self.num_cls  # [B]
      valid = mx.arange(hc)[None, :] < d_padded[:, None]  # [B, HC]
      mask = mx.repeat(valid, t, axis=0)[:, None, None, :]  # [B*T, 1, 1, HC]
    out = self._stage(src, mask)
    if self.output_full:
      return out.reshape(b, t, hc, e)
    return out.reshape(b, t, -1)


@dataclasses.dataclass
class QuantizedTensor:
  """Per-tensor symmetric integer quantization of a cached K or V tensor.

  data holds the quantized codes; scale is the per-tensor absmax / max_val
  factor, so the float value is data.astype(dtype) * scale.

  scale is a scalar as produced by _quantize_tensor(). Concatenating caches
  that were quantized separately (cross-member batching, see
  _concat_caches_mlx) instead stores one scale per member, shaped
  [B, 1, 1, 1] so it still broadcasts against data's [B, T, N, D]; that keeps
  the merged cache int8 instead of materializing it in full precision.
  """

  data: mx.array
  scale: mx.array

  def dequantize(self, dtype: mx.Dtype) -> mx.array:
    """Dequantizes back to a floating-point array of the given dtype."""
    return self.data.astype(dtype) * self.scale.astype(dtype)


# Per dtype: (lo, hi, max_val) clamp range, one code below the dtype's full
# range so -max and +max map symmetrically and dequantization cannot exceed
# the original absmax in magnitude.
_QUANTIZATION_RANGES: Dict[mx.Dtype, Tuple[int, int, int]] = {
    mx.int8: (-127, 127, 127),
}


def _quantize_tensor(t: mx.array, dtype: mx.Dtype = mx.int8) -> QuantizedTensor:
  """Per-tensor symmetric integer quantization to dtype."""
  if dtype not in _QUANTIZATION_RANGES:
    raise ValueError(
        f"Unsupported quantization dtype {dtype}; supported: "
        f"{list(_QUANTIZATION_RANGES.keys())}"
    )
  lo, hi, max_val = _QUANTIZATION_RANGES[dtype]
  absmax = mx.max(mx.abs(t))
  scale = absmax / max_val
  # Avoid division by zero for all-zero tensors. mx.finfo exposes the
  # smallest positive normal as `smallest_normal` (torch calls it `tiny`);
  # `.min` is the most negative value and would make this clamp a no-op.
  scale = mx.clip(scale, mx.finfo(mx.float32).smallest_normal, None)
  data = mx.clip(mx.round(t / scale), lo, hi).astype(dtype)
  return QuantizedTensor(data=data, scale=scale)


def move_cache_to_device(cache):
  """Recursively walks a TabFM.prefill() cache dict (MLX: unified memory, so
  this is an identity traversal kept for API parity with the PyTorch backend).
  """
  if isinstance(cache, mx.array):
    return cache
  if isinstance(cache, dict):
    return {k: move_cache_to_device(v) for k, v in cache.items()}
  if isinstance(cache, list):
    return [move_cache_to_device(v) for v in cache]
  if isinstance(cache, tuple):
    return tuple(move_cache_to_device(v) for v in cache)
  if dataclasses.is_dataclass(cache):
    kwargs = {
        f.name: move_cache_to_device(getattr(cache, f.name))
        for f in dataclasses.fields(cache)
    }
    return type(cache)(**kwargs)
  return cache


@dataclasses.dataclass
class ICLearningCache:
  """Per-block ICL K/V cache produced at prefill and reused at decode.

  layer_caches is a per-block list of (k, v) of shape [B, T_prefill, N, D],
  each a QuantizedTensor after quantize(). prefill_train_size is the [B] count
  of valid training rows used to build the decode attention mask; it stays
  full precision after quantize().
  """

  layer_caches: List[Tuple[Any, Any]]
  prefill_train_size: mx.array

  @property
  def prefill_seq_len(self) -> int:
    """Sequence length of the prefill cache, derived from tensor shape."""
    k, _ = self.layer_caches[0]
    data = k.data if isinstance(k, QuantizedTensor) else k
    return data.shape[1]

  def quantize(self, dtype: mx.Dtype = mx.int8) -> "ICLearningCache":
    """Returns a copy with the per-block ICL K/V quantized to dtype.

    Only the attention K/V is quantized; prefill_train_size stays full
    precision.
    """
    quantized = [
        (_quantize_tensor(k, dtype), _quantize_tensor(v, dtype))
        for k, v in self.layer_caches
    ]
    return ICLearningCache(
        layer_caches=quantized, prefill_train_size=self.prefill_train_size
    )


class ICLearning(nn.Module):

  def __init__(
      self,
      d_model,
      num_blocks,
      nhead,
      max_classes,
      dim_ff,
      decoder_hidden,
      is_classifier=True,
  ):
    super().__init__()
    self.tf_icl = Encoder(
        num_blocks, d_model, nhead, dim_ff, rope_base=None
    )  # ICL has no RoPE
    self.ln = RMSNorm(d_model)
    self.is_classifier = is_classifier
    if is_classifier:  # one-hot y-encode; decode to per-class logits
      self.y_encoder = OneHotAndLinear(max_classes, d_model)
      self.decoder = MLP(d_model, [decoder_hidden], max_classes)
    else:  # MLP y-encode the scalar target; decode to a single value
      self.y_encoder = MLP(1, [decoder_hidden], d_model)
      self.decoder = MLP(d_model, [decoder_hidden], 1)

  def __call__(
      self,
      reps,
      y,
      train_size,
      *,
      cache: Optional[ICLearningCache] = None,
      return_cache: bool = False,
  ):
    """Forward pass for ICLearning. reps: [B, T, E] row representations.

    train_size is None exactly when cache is given (decode): the attention mask
    is then derived from the cached prefill's train size and sequence length
    rather than this call's, and y is unused. return_cache (prefill only) also
    returns the ICLearningCache for this call alongside the decoded output.
    """
    assert (cache is not None) == (
        train_size is None
    ), "train_size must be None iff cache is given."
    b, t, _ = reps.shape

    if cache is not None:  # Decode.
      prefill_seq_len = cache.prefill_seq_len
      tm_ctx = (
          mx.arange(prefill_seq_len)[None, :]
          < cache.prefill_train_size[:, None]
      )
      mask = tm_ctx[:, None, None, :]
      out = self.tf_icl(reps, attn_mask=mask, cached_kv=cache.layer_caches)
      return self.decoder(self.ln(out))

    tm = mx.arange(t)[None, :] < train_size[:, None]
    if self.is_classifier:
      y_enc = self.y_encoder(y)
    else:
      y_enc = self.y_encoder(mx.expand_dims(y, -1).astype(reps.dtype))
    r = reps + y_enc * tm[..., None]
    mask = tm[:, None, None, :]
    if return_cache:  # Prefill.
      out, kvs = self.tf_icl(r, attn_mask=mask, return_kv=True)
      new_cache = ICLearningCache(
          layer_caches=kvs, prefill_train_size=train_size
      )
      return self.decoder(self.ln(out)), new_cache
    out = self.tf_icl(r, attn_mask=mask)
    return self.decoder(self.ln(out))


class TabFM(nn.Module):

  def __init__(
      self,
      *,
      embed_dim=8,
      max_classes=10,
      col_num_blocks=2,
      col_nhead=2,
      col_num_inds=4,
      row_num_blocks=2,
      row_nhead=2,
      row_num_cls=2,
      icl_num_blocks=2,
      icl_nhead=2,
      ff_factor=2,
      feature_group_size=3,
      num_freq=32,
      decoder_hidden=None,
      is_classifier=True,
  ):
    super().__init__()
    self.max_classes = max_classes
    self.is_classifier = is_classifier
    ff = embed_dim * ff_factor
    icl_dim = embed_dim * row_num_cls
    self.cell_embedder = CellEmbedder(
        embed_dim, max_classes, feature_group_size, num_freq, is_classifier
    )
    self.col_embedder = ColEmbedding(
        embed_dim, col_num_blocks, col_nhead, ff, col_num_inds
    )
    self.col_embedder_2 = ColEmbedding(
        embed_dim, col_num_blocks, col_nhead, ff, col_num_inds
    )
    self.row_interactor = RowInteraction(
        embed_dim, row_num_blocks, row_nhead, ff, row_num_cls, output_full=True
    )
    self.row_interactor_2 = RowInteraction(
        embed_dim, row_num_blocks, row_nhead, ff, row_num_cls, output_full=False
    )
    self.cls_tokens = mx.zeros((row_num_cls, embed_dim))
    self.icl_predictor = ICLearning(
        icl_dim,
        icl_num_blocks,
        icl_nhead,
        max_classes,
        icl_dim * ff_factor,
        decoder_hidden or icl_dim * 2,
        is_classifier,
    )

  def __call__(self, x, y, train_size, cat_mask=None, d=None):
    # Mirror the JAX model's entry: replace NaN with the -100 sentinel and cast
    # to the compute dtype (JAX: `jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)`).
    # NaN is already imputed in the shared preprocessing, so nan_to_num is a
    # no-op in the normal flow, but it keeps the model robust + JAX-faithful.
    x = mx.nan_to_num(x, nan=-100.0).astype(self.cls_tokens.dtype)
    emb = self.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb = self.col_embedder(emb, train_size)
    b, t, _, e = emb.shape
    cls = mx.broadcast_to(self.cls_tokens, (b, t) + self.cls_tokens.shape)
    emb = mx.concatenate([cls, emb], axis=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, train_size)
    reps = self.row_interactor_2(emb, d=d)
    return self.icl_predictor(reps, y, train_size)

  def prefill(self, x, y, cat_mask=None, d=None):
    """Encodes context (training) rows once and returns (logits, cache).

    x is [B, T, H] context rows and y is [B, T] context labels; cat_mask is an
    optional [B] categorical-feature mask and d an optional [B] active-
    feature count. Unlike the JAX/PyTorch prefill, the sequence is NOT padded
    to a multiple of 128: padding there exists for TPU sharding, and padded
    rows are pure waste on MLX -- every stage is either per-position
    (cell embedder), per-row independent (row interactors batch over B*T),
    or masked (column/ICL attention exclude non-train rows via train_size),
    so padding cannot change real-row values. train_size is derived from the
    non-sentinel y entries; runs the full pipeline while collecting the
    col-embedder induced-point reprs and the ICL encoder per-layer K/V.
    cache is a dict with keys 'col1', 'col2' (per-block induced-point reprs)
    and 'icl' (an ICLearningCache).
    """
    x = mx.nan_to_num(x, nan=-100.0).astype(self.cls_tokens.dtype)
    y = y.astype(self.cls_tokens.dtype)

    t_orig = x.shape[1]

    # All data is training data; train_size counts non-sentinel y entries so
    # externally-padded rows are excluded.
    is_valid = y != -100.0
    train_size = mx.sum(is_valid, axis=-1).astype(mx.int32)

    cell = self.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb, cache_col1 = self.col_embedder(cell, train_size, return_repr=True)
    b1, t1, _, e = emb.shape
    cls = mx.broadcast_to(self.cls_tokens, (b1, t1) + self.cls_tokens.shape)
    emb = mx.concatenate([cls, emb], axis=2)
    emb = self.row_interactor(emb, d=d)
    emb, cache_col2 = self.col_embedder_2(emb, train_size, return_repr=True)
    reps = self.row_interactor_2(emb, d=d)
    logits, cache_icl = self.icl_predictor(
        reps, y, train_size, return_cache=True
    )

    cache = {"col1": cache_col1, "col2": cache_col2, "icl": cache_icl}
    return logits[:, :t_orig, :], cache

  def decode(self, x, cache, cat_mask=None, d=None):
    """Generates predictions for test rows using a cache from prefill.

    x is [B, T, H] test rows and cache is the dict returned by prefill;
    cat_mask and d are as in prefill. Like prefill, the sequence is NOT
    padded to a multiple of 128 (see prefill): test rows only ever appear as
    attention queries against the cached train keys, so padding rows would
    produce outputs that are sliced off anyway. Re-runs the row-independent
    cell embedder and row interactors on the test rows, reuses the cached
    col-embedder reprs and ICL per-layer K/V instead of recomputing them
    from the context. Returns [B, T, K_or_1] logits.
    """
    x = mx.nan_to_num(x, nan=-100.0).astype(self.cls_tokens.dtype)
    b, t_orig, _ = x.shape

    # y carries no information for test rows in decode (unlabeled); use the
    # -100 sentinel throughout, matching JAX.
    y = mx.full((b, x.shape[1]), -100.0, dtype=self.cls_tokens.dtype)
    train_size_zero = mx.zeros((b,), dtype=mx.int32)

    cell = self.cell_embedder(x, y, train_size_zero, cat_mask, d=d)
    emb = self.col_embedder(cell, None, cached_repr=cache["col1"])
    b1, t1, _, e = emb.shape
    cls = mx.broadcast_to(self.cls_tokens, (b1, t1) + self.cls_tokens.shape)
    emb = mx.concatenate([cls, emb], axis=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, None, cached_repr=cache["col2"])
    reps = self.row_interactor_2(emb, d=d)
    out = self.icl_predictor(reps, y, None, cache=cache["icl"])

    return out[:, :t_orig, :]

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

"""Faithful MLX port of the TabFM architecture (forward path only).

Module/param names mirror the PyTorch port (tabfm/src/pytorch/model.py), which
in turn mirrors the JAX model, so the pre-trained PyTorch safetensors
checkpoint loads mechanically: MLX flattens list-of-module children with the
same ``blocks.0.attn.q_proj.weight`` key scheme as ``torch.nn.Module`` and
``mlx.nn.Linear`` stores its weight as ``[out, in]`` like PyTorch, so no
transposes or key remapping are required. Checkpoint buffers (``rope.freqs``,
``fourier_frequencies``) are plain ``mx.array`` attributes, which MLX treats
as loadable parameters.

Numerical-fidelity notes (see the parity test in model_test.py):
- RMSNorm, the Fourier expansion, PerDimScale's softplus and RoPE's phase
  computation all run in float32 with a cast back to the compute dtype,
  matching the JAX/PyTorch implementations.
- Attention uses SDPA at scale=1.0 (the 1/sqrt(d) factor is folded into
  PerDimScale) with q/k RMSNorm applied after RoPE.
- RoPE inverse frequencies are LOADED from the checkpoint, not recomputed.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


def _gelu_tanh(x):
  # jax.nn.gelu defaults to the tanh approximation -> match it exactly
  # (mlx.nn's gelu variants use erf / sigmoid approximations instead).
  return 0.5 * x * (
      1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
  )


def get_activation(name):
  return {"relu": nn.relu, "gelu": _gelu_tanh, "silu": nn.silu}[name]


def _softplus_f32(x):
  # logaddexp(x, 0) == log(1 + exp(x)) computed stably in float32.
  return mx.logaddexp(x.astype(mx.float32), mx.array(0.0, dtype=mx.float32))


class RMSNorm(nn.Module):

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = mx.ones((dim,))
    self.eps = eps

  def __call__(self, x):
    # Normalize entirely in float32 (x * rsqrt * weight), cast back at the
    # end -- matches JAX/Flax and the PyTorch port. Doing the multiply in bf16
    # loses precision and accumulates across the ~36 RMSNorms per stack.
    dt = x.dtype
    xf = x.astype(mx.float32)
    v = mx.mean(xf * xf, axis=-1, keepdims=True)
    out = (xf * mx.rsqrt(v + self.eps)) * self.weight.astype(mx.float32)
    return out.astype(dt)


def _rotate_with_freqs(x, freqs):
  """Interleaved RoPE over the T axis of [B, T, N, Dh] (lucidrains convention)."""
  t = x.shape[1]
  # Phases in float32; cos/sin cast to the compute dtype afterwards (matches
  # the PyTorch port, which upcasts the checkpoint-loaded freqs the same way).
  f = (
      mx.arange(t).astype(mx.float32)[:, None]
      * freqs.astype(mx.float32)[None, :]
  )
  cos = mx.repeat(mx.cos(f), 2, axis=-1)[None, :, None, :].astype(x.dtype)
  sin = mx.repeat(mx.sin(f), 2, axis=-1)[None, :, None, :].astype(x.dtype)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  rot = mx.stack([-x2, x1], axis=-1).reshape(x.shape)
  return x * cos + rot * sin


def rope_interleaved(x, base):
  """RoPE fallback that recomputes the inverse frequencies from ``base``."""
  dh = x.shape[-1]
  inv = 1.0 / (base ** (mx.arange(0, dh, 2).astype(mx.float32) / dh))
  return _rotate_with_freqs(x, inv)


class RoPE(nn.Module):
  """One RoPE per Encoder, holding the inverse-frequency buffer loaded FROM the
  checkpoint (JAX stores `rope.freqs`, computed in bf16 at train time --
  recomputing it in fp32 differs by ~1e-3 and that error grows with sequence
  length)."""

  def __init__(self, dim, base):
    super().__init__()
    # init = formula; overwritten on load
    self.freqs = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

  def rotate(self, x):  # x: [B, T, N, Dh], rotate over the T axis
    return _rotate_with_freqs(x, self.freqs)


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

  def __call__(self, query, key, value, attn_mask=None, rope=None):
    b, tq, d = query.shape
    q = self.q_proj(query).reshape(b, tq, self.nhead, self.hd)
    k = self.k_proj(key).reshape(b, key.shape[1], self.nhead, self.hd)
    v = self.v_proj(value).reshape(b, value.shape[1], self.nhead, self.hd)
    if self.rope_base is not None:
      # Use the Encoder's shared RoPE (checkpoint-loaded freqs) when provided;
      # fall back to recomputing only if absent.
      if rope is not None:
        q, k = rope.rotate(q), rope.rotate(k)
      else:
        q = rope_interleaved(q, self.rope_base)
        k = rope_interleaved(k, self.rope_base)
    q, k = self.query_ln(q), self.key_ln(k)
    # per-dim scale in float32 (softplus), then cast to compute dtype --
    # matches JAX PerDimScale.
    scale = 1.442695041 / math.sqrt(self.hd) * _softplus_f32(self.per_dim_scale)
    q = q * scale.astype(q.dtype)
    q, k, v = (z.transpose(0, 2, 1, 3) for z in (q, k, v))  # [B,N,T,D]
    # Boolean masks (True = attend) are supported natively; SDPA does its
    # softmax in float32 internally.
    o = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0, mask=attn_mask
    )
    return self.out_proj(o.transpose(0, 2, 1, 3).reshape(b, tq, d))


class MultiheadAttentionBlock(nn.Module):

  def __init__(self, d_model, nhead, dim_ff, activation="swiglu",
               rope_base=None):
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
      self.act = nn.silu
    else:
      self.act = get_activation(activation)
    self.linear2 = nn.Linear(dim_ff, d_model)
    self.ffn_chunk_size = None  # set to an int to chunk the FFN over tokens

  def _ff_impl(self, x):
    xn = self.pre_ff_ln(x)
    if self.swiglu:
      x = self.act(self.linear1_gate(xn)) * self.linear1(xn)
    else:
      x = self.act(self.linear1(xn))
    return self.post_ff_ln(self.linear2(x))

  def _ff(self, x):
    # FFN chunking: process tokens in slices so the expanded
    # [tokens, dim_feedforward] activation is never materialized in full.
    if self.ffn_chunk_size is None:
      return self._ff_impl(x)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    parts = [
        self._ff_impl(flat[s : s + self.ffn_chunk_size])
        for s in range(0, flat.shape[0], self.ffn_chunk_size)
    ]
    return mx.concatenate(parts, axis=0).reshape(shape)

  def __call__(self, q, k=None, v=None, attn_mask=None, rope=None):
    k = q if k is None else k
    v = q if v is None else v
    a = self.post_attn_ln(
        self.attn(
            self.pre_attn_ln(q),
            self.pre_attn_ln(k),
            self.pre_attn_ln(v),
            attn_mask,
            rope=rope,
        )
    )
    x = q + a
    return x + self._ff(x)


class InducedSelfAttentionBlock(nn.Module):

  def __init__(self, d_model, nhead, dim_ff, num_inds, activation="swiglu"):
    super().__init__()
    self.ind_vectors = mx.zeros((num_inds, d_model))
    self.mab1 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)
    self.mab2 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)

  def __call__(self, src, attn_mask=None):
    ind = mx.broadcast_to(
        self.ind_vectors[None], (src.shape[0],) + self.ind_vectors.shape
    )
    hidden = self.mab1(ind, src, src, attn_mask=attn_mask)
    return self.mab2(src, hidden, hidden)


class Encoder(nn.Module):

  def __init__(self, num_blocks, d_model, nhead, dim_ff, activation="swiglu",
               rope_base=100000.0):
    super().__init__()
    # One RoPE per Encoder (mirrors JAX `tf_row.rope.freqs`), shared by all
    # blocks.
    self.rope = (
        RoPE(d_model // nhead, rope_base) if rope_base is not None else None
    )
    self.blocks = [
        MultiheadAttentionBlock(d_model, nhead, dim_ff, activation, rope_base)
        for _ in range(num_blocks)
    ]

  def __call__(self, x, attn_mask=None):
    for blk in self.blocks:
      x = blk(x, attn_mask=attn_mask, rope=self.rope)
    return x


class SetTransformer(nn.Module):

  def __init__(self, num_blocks, d_model, nhead, dim_ff, num_inds,
               activation="swiglu"):
    super().__init__()
    self.blocks = [
        InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, activation)
        for _ in range(num_blocks)
    ]

  def __call__(self, src, attn_mask=None):
    for blk in self.blocks:
      src = blk(src, attn_mask=attn_mask)
    return src


class MLP(nn.Module):

  def __init__(self, in_dim, hidden_dims: List[int], out_dim,
               activation="gelu"):
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
    y_int = y.astype(mx.int32)
    # No mx.one_hot; compare against the class range instead. Out-of-range
    # labels (e.g. the -100 padding sentinel) match no column and contribute
    # only the projection bias -- identical to the PyTorch remap-then-slice.
    oh = (y_int[..., None] == mx.arange(self.num_classes)).astype(
        self.projection.weight.dtype
    )
    return self.projection(oh)


class CellEmbedder(nn.Module):

  def __init__(self, embed_dim, max_classes, feature_group_size=3, num_freq=32,
               is_classifier=True):
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
    self.row_chunk_size = None  # chunk the Fourier expansion over rows

  def _group(self, x, d=None):  # x: [B,T,H] -> [B,T,H,G]
    h = x.shape[-1]
    idxs = mx.arange(h)
    stacked = []
    if d is not None:
      # Per-batch wrap-around over each member's ACTIVE feature count d (not
      # the padded width h). Mirrors the JAX `% d_safe` path so zero-padded
      # slots are filled with wrapped real features rather than mixing padding
      # into groups.
      d_safe = mx.maximum(d.astype(mx.int32), 1)  # [B]
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        idx = (idxs[None, :] + offset) % d_safe[:, None]  # [B, H]
        idx = mx.broadcast_to(
            idx[:, None, :], (x.shape[0], x.shape[1], h)
        )  # [B, T, H]
        stacked.append(mx.take_along_axis(x, idx, axis=-1))
    else:
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        stacked.append(mx.take(x, (idxs + offset) % h, axis=-1))
    return mx.stack(stacked, axis=-1)

  def _cell(self, x, cat_mask, d=None):
    # [B,t,H] -> [B,t,HC,E] (Fourier expansion + sum over G).
    # float32 Fourier: args g*freq reach ~30, so sin/cos must run in fp32
    # (matches JAX, whose freq params stay float32). Cast the fourier features
    # back to compute dtype before in_linear.
    g = self._group(x, d=d)[..., None].astype(mx.float32)
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
      cmg = self._group(
          cat_mask[:, None, :].astype(mx.float32), d=d
      ).astype(mx.bool_)[..., None]
      return mx.where(cmg, cat_out, num_out).sum(axis=-2)
    return num_out.sum(axis=-2)

  def __call__(self, x, y, train_size, cat_mask=None, d=None):
    # The Fourier expansion materializes [B,T,HC,G,E]; chunk over rows so that
    # huge intermediate never exists in full (rows are independent here).
    if self.row_chunk_size is None:
      cell = self._cell(x, cat_mask, d=d)
    else:
      parts = [
          self._cell(x[:, s : s + self.row_chunk_size], cat_mask, d=d)
          for s in range(0, x.shape[1], self.row_chunk_size)
      ]
      cell = mx.concatenate(parts, axis=1)
    if self.is_classifier:
      num_embeddings = self.y_embedder_lookup.weight.shape[0]
      y_clean = mx.clip(y.astype(mx.int32), 0, num_embeddings - 1)
      y_emb = self.y_embedder_lookup(y_clean)  # [B,T,E]
    else:
      y_emb = self.y_embedder_lookup(y[..., None].astype(cell.dtype))
    t = x.shape[1]
    tm = (mx.arange(t)[None, :] < train_size[:, None])[..., None, None]
    out = mx.where(tm, cell + y_emb[:, :, None, :], cell)
    if d is not None:
      # Zero the padded feature columns (cols >= d): the % d wrap above fills
      # them with real features for valid indexing, but they must not enter
      # attention.
      hc = out.shape[2]
      colmask = (mx.arange(hc)[None, :] < d[:, None])[:, None, :, None]
      out = mx.where(colmask, out, mx.zeros_like(out))
    return out


class ColEmbedding(nn.Module):

  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_inds):
    super().__init__()
    self.tf_col = SetTransformer(num_blocks, d_model, nhead, dim_ff, num_inds)
    self.out_w = nn.Linear(d_model, d_model)
    self.ln_w = RMSNorm(d_model)
    self.col_chunk_size = None  # chunk the independent column axis (B*HC)

  def _stage(self, src, mask):
    return self.ln_w(self.out_w(self.tf_col(src, attn_mask=mask)))

  def __call__(self, x, train_size):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.transpose(0, 2, 1, 3).reshape(b * hc, t, e)  # [B*HC, T, E]
    ts = mx.repeat(train_size, hc, axis=0)  # [B*HC]
    mask = (mx.arange(t)[None, :] < ts[:, None])[:, None, None, :]
    cc = self.col_chunk_size
    if cc is None or src.shape[0] <= cc:
      out = self._stage(src, mask)
    else:
      out = mx.concatenate(
          [
              self._stage(src[s : s + cc], mask[s : s + cc])
              for s in range(0, src.shape[0], cc)
          ],
          axis=0,
      )
    return out.reshape(b, hc, t, e).transpose(0, 2, 1, 3)


class RowInteraction(nn.Module):

  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_cls,
               rope_base=100000.0, output_full=True):
    super().__init__()
    self.tf_row = Encoder(num_blocks, d_model, nhead, dim_ff,
                          rope_base=rope_base)
    self.out_ln = RMSNorm(d_model)
    self.num_cls = num_cls
    self.output_full = output_full
    self.row_chunk_size = None  # chunk the independent row axis (B*T)

  def _stage(self, src, mask=None):
    out = self.tf_row(src, attn_mask=mask)
    return self.out_ln(out if self.output_full else out[:, : self.num_cls, :])

  def __call__(self, x, d=None):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.reshape(b * t, hc, e)
    # Mask cross-column attention to the valid columns (CLS + d real
    # features); padded columns (>= d + num_cls) must not be attended to.
    # Matches JAX.
    mask = None
    if d is not None:
      d_padded = d.astype(mx.int32) + self.num_cls  # [B]
      valid = mx.arange(hc)[None, :] < d_padded[:, None]  # [B, HC]
      mask = mx.repeat(valid, t, axis=0)[:, None, None, :]  # [B*T, 1, 1, HC]
    rc = self.row_chunk_size
    if rc is None or src.shape[0] <= rc:
      out = self._stage(src, mask)
    else:
      out = mx.concatenate(
          [
              self._stage(
                  src[s : s + rc], None if mask is None else mask[s : s + rc]
              )
              for s in range(0, src.shape[0], rc)
          ],
          axis=0,
      )
    if self.output_full:
      return out.reshape(b, t, hc, e)
    return out.reshape(b, t, -1)


class ICLearning(nn.Module):

  def __init__(self, d_model, num_blocks, nhead, max_classes, dim_ff,
               decoder_hidden, is_classifier=True):
    super().__init__()
    # ICL has no RoPE.
    self.tf_icl = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=None)
    self.ln = RMSNorm(d_model)
    self.is_classifier = is_classifier
    if is_classifier:  # one-hot y-encode; decode to per-class logits
      self.y_encoder = OneHotAndLinear(max_classes, d_model)
      self.decoder = MLP(d_model, [decoder_hidden], max_classes)
    else:  # MLP y-encode the scalar target; decode to a single value
      self.y_encoder = MLP(1, [decoder_hidden], d_model)
      self.decoder = MLP(d_model, [decoder_hidden], 1)

  def __call__(self, reps, y, train_size):  # reps: [B,T,d_model]
    b, t, _ = reps.shape
    tm = mx.arange(t)[None, :] < train_size[:, None]
    if self.is_classifier:
      y_enc = self.y_encoder(y)
    else:
      y_enc = self.y_encoder(y[..., None].astype(reps.dtype))
    r = reps + y_enc * tm[..., None]
    mask = tm[:, None, None, :]
    out = self.tf_icl(r, attn_mask=mask)
    return self.decoder(self.ln(out))


class TabFM(nn.Module):

  def __init__(self, *, embed_dim=8, max_classes=3, col_num_blocks=2,
               col_nhead=2, col_num_inds=4, row_num_blocks=2, row_nhead=2,
               row_num_cls=2, icl_num_blocks=2, icl_nhead=2, ff_factor=2,
               feature_group_size=3, num_freq=32, decoder_hidden=None,
               is_classifier=True):
    super().__init__()
    self.max_classes = max_classes
    self.is_classifier = is_classifier
    ff = embed_dim * ff_factor
    icl_dim = embed_dim * row_num_cls
    self.cell_embedder = CellEmbedder(embed_dim, max_classes,
                                      feature_group_size, num_freq,
                                      is_classifier)
    self.col_embedder = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff,
                                     col_num_inds)
    self.col_embedder_2 = ColEmbedding(embed_dim, col_num_blocks, col_nhead,
                                       ff, col_num_inds)
    self.row_interactor = RowInteraction(embed_dim, row_num_blocks, row_nhead,
                                         ff, row_num_cls, output_full=True)
    self.row_interactor_2 = RowInteraction(embed_dim, row_num_blocks,
                                           row_nhead, ff, row_num_cls,
                                           output_full=False)
    self.cls_tokens = mx.zeros((row_num_cls, embed_dim))
    self.icl_predictor = ICLearning(icl_dim, icl_num_blocks, icl_nhead,
                                    max_classes, icl_dim * ff_factor,
                                    decoder_hidden or icl_dim * 2,
                                    is_classifier)

  def __call__(self, x, y, train_size, cat_mask=None, d=None):
    # Mirror the JAX model's entry: replace NaN with the -100 sentinel and
    # cast to the compute dtype (JAX:
    # `jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)`). NaN is already
    # imputed in the shared preprocessing, so nan_to_num is a no-op in the
    # normal flow, but it keeps the model robust + JAX-faithful.
    x = mx.nan_to_num(x, nan=-100.0).astype(self.cls_tokens.dtype)
    emb = self.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb = self.col_embedder(emb, train_size)
    b, t, _, e = emb.shape
    cls = mx.broadcast_to(
        self.cls_tokens[None, None], (b, t) + self.cls_tokens.shape
    )
    emb = mx.concatenate([cls, emb], axis=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, train_size)
    reps = self.row_interactor_2(emb, d=d)
    return self.icl_predictor(reps, y, train_size)

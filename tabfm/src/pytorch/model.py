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

"""EXPERIMENTAL faithful-ish PyTorch port of the TabFM architecture (fwd path).

Module/param names mirror the JAX model (tabfm/src/model.py) so weight
conversion is mechanical. The transformer math (attention with RoPE +
PerDimScale + q/k RMSNorm + SDPA at scale=1.0, swiglu FFN, full MAB) has been
parity-verified against JAX to ~1e-6 (float32). The embedding/ICL paths mirror
the JAX code but are validated by the end-to-end converter parity test.

NOT yet wired to Orbax weights; see torch_parity_harness.py for the converter
and parity gates.
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


def get_activation(name):
  # jax.nn.gelu defaults to the tanh approximation -> match it.
  return {"relu": F.relu,
          "gelu": lambda x: F.gelu(x, approximate="tanh"),
          "silu": F.silu}[name]


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    # Normalize entirely in float32 (x * rsqrt * weight), cast back at the end --
    # matches JAX/Flax, which keeps x*rsqrt in float32. Doing the multiply in bf16
    # (casting rsqrt down first) loses precision and accumulates across the ~36
    # RMSNorms per transformer stack.
    dt = x.dtype
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(v + self.eps)) * self.weight.float()).to(dt)


def rope_interleaved(x, base):
  """Interleaved RoPE over the T axis of [B, T, N, Dh] (lucidrains convention)."""
  dh, t = x.shape[-1], x.shape[1]
  inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
  f = torch.outer(torch.arange(t, device=x.device).float(), inv)
  cos = f.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
  sin = f.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  rot = torch.stack((-x2, x1), -1).reshape_as(x)
  return x * cos + rot * sin


class RoPE(nn.Module):
  """One RoPE per Encoder, holding the inverse-frequency buffer loaded FROM the
  checkpoint (JAX stores `rope.freqs`, computed in bf16 at train time -- recomputing
  it in fp32 differs by ~1e-3 and that error grows with sequence length)."""

  def __init__(self, dim, base):
    super().__init__()
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # init = formula; overwritten on load
    self.register_buffer("freqs", inv)

  def rotate(self, x):  # x: [B, T, N, Dh], rotate over the T axis
    t = x.shape[1]
    f = torch.outer(torch.arange(t, device=x.device).float(), self.freqs.float())
    cos = f.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
    sin = f.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), -1).reshape_as(x)
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
    self.per_dim_scale = nn.Parameter(torch.zeros(self.hd))

  def forward(self, query, key, value, attn_mask=None, rope=None):
    b, tq, d = query.shape
    q = self.q_proj(query).view(b, tq, self.nhead, self.hd)
    k = self.k_proj(key).view(b, key.shape[1], self.nhead, self.hd)
    v = self.v_proj(value).view(b, value.shape[1], self.nhead, self.hd)
    if self.rope_base is not None:
      # Use the Encoder's shared RoPE (checkpoint-loaded freqs) when provided;
      # fall back to recomputing only if absent.
      if rope is not None:
        q, k = rope.rotate(q), rope.rotate(k)
      else:
        q, k = rope_interleaved(q, self.rope_base), rope_interleaved(k, self.rope_base)
    q, k = self.query_ln(q), self.key_ln(k)
    # per-dim scale in float32 (softplus), then cast to compute dtype -- matches JAX PerDimScale.
    scale = 1.442695041 / math.sqrt(self.hd) * F.softplus(self.per_dim_scale.float())
    q = q * scale.to(q.dtype)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # [B,N,T,D]
    # bf16 SDPA (flash already does the softmax in float32 internally).
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
    return self.out_proj(o.transpose(1, 2).reshape(b, tq, d))


class MultiheadAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, activation="swiglu", rope_base=None):
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
      self.act = F.silu
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
    # Eager FFN chunking: process tokens in slices so the expanded
    # [tokens, dim_feedforward] activation is never materialized in full.
    if self.ffn_chunk_size is None:
      return self._ff_impl(x)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = torch.empty(flat.shape[0], self.linear2.out_features,
                      dtype=flat.dtype, device=flat.device)
    for s in range(0, flat.shape[0], self.ffn_chunk_size):
      out[s:s + self.ffn_chunk_size] = self._ff_impl(flat[s:s + self.ffn_chunk_size])
    return out.reshape(shape)

  def forward(self, q, k=None, v=None, attn_mask=None, rope=None):
    k = q if k is None else k
    v = q if v is None else v
    a = self.post_attn_ln(self.attn(self.pre_attn_ln(q), self.pre_attn_ln(k),
                                    self.pre_attn_ln(v), attn_mask, rope=rope))
    x = q + a
    return x + self._ff(x)


class InducedSelfAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, num_inds, activation="swiglu"):
    super().__init__()
    self.ind_vectors = nn.Parameter(torch.zeros(num_inds, d_model))
    self.mab1 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)
    self.mab2 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)

  def forward(self, src, attn_mask=None):
    ind = self.ind_vectors.unsqueeze(0).expand(src.shape[0], -1, -1)
    hidden = self.mab1(ind, src, src, attn_mask=attn_mask)
    return self.mab2(src, hidden, hidden)


class Encoder(nn.Module):
  def __init__(self, num_blocks, d_model, nhead, dim_ff, activation="swiglu",
               rope_base=100000.0):
    super().__init__()
    # One RoPE per Encoder (mirrors JAX `tf_row.rope.freqs`), shared by all blocks.
    self.rope = RoPE(d_model // nhead, rope_base) if rope_base is not None else None
    self.blocks = nn.ModuleList([
        MultiheadAttentionBlock(d_model, nhead, dim_ff, activation, rope_base)
        for _ in range(num_blocks)
    ])

  def forward(self, x, attn_mask=None):
    for blk in self.blocks:
      x = blk(x, attn_mask=attn_mask, rope=self.rope)
    return x


class SetTransformer(nn.Module):
  def __init__(self, num_blocks, d_model, nhead, dim_ff, num_inds,
               activation="swiglu"):
    super().__init__()
    self.blocks = nn.ModuleList([
        InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, activation)
        for _ in range(num_blocks)
    ])

  def forward(self, src, attn_mask=None):
    for blk in self.blocks:
      src = blk(src, attn_mask=attn_mask)
    return src


class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dims: List[int], out_dim, activation="gelu"):
    super().__init__()
    self.act = get_activation(activation)
    dims = [in_dim] + list(hidden_dims)
    self.layers = nn.ModuleList()  # only Linears; activation applied between
    for i in range(len(hidden_dims)):
      self.layers.append(nn.Linear(dims[i], dims[i + 1]))
    self.layers.append(nn.Linear(dims[-1], out_dim))

  def forward(self, x):
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

  def forward(self, y):  # y: [B, T] int
    y_long = y.long()
    y_mapped = torch.where(
        (y_long >= 0) & (y_long < self.num_classes),
        y_long,
        torch.tensor(self.num_classes, device=y.device)
    )
    oh = F.one_hot(y_mapped, self.num_classes + 1).to(self.projection.weight.dtype)
    oh_sliced = oh[..., :self.num_classes]
    return self.projection(oh_sliced)


class CellEmbedder(nn.Module):
  def __init__(self, embed_dim, max_classes, feature_group_size=3, num_freq=32,
               is_classifier=True):
    super().__init__()
    self.embed_dim = embed_dim
    self.fgs = feature_group_size
    self.is_classifier = is_classifier
    in_dim = feature_group_size
    self.register_buffer("fourier_frequencies", torch.zeros(in_dim, num_freq))
    self.register_buffer("fourier_frequencies_cat", torch.zeros(in_dim, num_freq))
    self.in_linear = nn.Linear(num_freq * 2, embed_dim)
    self.in_linear_cat = nn.Linear(num_freq * 2, embed_dim)
    if is_classifier:  # classification: embedding lookup over class ids
      self.y_embedder_lookup = nn.Embedding(max_classes, embed_dim)
    else:  # regression: MLP over the scalar target (y_col_embedder_encoder_nhid=6)
      self.y_embedder_lookup = MLP(1, [6], embed_dim, activation="gelu")
    self.row_chunk_size = None  # chunk the Fourier expansion over rows

  def _group(self, x, d=None):  # x: [B,T,H] -> [B,T,H,G]
    h = x.shape[-1]
    idxs = torch.arange(h, device=x.device)
    stacked = []
    if d is not None:
      # Per-batch wrap-around over each member's ACTIVE feature count d (not the
      # padded width h). Mirrors the JAX `% d_safe` path so zero-padded slots are
      # filled with wrapped real features rather than mixing padding into groups.
      d_safe = torch.clamp(d.to(torch.long), min=1)  # [B]
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        idx = (idxs[None, :] + offset) % d_safe[:, None]            # [B, H]
        idx = idx[:, None, :].expand(x.shape[0], x.shape[1], h)     # [B, T, H]
        stacked.append(torch.gather(x, -1, idx))
    else:
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        stacked.append(x[..., (idxs + offset) % h])
    return torch.stack(stacked, dim=-1)

  def _cell(self, x, cat_mask, d=None):  # [B,t,H] -> [B,t,HC,E] (Fourier expansion + sum over G)
    g = self._group(x, d=d).unsqueeze(-1).float()  # float32 Fourier: args g*freq reach ~30,
    dt = x.dtype                                    # so sin/cos must run in fp32 (matches JAX,
    ff = self.fourier_frequencies.float()           # whose freq params stay float32). Cast the
    ffc = self.fourier_frequencies_cat.float()      # fourier features back to compute dtype before in_linear.
    num_out = self.in_linear(torch.cat([(g * ff).sin(), (g * ff).cos()], dim=-1).to(dt))
    if cat_mask is not None:
      cat_out = self.in_linear_cat(torch.cat([(g * ffc).sin(), (g * ffc).cos()], dim=-1).to(dt))
      cmg = self._group(cat_mask[:, None, :].float(), d=d).bool()[..., None]
      return torch.where(cmg, cat_out, num_out).sum(-2)
    return num_out.sum(-2)

  def forward(self, x, y, train_size, cat_mask=None, d=None):
    # The Fourier expansion materializes [B,T,HC,G,E]; chunk over rows so that
    # huge intermediate never exists in full (rows are independent here).
    if self.row_chunk_size is None:
      cell = self._cell(x, cat_mask, d=d)
    else:
      parts = [self._cell(x[:, s:s + self.row_chunk_size], cat_mask, d=d)
               for s in range(0, x.shape[1], self.row_chunk_size)]
      cell = torch.cat(parts, dim=1)
    if self.is_classifier:
      y_clean = torch.clamp(y.long(), 0, self.y_embedder_lookup.num_embeddings - 1)
      y_emb = self.y_embedder_lookup(y_clean)  # [B,T,E]
    else:
      y_emb = self.y_embedder_lookup(y[..., None].to(cell.dtype))  # scalar -> [B,T,E]
    t = x.shape[1]
    tm = (torch.arange(t, device=x.device)[None, :] < train_size[:, None])[..., None, None]
    out = torch.where(tm, cell + y_emb[:, :, None, :], cell)
    if d is not None:
      # Zero the padded feature columns (cols >= d): the % d wrap above fills them
      # with real features for valid indexing, but they must not enter attention.
      hc = out.shape[2]
      colmask = (torch.arange(hc, device=out.device)[None, :]
                 < d[:, None])[:, None, :, None]  # [B, 1, HC, 1]
      out = torch.where(colmask, out, torch.zeros_like(out))
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

  def forward(self, x, train_size):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.permute(0, 2, 1, 3).reshape(b * hc, t, e)  # [B*HC, T, E]
    ts = train_size.repeat_interleave(hc)  # [B*HC]
    mask = (torch.arange(t, device=x.device)[None, :] < ts[:, None])[:, None, None, :]
    cc = self.col_chunk_size
    if cc is None or src.shape[0] <= cc:
      out = self._stage(src, mask)
    else:
      out = torch.cat([self._stage(src[s:s + cc], mask[s:s + cc])
                       for s in range(0, src.shape[0], cc)], dim=0)
    return out.reshape(b, hc, t, e).permute(0, 2, 1, 3)


class RowInteraction(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_cls,
               rope_base=100000.0, output_full=True):
    super().__init__()
    self.tf_row = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=rope_base)
    self.out_ln = RMSNorm(d_model)
    self.num_cls = num_cls
    self.output_full = output_full
    self.row_chunk_size = None  # chunk the independent row axis (B*T)

  def _stage(self, src, mask=None):
    out = self.tf_row(src, attn_mask=mask)
    return self.out_ln(out if self.output_full else out[:, : self.num_cls, :])

  def forward(self, x, d=None):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.reshape(b * t, hc, e)
    # Mask cross-column attention to the valid columns (CLS + d real features);
    # padded columns (>= d + num_cls) must not be attended to. Matches JAX.
    mask = None
    if d is not None:
      d_padded = d.to(torch.long) + self.num_cls  # [B]
      valid = torch.arange(hc, device=x.device)[None, :] < d_padded[:, None]  # [B, HC]
      mask = valid.repeat_interleave(t, dim=0)[:, None, None, :]  # [B*T, 1, 1, HC]
    rc = self.row_chunk_size
    if rc is None or src.shape[0] <= rc:
      out = self._stage(src, mask)
    else:
      out = torch.cat([self._stage(src[s:s + rc],
                                   None if mask is None else mask[s:s + rc])
                       for s in range(0, src.shape[0], rc)], dim=0)
    if self.output_full:
      return out.reshape(b, t, hc, e)
    return out.reshape(b, t, -1)


class ICLearning(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, max_classes, dim_ff,
               decoder_hidden, is_classifier=True):
    super().__init__()
    self.tf_icl = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=None)  # ICL has no RoPE
    self.ln = RMSNorm(d_model)
    self.is_classifier = is_classifier
    if is_classifier:  # one-hot y-encode; decode to per-class logits
      self.y_encoder = OneHotAndLinear(max_classes, d_model)
      self.decoder = MLP(d_model, [decoder_hidden], max_classes)
    else:  # MLP y-encode the scalar target; decode to a single value
      self.y_encoder = MLP(1, [decoder_hidden], d_model)
      self.decoder = MLP(d_model, [decoder_hidden], 1)

  def forward(self, reps, y, train_size):  # reps: [B,T,d_model]
    b, t, _ = reps.shape
    tm = (torch.arange(t, device=reps.device)[None, :] < train_size[:, None])
    if self.is_classifier:
      y_enc = self.y_encoder(y)
    else:
      y_enc = self.y_encoder(y[..., None].to(reps.dtype))
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
    self.cell_embedder = CellEmbedder(embed_dim, max_classes, feature_group_size,
                                      num_freq, is_classifier)
    self.col_embedder = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff, col_num_inds)
    self.col_embedder_2 = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff, col_num_inds)
    self.row_interactor = RowInteraction(embed_dim, row_num_blocks, row_nhead, ff,
                                         row_num_cls, output_full=True)
    self.row_interactor_2 = RowInteraction(embed_dim, row_num_blocks, row_nhead, ff,
                                           row_num_cls, output_full=False)
    self.cls_tokens = nn.Parameter(torch.zeros(row_num_cls, embed_dim))
    self.icl_predictor = ICLearning(icl_dim, icl_num_blocks, icl_nhead, max_classes,
                                    icl_dim * ff_factor,
                                    decoder_hidden or icl_dim * 2, is_classifier)

  def forward(self, x, y, train_size, cat_mask=None, d=None):
    emb = self.cell_embedder(x, y, train_size, cat_mask, d=d)
    emb = self.col_embedder(emb, train_size)
    b, t, _, e = emb.shape
    cls = self.cls_tokens.expand(b, t, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, train_size)
    reps = self.row_interactor_2(emb, d=d)
    return self.icl_predictor(reps, y, train_size)

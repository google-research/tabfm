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

"""Sequence-parallel (row-sharded) inference for the PyTorch TabFM backend.

TabFM reads the whole training fold as one in-context sequence, so a single
forward's activations grow with the number of rows and exceed one device's
memory for very large contexts (roughly >450k rows on an 80GB GPU). This
module shards the *rows* of one ensemble member's sequence across the ranks of
a ``torch.distributed`` process group, computing bit-equivalent attention (up
to floating-point summation order) without any approximation:

  * Cell embedding, row interaction (attention over features), feed-forward
    blocks and the decoder are row-independent and run locally on each shard.
  * The column set-transformer's inducing attention (``mab1``) attends over all
    context rows: each rank computes an online-softmax over its shard and the
    partial results are combined exactly across ranks via log-sum-exp weights.
  * The ICL blocks' self-attention attends from every row to all context rows:
    each rank projects K/V for its context shard, all ranks gather them, and a
    single fused SDPA runs locally per rank. Test rows are never attention
    keys anywhere in the model, so only context K/V is communicated.

Typical use (one process per GPU, e.g. under ``torchrun``)::

    dist.init_process_group("nccl")
    model = tabfm_v1_0_0.load(model_type="regression", device=f"cuda:{rank}")
    reg = TabFMRegressor(model=model, n_estimators=1)
    reg.fit(X_train, y_train)                      # cheap; no GPU forward
    preds = seqpar.predict(reg, X_test)            # sharded across the group

``predict`` runs every ensemble member through the sharded forward and applies
the same ensemble combination as the estimator's own ``predict`` /
``predict_proba``, so results match the single-device path up to bf16 noise.
"""

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

_MAB1_KEY_CHUNK = 8192
_ROW_CHUNK = 32768


def _proj_q(attn, xq):
  """Projects + normalizes queries the way ``MultiheadAttention`` does."""
  b, tq, _ = xq.shape
  q = attn.q_proj(xq).view(b, tq, attn.nhead, attn.hd)
  q = attn.query_ln(q)
  scale = (
      1.442695041 / math.sqrt(attn.hd) * F.softplus(attn.per_dim_scale.float())
  )
  return (q * scale.to(q.dtype)).transpose(1, 2).contiguous()  # [B,nh,T,hd]


def _dist_mab1(mab, ind_vectors, k_src):
  """``mab1`` (inducing queries over all context rows) with sharded keys.

  Args:
    mab: The ``MultiheadAttentionBlock`` used as mab1.
    ind_vectors: ``[num_inds, E]`` inducing vectors (identical on every rank).
    k_src: ``[B, T_ctx_local, E]`` this rank's context rows. Ranks may hold
      different numbers of context rows; only softmax statistics (whose shapes
      do not depend on the key count) cross ranks.

  Returns:
    ``[B, num_inds, E]`` block output, identical on every rank.
  """
  attn = mab.attn
  b = k_src.shape[0]
  q0 = ind_vectors.unsqueeze(0).expand(b, -1, -1)
  q = _proj_q(attn, mab.pre_attn_ln(q0))
  nh, ni, hd = q.shape[1], q.shape[2], q.shape[3]

  # Online softmax over local key chunks with fp32 accumulators. Keys are
  # pre-normed/projected chunk-by-chunk so no full-length copy materializes.
  m = torch.full((b, nh, ni), -float("inf"), device=q.device)
  den = torch.zeros((b, nh, ni), device=q.device)
  num = torch.zeros((b, nh, ni, hd), device=q.device)
  qf = q.float()
  for s in range(0, k_src.shape[1], _MAB1_KEY_CHUNK):
    xk = mab.pre_attn_ln(k_src[:, s : s + _MAB1_KEY_CHUNK])
    tk = xk.shape[1]
    ks = attn.key_ln(attn.k_proj(xk).view(b, tk, nh, hd)).transpose(1, 2)
    vs = attn.v_proj(xk).view(b, tk, nh, hd).transpose(1, 2)
    scores = qf @ ks.float().transpose(-1, -2)  # [B,nh,I,chunk] fp32
    m_new = torch.maximum(m, scores.amax(-1))
    alpha = torch.exp(m - m_new)
    p = torch.exp(scores - m_new[..., None])
    den = den * alpha + p.sum(-1)
    num = num * alpha[..., None] + p @ vs.float()
    m = m_new

  lse = m + torch.log(den)
  out_local = num / den[..., None]

  # Exact combine: out = sum_r softmax_r(lse_r) * out_r.
  world = dist.get_world_size()
  lse_all = [torch.empty_like(lse) for _ in range(world)]
  out_all = [torch.empty_like(out_local) for _ in range(world)]
  dist.all_gather(lse_all, lse.contiguous())
  dist.all_gather(out_all, out_local.contiguous())
  w = torch.softmax(torch.stack(lse_all), dim=0)[..., None]
  out = (torch.stack(out_all) * w).sum(0).to(q.dtype)

  out = out.transpose(1, 2).reshape(b, ni, nh * hd)
  x = q0 + mab.post_attn_ln(attn.out_proj(out))
  return x + mab._ff(x)  # pylint: disable=protected-access


def _row_chunked(fn, src):
  """Applies a row-independent fn over row slices of ``[B, T, E]``.

  Bounds the fp32 RMSNorm / FFN transients inside stock module calls, which
  otherwise materialize full-sequence-length copies.
  """
  out = None
  for s in range(0, src.shape[1], _ROW_CHUNK):
    o = fn(src[:, s : s + _ROW_CHUNK])
    if out is None:
      out = torch.empty(
          src.shape[0],
          src.shape[1],
          o.shape[-1],
          dtype=o.dtype,
          device=o.device,
      )
    out[:, s : s + _ROW_CHUNK] = o
  return out


def _dist_col_embedding(col, emb, c_local):
  """``ColEmbedding`` with the row axis sharded. ``emb``: [B,T_local,HC,E]."""
  b, t, hc, e = emb.shape
  src = emb.permute(0, 2, 1, 3).reshape(b * hc, t, e)
  del emb
  for blk in col.tf_col.blocks:
    hidden = _dist_mab1(blk.mab1, blk.ind_vectors, src[:, :c_local])
    # mab2 rows attend only to the replicated inducing outputs: local.
    src = _row_chunked(lambda s: blk.mab2(s, hidden, hidden), src)  # pylint: disable=cell-var-from-loop
  out = _row_chunked(lambda s: col.ln_w(col.out_w(s)), src)
  return out.reshape(b, hc, t, e).permute(0, 2, 1, 3)


def _dist_icl_block(blk, x, c_local, c_max, key_mask):
  """One ICL self-attention block with keys gathered from every rank.

  Ranks may hold unequal context shards: each rank zero-pads its projected
  K/V to ``c_max`` rows before the all-gather and the padded slots are
  excluded via ``key_mask``.
  """
  attn = blk.attn
  nh, hd = attn.nhead, attn.hd
  xn = blk.pre_attn_ln(x)
  q = _proj_q(attn, xn)
  kn = xn[:, :c_local]
  b = kn.shape[0]
  k = attn.key_ln(attn.k_proj(kn).view(b, c_local, nh, hd)).transpose(1, 2)
  v = attn.v_proj(kn).view(b, c_local, nh, hd).transpose(1, 2)
  if c_local < c_max:
    pad = (0, 0, 0, c_max - c_local)  # pad the row (dim -2) axis
    k, v = F.pad(k, pad), F.pad(v, pad)
  world = dist.get_world_size()
  k_all = [torch.empty_like(k) for _ in range(world)]
  v_all = [torch.empty_like(v) for _ in range(world)]
  dist.all_gather(k_all, k.contiguous())
  dist.all_gather(v_all, v.contiguous())
  key = torch.cat(k_all, dim=2)
  val = torch.cat(v_all, dim=2)
  del k_all, v_all
  o = F.scaled_dot_product_attention(q, key, val, attn_mask=key_mask, scale=1.0)
  del key, val
  b, nh, tq, hd = o.shape
  o = o.transpose(1, 2).reshape(b, tq, nh * hd)
  x = x + blk.post_attn_ln(attn.out_proj(o))
  return x + blk._ff(x)  # pylint: disable=protected-access


@torch.inference_mode()
def seqpar_forward(model, x_local, y_local, c_local, cat_mask=None, d=None):
  """Sharded forward for one ensemble member. Call on every rank.

  Args:
    model: Loaded PyTorch ``TabFM`` (classifier or regressor variant).
    x_local: ``[1, T_local, H]`` float array; this rank's context rows first,
      then its share of the test rows.
    y_local: ``[1, T_local]`` float array; ``-100.0`` on test positions.
    c_local: Number of context rows in this rank's shard (may differ by rank).
    cat_mask: Optional ``[1, H]`` bool array of categorical-feature flags.
    d: Optional ``[1]`` int array with the active-feature count (for feature
      padding).

  Returns:
    ``[T_local_test, L_out]`` numpy array of this rank's test outputs (scaled
    predictions for regression, per-class logits for classification).
  """
  dev = next(model.parameters()).device
  x = torch.as_tensor(np.asarray(x_local, dtype=np.float32), device=dev)
  y = torch.as_tensor(np.asarray(y_local, dtype=np.float32), device=dev)
  ts = torch.tensor([c_local], device=dev, dtype=torch.long)
  cm = (
      torch.as_tensor(np.asarray(cat_mask), device=dev, dtype=torch.bool)
      if cat_mask is not None
      else None
  )
  dt = (
      torch.as_tensor(np.asarray(d), device=dev, dtype=torch.long)
      if d is not None
      else None
  )

  # Every rank must agree on the padded per-rank key length for the gathers.
  world = dist.get_world_size()
  c_locals = [None] * world
  dist.all_gather_object(c_locals, c_local)
  c_max = max(c_locals)
  key_valid = torch.cat([torch.arange(c_max, device=dev) < c for c in c_locals])
  key_mask = (
      None
      if all(c == c_max for c in c_locals)
      else key_valid[None, None, None, :]
  )

  x = torch.nan_to_num(x, nan=-100.0).to(model.cls_tokens.dtype)
  emb = model.cell_embedder(x, y, ts, cat_mask=cm, d=dt)
  emb = _dist_col_embedding(model.col_embedder, emb, c_local)
  b, t, _, _ = emb.shape
  cls = model.cls_tokens.expand(b, t, -1, -1)
  emb = torch.cat([cls, emb], dim=2)
  emb = model.row_interactor(emb, d=dt)
  emb = _dist_col_embedding(model.col_embedder_2, emb, c_local)
  reps = model.row_interactor_2(emb, d=dt)
  del emb

  icl = model.icl_predictor
  tm = torch.arange(t, device=dev)[None, :] < ts[:, None]
  if icl.is_classifier:
    y_enc = icl.y_encoder(y)
  else:
    y_enc = icl.y_encoder(y[..., None].to(reps.dtype))
  r = reps + y_enc * tm[..., None]
  del reps, y_enc
  for blk in icl.tf_icl.blocks:
    r = _dist_icl_block(blk, r, c_local, c_max, key_mask)
  out = icl.decoder(icl.ln(r))
  return out[0, c_local:, :].float().cpu().numpy()


def _shard_bounds(n, world, rank):
  """Contiguous near-equal split of ``n`` items; returns (start, stop)."""
  base, rem = divmod(n, world)
  start = rank * base + min(rank, rem)
  return start, start + base + (1 if rank < rem else 0)


def _member_outputs(estimator, X, rank, world):
  """Runs every ensemble member through the sharded forward.

  Returns ``[n_members, n_test, L_out]`` outputs, replicated on every rank.
  """
  x_enc = estimator.X_encoder_.transform(X)
  data = estimator.ensemble_generator_.transform(x_enc)
  xs, ys, cat_masks, ds, _ = (
      estimator.ensemble_generator_.prepare_ensemble_tensors(data)
  )
  n_members = xs.shape[0]
  n_train = ys.shape[1]
  n_test = xs.shape[1] - n_train

  c0, c1 = _shard_bounds(n_train, world, rank)
  t0, t1 = _shard_bounds(n_test, world, rank)
  outs = []
  for mi in range(n_members):
    x_local = np.concatenate(
        [xs[mi, c0:c1], xs[mi, n_train + t0 : n_train + t1]], axis=0
    )[None]
    y_local = np.concatenate([ys[mi, c0:c1], np.full(t1 - t0, -100.0)], axis=0)[
        None
    ]
    out = seqpar_forward(
        estimator.model,
        x_local,
        y_local,
        c1 - c0,
        cat_mask=cat_masks[mi : mi + 1] if cat_masks is not None else None,
        d=ds[mi : mi + 1] if ds is not None else None,
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, out)
    outs.append(np.concatenate(gathered, axis=0))
  return np.stack(outs, axis=0)


def predict(estimator, X):
  """Sharded equivalent of ``TabFMRegressor.predict``.

  Must be called collectively on every rank of the process group with the same
  fitted estimator (fit is deterministic given ``random_state``) and the same
  ``X``. Every rank returns the full prediction vector.
  """
  outputs = _member_outputs(
      estimator, X, dist.get_rank(), dist.get_world_size()
  )
  predictions = outputs.squeeze(-1)  # [E, T]
  return estimator._combine_predictions(predictions)  # pylint: disable=protected-access


def predict_proba(estimator, X):
  """Sharded equivalent of ``TabFMClassifier.predict_proba`` (all ranks)."""
  outputs = _member_outputs(
      estimator, X, dist.get_rank(), dist.get_world_size()
  )
  outputs = outputs[..., : estimator.n_classes_]
  offsets = []
  for offs in estimator.ensemble_generator_.class_shift_offsets_.values():
    offsets.extend(offs)
  logits = np.stack([
      np.concatenate([out[..., off:], out[..., :off]], axis=-1)
      for out, off in zip(outputs, offsets)
  ])
  return estimator._process_logits(logits)  # pylint: disable=protected-access

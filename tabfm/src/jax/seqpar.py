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

"""Sequence-parallel (row-sharded) inference for the JAX TabFM backend.

Mirrors ``tabfm.src.pytorch.seqpar``: the in-context rows of each ensemble
member are sharded across the devices of a mesh with ``jax.shard_map``, so
training folds that exceed a single device's memory can be used as context.
Weights are replicated; attention is exact (no approximation):

  * Cell embedding, row interaction (attention over features), feed-forward
    blocks and the decoder are row-independent and run locally per shard,
    reusing the stock nnx modules.
  * The column set-transformer's inducing attention runs an fp32 online
    softmax over each device's context shard (scanned in key chunks so no
    full-length score matrix materializes) and combines the partial results
    exactly across devices via log-sum-exp weights.
  * ICL self-attention all-gathers each device's projected context K/V (test
    rows are never keys) and runs the chunked memory-efficient attention
    locally per device, with padded key slots masked via an additive bias.

Typical use (single process driving all local devices)::

    model = tabfm_v1_0_0.load(model_type="regression")
    reg = TabFMRegressor(model=model, n_estimators=4)
    reg.fit(X_train, y_train)              # cheap; no device forward
    preds = seqpar.predict(reg, X_test)    # sharded over jax.devices()

``predict`` / ``predict_proba`` run every ensemble member through the sharded
forward and apply the same ensemble combination as the estimators' own
prediction paths. Device blocks are padded to a common 128-multiple length
(the chunked-attention granularity), so unequal shards are supported; the
padded rows are masked out of every attention and trimmed from the output.

Multi-process (e.g. multi-host TPU slice) runs are supported: initialize the
runtime with ``jax.distributed.initialize()`` first, then run the same
program on every process -- ``fit`` (deterministic given ``random_state``)
and ``predict`` are called on all processes, and every process returns the
full predictions. Input shards are assembled per process with
``jax.make_array_from_callback`` and the (tiny) output is re-replicated
across devices so it is readable everywhere.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from flax import nnx

from tabfm.src.jax import memory_efficient_attention as mea

_AXIS = "seqpar_rows"
_PAD = 128  # chunked-attention length granularity
_MAB1_KEY_CHUNK = 2048  # keys per online-softmax step (multiple of _PAD)
_ROW_CHUNK = 32768  # rows per slice for the local mab2 / stage-tail calls


def _round_up(n, k):
  return ((n + k - 1) // k) * k


def _shard_bounds(n, world, rank):
  """Contiguous near-equal split of ``n`` items; returns (start, stop)."""
  base, rem = divmod(n, world)
  start = rank * base + min(rank, rem)
  return start, start + base + (1 if rank < rem else 0)


def _iter_blocks(stacked):
  """Yields per-layer modules from an ``nnx.vmap``-stacked block module."""
  if hasattr(stacked, "__iter__"):
    yield from stacked
    return
  graphdef, state = nnx.split(stacked)
  n = jax.tree.leaves(state)[0].shape[0]
  for i in range(n):
    yield nnx.merge(graphdef, jax.tree.map(lambda a, i=i: a[i], state))


def _row_chunked(fn, src):
  """Applies a row-independent fn over row slices of ``[B, T, E]``."""
  outs = [
      fn(src[:, s : s + _ROW_CHUNK]) for s in range(0, src.shape[1], _ROW_CHUNK)
  ]
  return outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=1)


def _mab1_combined(mab, ind, k_src, ts_local):
  """mab1 (inducing queries over all context rows) with sharded keys.

  ``k_src``: [B, c_blk, E] local context block; rows at positions >=
  ``ts_local`` are padding and masked out. Online softmax over key chunks
  (fp32 accumulators), then an exact cross-device log-sum-exp combine. Only
  softmax statistics (whose shapes do not depend on the key count) cross
  devices, so unequal shards work natively.
  """
  attn = mab.attn
  b, c_blk, e = k_src.shape
  ni = ind.shape[0]
  nh, hd = attn.num_heads, attn.head_dim
  f32 = jnp.float32

  q0 = jnp.broadcast_to(ind, (b, ni, e))
  qn = mab.pre_attn_ln(q0)
  q = attn.q_proj(qn).reshape(b, ni, nh, hd)
  q = attn.per_dim_scale(attn.query_ln(q))
  qf = jnp.einsum("bind->bnid", q.astype(f32))  # [b, nh, I, hd]

  n_chunks = c_blk // _MAB1_KEY_CHUNK
  k_chunks = k_src.reshape(b, n_chunks, _MAB1_KEY_CHUNK, e).transpose(
      (1, 0, 2, 3)
  )

  def step(carry, kc_and_idx):
    m, den, num = carry
    kc, idx = kc_and_idx
    kn = mab.pre_attn_ln(kc)  # [b, KC, e]
    k = attn.key_ln(attn.k_proj(kn).reshape(b, -1, nh, hd))
    v = attn.v_proj(kn).reshape(b, -1, nh, hd)
    s = jnp.einsum("bnid,bjnd->bnij", qf, k.astype(f32))  # [b,nh,I,KC]
    pos = idx * _MAB1_KEY_CHUNK + jnp.arange(_MAB1_KEY_CHUNK)
    s = jnp.where(pos[None, None, None, :] < ts_local, s, -jnp.inf)
    m_new = jnp.maximum(m, s.max(-1))
    alpha = jnp.exp(m - m_new)
    p = jnp.exp(s - m_new[..., None])
    den = den * alpha + p.sum(-1)
    num = num * alpha[..., None] + jnp.einsum(
        "bnij,bjnd->bnid", p, v.astype(f32)
    )
    return (m_new, den, num), None

  init = (
      jnp.full((b, nh, ni), -jnp.inf),
      jnp.zeros((b, nh, ni)),
      jnp.zeros((b, nh, ni, hd)),
  )
  (m, den, num), _ = jax.lax.scan(step, init, (k_chunks, jnp.arange(n_chunks)))

  # Exact cross-device combine of the online-softmax statistics.
  m_glob = jax.lax.pmax(m, _AXIS)
  scale = jnp.exp(m - m_glob)
  den = jax.lax.psum(den * scale, _AXIS)
  num = jax.lax.psum(num * scale[..., None], _AXIS)
  out = (num / den[..., None]).astype(q0.dtype)  # [b, nh, I, hd]

  out = jnp.einsum("bnid->bind", out).reshape(b, ni, nh * hd)
  x = q0 + mab.post_attn_ln(attn.out_proj(out))
  return x + mab._ff_block(x)  # pylint: disable=protected-access


def _col_sharded(col, emb, c_blk, ts_local):
  """ColEmbedding with the row axis sharded. ``emb``: [B, T_local, HC, E]."""
  b, t, hc, e = emb.shape
  src = emb.transpose((0, 2, 1, 3)).reshape(b * hc, t, e)
  for blk in _iter_blocks(col.tf_col.blocks):
    hidden = _mab1_combined(
        blk.mab1, blk.ind_vectors[...], src[:, :c_blk], ts_local
    )
    # mab2 rows attend only to the replicated inducing outputs: local.
    src = _row_chunked(lambda s: blk.mab2(q=s, k=hidden, v=hidden), src)  # pylint: disable=cell-var-from-loop
  out = _row_chunked(lambda s: col.ln_w(col.out_w(s)), src)
  return out.reshape(b, hc, t, e).transpose((0, 2, 1, 3))


def _icl_block_sharded(blk, r, c_blk, key_bias):
  """One ICL self-attention block; keys = all devices' context blocks."""
  attn = blk.attn
  nh, hd = attn.num_heads, attn.head_dim
  xn = blk.pre_attn_ln(r)
  q = attn.q_proj(xn).reshape(1, -1, nh, hd)
  kn = xn[:, :c_blk]
  k = attn.key_ln(attn.k_proj(kn).reshape(1, c_blk, nh, hd))
  v = attn.v_proj(kn).reshape(1, c_blk, nh, hd)
  q = attn.per_dim_scale(attn.query_ln(q))
  key = jax.lax.all_gather(k, _AXIS, axis=1, tiled=True)
  val = jax.lax.all_gather(v, _AXIS, axis=1, tiled=True)
  tgt = q.shape[1]
  ao = mea.dot_product_attention_multihead(
      query=q,
      key=key,
      value=val,
      bias=key_bias,
      dtype=np.dtype(r.dtype.name),
      enable_dropout=False,
      query_chunk_size=_PAD if tgt >= _PAD else tgt,
      key_chunk_size=_PAD,
  )
  o = attn.out_proj(ao.reshape(1, -1, nh * hd))
  x = r + blk.post_attn_ln(o)
  return x + blk._ff_block(x)  # pylint: disable=protected-access


def _make_forward(graphdef, c_blk, t_blk, has_cat, has_d):
  """Builds the per-device shard_map body for one ensemble member."""

  def forward(state, x, y, ts, cat_mask, d):
    m = nnx.merge(graphdef, state)
    dtype = m.dtype
    x = jnp.nan_to_num(x, nan=-100.0).astype(dtype)
    y = y.astype(dtype)
    cm = cat_mask if has_cat else None
    dd = d if has_d else None
    emb = m.cell_embedder(x, y, train_size=ts, d=dd, cat_mask=cm)
    emb = _col_sharded(m.col_embedder, emb, c_blk, ts[0])
    b1, t1 = emb.shape[:2]
    cls = jnp.broadcast_to(
        m.cls_tokens[...], (b1, t1, m.row_num_cls, m.embed_dim)
    )
    emb = jnp.concatenate([cls, emb], axis=-2)
    emb = m.row_interactor(emb, d=dd)
    emb = _col_sharded(m.col_embedder_2, emb, c_blk, ts[0])
    reps = m.row_interactor_2(emb, d=dd)

    icl = m.icl_predictor
    if icl.is_classifier:
      y_enc = icl.y_encoder(y.astype(jnp.int32))
    else:
      y_enc = icl.y_encoder(y[..., None])
    tmask = jnp.arange(reps.shape[1])[None, :] < ts[:, None]
    r = reps + y_enc * tmask[..., None]

    # Key-validity bias for the gathered context blocks of every device.
    ts_all = jax.lax.all_gather(ts[0], _AXIS)  # [world]
    valid = (jnp.arange(c_blk)[None, :] < ts_all[:, None]).reshape(-1)
    key_bias = jnp.where(valid, 0.0, -1e30)[None, None, None, :]
    for blk in _iter_blocks(icl.tf_icl.blocks):
      r = _icl_block_sharded(blk, r, c_blk, key_bias)
    out = icl.decoder(icl.ln(r))  # [1, T_local, L]
    return out[:, c_blk:, :]  # [1, t_blk, L]

  return forward


def _member_outputs(estimator, X, mesh):
  """Runs every ensemble member through the sharded forward.

  Returns ``[n_members, n_test, L_out]`` float32 outputs.
  """
  x_enc = estimator.X_encoder_.transform(X)
  data = estimator.ensemble_generator_.transform(x_enc)
  xs, ys, cat_masks, ds, _ = (
      estimator.ensemble_generator_.prepare_ensemble_tensors(data)
  )
  n_members = xs.shape[0]
  n_train = ys.shape[1]
  n_test = xs.shape[1] - n_train
  h = xs.shape[-1]
  world = mesh.devices.size
  if n_train < world:
    raise ValueError(
        f"n_train ({n_train}) must be >= the device count ({world})."
    )

  bounds_c = [_shard_bounds(n_train, world, r) for r in range(world)]
  bounds_t = [_shard_bounds(n_test, world, r) for r in range(world)]
  c_blk = _round_up(max(c1 - c0 for c0, c1 in bounds_c), _MAB1_KEY_CHUNK)
  t_blk = max(_round_up(max(t1 - t0 for t0, t1 in bounds_t), _PAD), _PAD)

  graphdef, state = nnx.split(estimator.model)
  state = jax.device_put(state, NamedSharding(mesh, P()))
  has_cat = cat_masks is not None
  has_d = ds is not None
  fwd = _make_forward(graphdef, c_blk, t_blk, has_cat, has_d)
  # The output is re-replicated across all devices so that every process of a
  # multi-process (e.g. multi-host TPU) run can read it back directly; the
  # gathered tensor is tiny ([1, W * t_blk, L]).
  sharded = jax.jit(
      jax.shard_map(
          fwd,
          mesh=mesh,
          in_specs=(
              P(),
              P(None, _AXIS, None),
              P(None, _AXIS),
              P(_AXIS),
              P(),
              P(),
          ),
          out_specs=P(None, _AXIS, None),
          check_vma=False,
      ),
      out_shardings=NamedSharding(mesh, P()),
  )

  x_shard = NamedSharding(mesh, P(None, _AXIS, None))
  y_shard = NamedSharding(mesh, P(None, _AXIS))
  ts_shard = NamedSharding(mesh, P(_AXIS))
  repl = NamedSharding(mesh, P())

  def to_global(arr, sharding):
    # Every process holds the full host copy (fit is deterministic, so all
    # processes computed identical tensors); each contributes the slices its
    # addressable devices need. Works identically in single-process runs.
    return jax.make_array_from_callback(
        arr.shape, sharding, lambda idx: arr[idx]
    )

  outs = []
  for mi in range(n_members):
    xg = np.zeros((1, world * (c_blk + t_blk), h), np.float32)
    yg = np.full((1, world * (c_blk + t_blk)), -100.0, np.float32)
    ts_g = np.zeros((world,), np.int32)
    for r, ((c0, c1), (t0, t1)) in enumerate(zip(bounds_c, bounds_t)):
      base = r * (c_blk + t_blk)
      xg[0, base : base + c1 - c0] = xs[mi, c0:c1]
      yg[0, base : base + c1 - c0] = ys[mi, c0:c1]
      xg[0, base + c_blk : base + c_blk + t1 - t0] = xs[
          mi, n_train + t0 : n_train + t1
      ]
      ts_g[r] = c1 - c0
    args = (
        state,
        to_global(xg, x_shard),
        to_global(yg, y_shard),
        to_global(ts_g, ts_shard),
        to_global(
            np.asarray(cat_masks[mi : mi + 1])
            if has_cat
            else np.zeros((1, h), bool),
            repl,
        ),
        to_global(
            np.asarray(ds[mi : mi + 1], np.int32)
            if has_d
            else np.zeros((1,), np.int32),
            repl,
        ),
    )
    out = np.asarray(
        jax.block_until_ready(sharded(*args)), np.float32
    )  # [1, W*t_blk, L]; fully replicated, so readable from any process
    parts = [
        out[0, r * t_blk : r * t_blk + (t1 - t0)]
        for r, (t0, t1) in enumerate(bounds_t)
    ]
    outs.append(np.concatenate(parts, axis=0))
  return np.stack(outs, axis=0)


def _default_mesh():
  return jax.make_mesh((len(jax.devices()),), (_AXIS,))


def predict(estimator, X, mesh=None):
  """Sharded equivalent of ``TabFMRegressor.predict``."""
  outputs = _member_outputs(estimator, X, mesh or _default_mesh())
  return estimator._combine_predictions(outputs.squeeze(-1))  # pylint: disable=protected-access


def predict_proba(estimator, X, mesh=None):
  """Sharded equivalent of ``TabFMClassifier.predict_proba``."""
  outputs = _member_outputs(estimator, X, mesh or _default_mesh())
  outputs = outputs[..., : estimator.n_classes_]
  offsets = []
  for offs in estimator.ensemble_generator_.class_shift_offsets_.values():
    offsets.extend(offs)
  logits = np.stack([
      np.concatenate([out[..., off:], out[..., :off]], axis=-1)
      for out, off in zip(outputs, offsets)
  ])
  return estimator._process_logits(logits)  # pylint: disable=protected-access

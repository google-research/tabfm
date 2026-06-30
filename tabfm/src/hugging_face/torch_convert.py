"""End-to-end JAX->PyTorch converter + parity test on a small TabFM.

Validates the whole port + the weight converter (unstacking vmap'd blocks,
transposing Linear kernels, name remap) by comparing a small random-init JAX
TabFM against the torch port on identical input, in float32.
"""
import re
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import torch
from flax import nnx

from tabfm.src.jax.model import TabFM as JaxTabFM, YEmbeddingScheme
from tabfm.src.pytorch import model as MT

CFG = dict(embed_dim=8, max_classes=3, col_num_blocks=2, col_nhead=2,
           col_num_inds=4, row_num_blocks=2, row_nhead=2, row_num_cls=2,
           icl_num_blocks=2, icl_nhead=2, ff_factor=2, feature_group_size=3)


def jax_params(m):
  flat = jtu.tree_flatten_with_path(nnx.state(m))[0]
  return {".".join(str(getattr(k, "key", k)) for k in kp).replace("..value", ""): np.asarray(v)
          for kp, v in flat}


def to_jax_name(tkey):
  """Map a torch state_dict key -> (jax_name, block_idx_or_None, kind)."""
  block_idx = None
  jk = tkey
  m = re.search(r"\.blocks\.(\d+)\.", tkey)
  if m:
    block_idx = int(m.group(1))
    jk = jk.replace(f".blocks.{block_idx}.", ".blocks.")
  # any MLP (decoder, y_encoder, regression y_embedder_lookup): torch layers.k
  # (consecutive Linears) -> jax layers.2k (activation module sits between).
  m2 = re.search(r"\.layers\.(\d+)\.", jk)
  if m2:
    k = int(m2.group(1))
    jk = re.sub(r"\.layers\.\d+\.", f".layers.{2*k}.", jk, count=1)
  if jk.endswith(".per_dim_scale"):
    return jk + ".per_dim_scale", block_idx, "direct"
  if jk.endswith(".weight"):
    base = jk[:-len(".weight")]
    # classification y-embedding is an nnx.Embed (param named ".embedding");
    # the regression variant is an MLP and falls through to kernel resolution.
    if jk.endswith("y_embedder_lookup.weight"):
      return base + ".embedding", block_idx, "direct"
    return base, block_idx, "weight"  # resolve scale/kernel by shape later
  if jk.endswith(".bias"):
    return jk, block_idx, "direct"
  return jk, block_idx, "direct"  # buffers, cls_tokens, ind_vectors


def convert(jax_p, torch_model):
  sd = {}
  missing = []
  for tkey, tparam in {**dict(torch_model.named_parameters()),
                       **dict(torch_model.named_buffers())}.items():
    jname, bi, kind = to_jax_name(tkey)
    # resolve ".weight" ambiguity (RMSNorm scale vs Linear kernel) by trying both
    candidates = [jname] if kind != "weight" else [jname + ".scale", jname + ".kernel"]
    arr = None
    for c in candidates:
      if c in jax_p:
        arr = jax_p[c]; chosen = c; break
    if arr is None:
      # use_bias=False in the source => no bias param; equivalent to bias 0.
      if tkey.endswith(".bias"):
        sd[tkey] = torch.zeros_like(tparam)
        continue
      missing.append((tkey, candidates)); continue
    if bi is not None:
      arr = arr[bi]
    if chosen.endswith(".kernel") and arr.ndim == 2:
      arr = arr.T
    t = torch.tensor(np.array(arr, dtype=np.float32))
    if tuple(t.shape) != tuple(tparam.shape):
      missing.append((tkey, f"shape {tuple(t.shape)} != {tuple(tparam.shape)} from {chosen}")); continue
    sd[tkey] = t
  return sd, missing


def run_parity(loss, is_classifier):
  tag = "classification" if is_classifier else "regression"
  jm = JaxTabFM(loss=loss, use_fourier_features=True, feature_group=True,
                activation="swiglu",
                y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
                rngs=nnx.Rngs(0), dtype=jnp.float32, **CFG)
  jp = jax_params(jm)
  tm = MT.TabFM(is_classifier=is_classifier, **{k: v for k, v in CFG.items()}).eval().float()

  sd, missing = convert(jp, tm)
  print(f"[{tag}] converted {len(sd)} / "
        f"{len(list(tm.named_parameters()))+len(list(tm.named_buffers()))} tensors; "
        f"missing {len(missing)}")
  for mk in missing[:12]:
    print("  MISSING:", mk)
  if missing:
    return False
  tm.load_state_dict(sd, strict=True)

  B, T, H = 1, 12, 5
  rng = np.random.default_rng(0)
  x = rng.random((B, T, H)).astype(np.float32)
  if is_classifier:
    y = rng.integers(0, 3, (B, T)).astype(np.int32)
  else:
    y = rng.standard_normal((B, T)).astype(np.float32)
  ts = np.array([7], dtype=np.int32)
  jout = np.asarray(jm(jnp.asarray(x), jnp.asarray(y), train_size=jnp.asarray(ts)))
  with torch.no_grad():
    tout = tm(torch.tensor(x), torch.tensor(y), torch.tensor(ts)).numpy()
  diff = float(np.max(np.abs(jout - tout)))
  ok = diff < 1e-4
  print(f"[{tag}] shapes {jout.shape} {tout.shape}  max abs diff = {diff:.3e}"
        f"  -> {'PARITY' if ok else 'DIFF'}")
  return ok


def main():
  ok = [run_parity("cross_entropy", True), run_parity("rmse", False)]
  print("ALL OK" if all(ok) else "FAILED")


if __name__ == "__main__":
  main()

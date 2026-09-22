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

"""Converts a TabFM PyTorch checkpoint (safetensors) to MLX (.npz) weights.

Module/param names are identical between tabfm.src.pytorch.model and
tabfm.src.mlx.model, so conversion is a pure format translation (no key
remapping). Needs only `safetensors` + `mlx` (+ numpy) -- no torch.

Usage:
  python -m tabfm.src.mlx.convert <model.safetensors> <out.npz>
      [--dtype bfloat16|float32]
"""

import argparse
import os
import sys

import mlx.core as mx
import numpy as np


def convert_safetensors(
    safetensors_path: str, npz_path: str, dtype: mx.Dtype = mx.bfloat16
) -> str:
  """Converts safetensors -> MLX npz (optionally casting storage dtype)."""
  try:
    from safetensors import safe_open
  except ImportError as exc:
    raise ImportError(
        "safetensors is required for conversion: pip install safetensors"
    ) from exc
  weights = {}
  with safe_open(safetensors_path, framework="np") as f:
    keys = list(f.keys())
    for k in keys:
      arr = mx.array(f.get_tensor(k))
      if dtype is not None:
        arr = arr.astype(dtype)
      weights[k] = arr
  mx.savez(npz_path, **weights)
  mx.eval(weights)
  n_params = sum(int(np.prod(v.shape)) for v in weights.values())
  print(
      f"converted {len(weights)} tensors, {n_params / 1e9:.3f}B params "
      f"-> {npz_path} "
      f"({os.path.getsize(npz_path) / 1e9:.2f} GB)"
  )
  return npz_path


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("safetensors_path")
  parser.add_argument("npz_path")
  parser.add_argument(
      "--dtype", default="bfloat16", choices=["bfloat16", "float32"]
  )
  args = parser.parse_args(argv)
  convert_safetensors(
      args.safetensors_path,
      args.npz_path,
      dtype=mx.bfloat16 if args.dtype == "bfloat16" else mx.float32,
  )


if __name__ == "__main__":
  main(sys.argv[1:])

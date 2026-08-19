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

"""Multi-GPU regression with TabFM v1.0.0 via sequence-parallel inference.

Shards the in-context rows of each ensemble member across all GPUs of a
``torch.distributed`` process group, so training folds that exceed a single
device's memory can be used as context. Launch one process per GPU:

    torchrun --standalone --nproc_per_node=4 examples/seqpar_regression_example.py

The script also runs on a single GPU (``--nproc_per_node=1``).
"""

import os

import numpy as np
import torch
import torch.distributed as dist

import tabfm
from tabfm.src.pytorch import seqpar


def make_data(n_train=20_000, n_test=1_000, n_features=20, seed=0):
  """Synthetic regression data: linear signal plus noise."""
  rng = np.random.default_rng(seed)
  x = rng.standard_normal((n_train + n_test, n_features)).astype(np.float32)
  w = np.random.default_rng(1).standard_normal(n_features)
  y = x @ w + 0.1 * rng.standard_normal(n_train + n_test)
  return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def main():
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  torch.cuda.set_device(local_rank)
  dist.init_process_group("nccl")

  x_train, y_train, x_test, y_test = make_data()

  model = tabfm.tabfm_v1_0_0_pytorch.load(
      model_type="regression", device=f"cuda:{local_rank}"
  )
  reg = tabfm.TabFMRegressor(model=model, n_estimators=4, random_state=0)
  reg.fit(x_train, y_train)  # cheap: preprocessing only, no GPU forward

  # Collective call: every rank participates and returns the full predictions.
  preds = seqpar.predict(reg, x_test)

  if rank == 0:
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    r2 = 1 - np.sum((y_test - preds) ** 2) / np.sum(
        (y_test - y_test.mean()) ** 2
    )
    print(f"world_size={dist.get_world_size()}  RMSE={rmse:.4f}  R2={r2:.4f}")

  dist.destroy_process_group()


if __name__ == "__main__":
  main()

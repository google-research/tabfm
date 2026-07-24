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

"""Example showing how to generate synthetic data with TabFM v1.0.0."""

import numpy as np
import pandas as pd
import tabfm


def run_example(model=None) -> pd.DataFrame:
  """Generates synthetic rows mimicking a small mixed-type dataset."""
  if model is None:
    # Option A: JAX Backend (default)
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")

    # Option B: PyTorch Backend
    # model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")

  # 2. Build a reference dataset with correlated mixed-type columns.
  rng = np.random.default_rng(0)
  n = 200
  age = rng.uniform(20, 65, n)
  job = rng.choice(["engineer", "manager", "analyst"], n)
  income = (40000 + age * 1500 + (job == "manager") * 30000
            + rng.normal(0, 5000, n))
  X = pd.DataFrame({"age": age, "job": job, "income": income})

  # 3. Fit the generator and sample new rows. Generation only needs the
  # classification model; t < 1 concentrates near the modes of the data.
  gen = tabfm.TabFMDataGenerator(model=model, random_state=42)
  gen.fit(X, categorical_features=["job"])
  synthetic = gen.sample(n_samples=100, t=1.0)

  # 4. Compare real and synthetic statistics.
  print("Real income by job:\n", X.groupby("job")["income"].mean())
  print("Synthetic income by job:\n",
        synthetic.groupby("job")["income"].mean())
  return synthetic


if __name__ == "__main__":
  print("Running TabFM synthetic data generation... (Note: compilation and "
        "model execution may take a few minutes on first run)")
  data = run_example()
  print("Synthetic sample:\n", data.head())

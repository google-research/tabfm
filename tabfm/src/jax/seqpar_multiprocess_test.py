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

"""Multi-process tests for sequence-parallel JAX inference.

Rehearses the multi-host execution model (e.g. a multi-host TPU slice) on
CPU: two OS processes are spawned, each with two simulated host devices
(four global devices), joined via ``jax.distributed.initialize``. Every
process runs the same fit + sharded predict, mirroring how every host of a
TPU slice runs the same program; the results are compared against the
estimator's plain single-process prediction path.
"""

import multiprocessing
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest
import numpy as np

try:
  import jax

  # Full-precision fp32 matmuls: the parent-process reference may run on GPU
  # dev machines, where TF32 would exceed the comparison tolerance.
  jax.config.update("jax_default_matmul_precision", "highest")
  HAS_JAX = True
except ImportError:
  HAS_JAX = False

_NPROCS = 2


def _make_fitted_estimator(is_classifier):
  """Builds a tiny fitted estimator + test rows. Deterministic everywhere.

  Imports are lazy so spawned workers can configure JAX (platform, device
  count, distributed runtime) before its first use.
  """
  from flax import nnx  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from tabfm.src.classifier_and_regressor import TabFMClassifier  # pylint: disable=g-import-not-at-top
  from tabfm.src.classifier_and_regressor import TabFMRegressor  # pylint: disable=g-import-not-at-top
  from tabfm.src.jax import model as tabfm_model  # pylint: disable=g-import-not-at-top

  model = tabfm_model.TabFM(
      loss="cross_entropy" if is_classifier else "mse",
      max_classes=2,
      embed_dim=8,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=8,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=1,
      icl_num_blocks=1,
      icl_nhead=2,
      dtype=jnp.float32,
      rngs=nnx.Rngs(0),
  )
  n_train, n_test = 21, 9
  rng = np.random.RandomState(0)
  X = rng.rand(n_train + n_test, 4)
  if is_classifier:
    y = rng.randint(0, 2, size=n_train)
    est = TabFMClassifier(model=model, n_estimators=2, random_state=0)
  else:
    y = X[:n_train] @ rng.rand(4)
    est = TabFMRegressor(model=model, n_estimators=2, random_state=0)
  est.fit(X[:n_train], y)
  return est, X[n_train:]


def _worker(pid, port, is_classifier, q):
  """One simulated host: joins the distributed runtime, runs the predict."""
  os.environ["JAX_PLATFORMS"] = "cpu"
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
  import jax  # pylint: disable=g-import-not-at-top

  jax.distributed.initialize(
      coordinator_address=f"localhost:{port}",
      num_processes=_NPROCS,
      process_id=pid,
  )
  assert jax.process_count() == _NPROCS
  assert len(jax.devices()) == 2 * _NPROCS

  from tabfm.src.jax import seqpar  # pylint: disable=g-import-not-at-top

  est, x_test = _make_fitted_estimator(is_classifier)
  if is_classifier:
    out = seqpar.predict_proba(est, x_test)
  else:
    out = seqpar.predict(est, x_test)
  q.put((pid, out))  # every process returns the full predictions


@absltest.skipIf(not HAS_JAX, "JAX not installed")
class SeqparMultiprocessTest(absltest.TestCase):

  def _run(self, is_classifier):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    port = 21212 + int(is_classifier)
    procs = [
        ctx.Process(target=_worker, args=(pid, port, is_classifier, q))
        for pid in range(_NPROCS)
    ]
    for p in procs:
      p.start()
    results = dict(q.get(timeout=300) for _ in range(_NPROCS))
    for p in procs:
      p.join(timeout=300)
      self.assertEqual(p.exitcode, 0)

    # Reference: the estimator's plain (non-distributed) prediction path,
    # computed in this parent process.
    est, x_test = _make_fitted_estimator(is_classifier)
    ref = est.predict_proba(x_test) if is_classifier else est.predict(x_test)

    for pid in range(_NPROCS):
      np.testing.assert_allclose(results[pid], ref, rtol=1e-4, atol=1e-4)

  def test_regressor(self):
    self._run(is_classifier=False)

  def test_classifier(self):
    self._run(is_classifier=True)


if __name__ == "__main__":
  absltest.main()

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

"""Pytest configuration.

The test suite uses ``absltest.TestCase`` helpers (e.g. ``create_tempdir``),
which read absl flags such as ``--test_tmpdir``. Those flags are only parsed by
``absltest.main()``; under the pytest runner they are never parsed, raising
``UnparsedFlagAccessError``. Parse them here so absltest-based tests work under
pytest.
"""

import importlib.util
import sys

from absl import flags

# Backend-specific test modules import optional extras at import time (torch for
# the pytorch backend; jax/flax/chex for the jax backend). The CI "core tests"
# job runs `pip install -e .[dev]`, which pulls neither extra, so these modules
# would otherwise fail collection with ModuleNotFoundError. Skip them when their
# backend is not installed.
_has_torch = importlib.util.find_spec("torch") is not None
_has_jax = importlib.util.find_spec("jax") is not None
_has_mlx = importlib.util.find_spec("mlx") is not None
collect_ignore = []
if not _has_torch:
  collect_ignore.append("tabfm/src/classifier_and_regressor_pytorch_test.py")
if not _has_jax:
  collect_ignore += [
      "tabfm/src/jax/model_test.py",
      "tabfm/src/jax/checkpointing_test.py",
      "tabfm/src/jax/memory_efficient_attention_test.py",
  ]
if not _has_mlx:
  collect_ignore.append("tabfm/src/classifier_and_regressor_mlx_test.py")
# pytorch/model_test.py is a torch<->jax parity test: it imports both flax and
# torch, so it needs *both* backends installed.
if not (_has_torch and _has_jax):
  collect_ignore.append("tabfm/src/pytorch/model_test.py")
# mlx/model_test.py is a torch<->mlx parity test: it needs both backends.
if not (_has_torch and _has_mlx):
  collect_ignore.append("tabfm/src/mlx/model_test.py")


def pytest_configure(config):  # noqa: D401  (pytest hook)
  del config  # Unused.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1])

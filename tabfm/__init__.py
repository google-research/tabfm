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

"""tabfm API."""

try:
  from tabfm.src.jax import tabfm_v1_0_0 as tabfm_v1_0_0_jax
  tabfm_v1_0_0 = tabfm_v1_0_0_jax
except ImportError:
  # JAX is not installed or incomplete, tabfm_v1_0_0_jax is not available.
  pass

try:
  from tabfm.src.pytorch import tabfm_v1_0_0 as tabfm_v1_0_0_pytorch
except ImportError:
  # PyTorch is not installed or incomplete, tabfm_v1_0_0_pytorch is not available.
  pass

try:
  from tabfm.src.mlx import tabfm_v1_0_0 as tabfm_v1_0_0_mlx
except ImportError:
  # MLX is not installed or incomplete, tabfm_v1_0_0_mlx is not available.
  pass

from tabfm.src.classifier_and_regressor import TabFMClassifier, TabFMRegressor

# A new PyPI release will be pushed every time `__version__` is increased.
# When changing this, also update the CHANGELOG.md.
__version__ = '1.0.0'

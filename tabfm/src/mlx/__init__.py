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

"""MLX backend for TabFM (Apple silicon, no PyTorch/JAX required).

Mirrors tabfm.src.pytorch (module/param names identical) so weight conversion
is mechanical and numerical parity with the PyTorch backend can be verified
layer by layer.

NOTE: the default pretrained weights downloaded by tabfm_v1_0_0_mlx.load()
are distributed under the separate tabfm-non-commercial-v1.0 license
(non-commercial, non-production use only).
"""

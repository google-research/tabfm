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

import unittest

from tabfm.src.pytorch import model


class ConfigValidationTest(unittest.TestCase):

  def test_attention_rejects_non_divisible_model_dimension(self):
    with self.assertRaisesRegex(ValueError, 'must be divisible by nhead'):
      model.MultiheadAttention(d_model=7, nhead=2)

  def test_rope_rejects_dimension_below_two(self):
    for dim in (0, 1):
      with self.subTest(dim=dim):
        with self.assertRaisesRegex(ValueError, 'dim must be at least 2'):
          model.RoPE(dim=dim, base=10000.0)

  def test_transformer_activation_set_matches_jax(self):
    for activation in ('relu', 'gelu', 'swiglu'):
      with self.subTest(activation=activation):
        block = model.MultiheadAttentionBlock(
            d_model=8, nhead=2, dim_ff=16, activation=activation
        )
        self.assertIsNotNone(block)

    with self.assertRaisesRegex(ValueError, 'Activation must be one of'):
      model.MultiheadAttentionBlock(
          d_model=8, nhead=2, dim_ff=16, activation='silu'
      )

  def test_unknown_activation_has_clear_error(self):
    with self.assertRaisesRegex(ValueError, 'Activation must be one of'):
      model.get_activation('unknown')


if __name__ == '__main__':
  unittest.main()

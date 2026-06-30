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

import hashlib

from absl import logging
import chex
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tabfm.src.jax import model as tabfm_model
from tabfm.src.jax.model import AttentionImplementation, YEmbeddingScheme

from absl.testing import absltest
from absl.testing import parameterized


def _init_model_deterministically(model: nnx.Module):
  """Initializes model parameters independently of their creation order."""
  state = nnx.state(model, nnx.Param)
  master_key = jax.random.key(42)

  new_state_dict = {}
  flat_state = nnx.to_flat_state(state)
  for path, var in flat_state:
    # Derive a deterministic seed from the parameter name.
    logging.info('Initializing parameter: %s', '.'.join(str(p) for p in path))
    key = jax.random.fold_in(
        master_key,
        int(
            hashlib.sha256('.'.join(str(p) for p in path).encode('utf-8')).hexdigest(),
            16,
        )
        % (2**32),
    )
    # var.value.size)
    new_state_dict[path] = jax.random.normal(
        key, var.value.shape, var.value.dtype
    )

  nnx.update(model, nnx.State.from_flat_path(new_state_dict))


class TabFMDependenciesTest(parameterized.TestCase):
  """Tests the input/output dependencies of the TabFM model."""

  @parameterized.product(
      y_embedding_scheme=[
          YEmbeddingScheme.NONE,
          YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
      ],
      attention_impl=[
          AttentionImplementation.JAX
      ],
  )
  def test_call_dependencies_with_perturbation(
      self,
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
  ):
    # --- 1. Test Setup ---
    # Define small dimensions for the test case.
    B, T, H = 2, 8, 4  # Batch size, sequence length, number of features
    train_size = jnp.array([4, 5])
    max_classes = 5
    embed_dim = 6

    # Create a random key for reproducibility.
    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    # Instantiate the TabFM model with a minimal configuration.
    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        dtype=jnp.float32,
        col_attention_impl=attention_impl,
        row_attention_impl=AttentionImplementation.JAX,
        icl_attention_impl=attention_impl,
    )
    _init_model_deterministically(model)

    # Create dummy input tensors.
    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Baseline Output ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
      )

    # Compute the original output for comparison.
    original_output = get_logits(model, X, y, train_size)

    # --- 3. Perturbation Test for X (Features) ---
    delta = 1
    batch_idx = np.arange(B)
    # Iterate over every feature in the input tensor X.
    for b in range(B):
      for t in range(T):
        for h in range(H):
          logging.info('Perturbing feature %d, %d, %d', b, t, h)
          # Perturb a single element of X by a small amount.
          X_perturbed = X.at[b, t, h].add(delta)  # pylint: disable=invalid-name
          perturbed_output = get_logits(model, X_perturbed, y, train_size)
          output_diff = jnp.abs(perturbed_output - original_output)
          # Make sure the outputs for other batch elements didn't change.
          self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))
          # Consider a prediction changed if any of its logits changed.
          prediction_changed = np.any(output_diff[b, train_size[b]:, :] != 0,
                                      axis=-1)
          chex.assert_shape(prediction_changed, (T - train_size[b],))
          if t < train_size[b]:
            # Test 1: Perturbing a training feature should affect
            # ALL test predictions
            self.assertTrue(np.all(prediction_changed))
          else:
            # Test 2: Perturbing a test feature should ONLY affect its
            # own prediction.
            test_idx = t - train_size[b]
            # Check that the corresponding test output changed.
            self.assertTrue(
                prediction_changed[test_idx],
                'Perturbing a test feature should affect its own prediction.',
            )
            # Check that other test outputs did NOT change.
            mask = np.ones(T - train_size[b], dtype=bool)
            mask[test_idx] = False
            self.assertTrue(
                np.all(prediction_changed[mask] == False),   # pylint: disable=g-explicit-bool-comparison
                'Perturbing a test feature should not affect other'
                ' test predictions.',
            )

    # --- 4. Perturbation Test for y (Labels) ---
    # Iterate over every label in the input tensor y.
    for b in range(B):
      for t in range(T):
        # Perturb a single label by changing it to the next class index.
        y_perturbed = y.at[b, t].set((y[b, t] + 1) % max_classes)
        perturbed_output = get_logits(model, X, y_perturbed, train_size)
        output_diff = jnp.abs(perturbed_output - original_output)
        # Make sure the outputs for other batch elements didn't change.
        self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))
        # Consider a prediction changed if any of its logits changed.
        prediction_changed = np.any(output_diff[b, train_size[b]:, :] != 0,
                                    axis=-1)

        if t < train_size[b]:
          # Test 3: Perturbing a training label should affect ALL test predictions.
          self.assertTrue(
              np.all(prediction_changed),
              'Perturbing a training label should affect all test predictions.',
          )
        else:
          # Test 4: Perturbing a test label should affect NO test predictions.
          self.assertTrue(
              np.all(prediction_changed == False),  # pylint: disable=g-explicit-bool-comparison
              'Perturbing a test label should not affect any test predictions.',
          )

  def test_regression_add_y_to_x_post_embedding(self):
    B, T, H = 2, 8, 4
    train_size = jnp.array([4, 5])
    max_classes = 5
    embed_dim = 6

    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='rmse',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=2,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=1,
        rngs=rngs,
        y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
    )

    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.normal(y_key, (B, T))

    @nnx.jit
    def get_logits(model, x_input, y_input, train_size):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
      )

    output = get_logits(model, X, y, train_size)
    self.assertEqual(output.shape, (B, T, 1))
    self.assertFalse(jnp.isnan(output).any())

  @parameterized.product(
      y_embedding_scheme=[
          YEmbeddingScheme.NONE,
          YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
      ],
  )
  def test_call_dependencies_with_padding_perturbation(
      self,
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
  ):
    # --- 1. Test Setup ---
    B, T, H = 2, 16, 8  # Batch size, sequence length, number of features
    T_actual = jnp.array([4, 5])  # Actual sequence length before padding
    H_actual = jnp.array([3, 4])  # Actual number of features before padding
    max_classes = 5
    embed_dim = 6

    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model)

    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Baseline Output ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size, d):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
          d=d,
      )

    original_output = get_logits(model, X, y, T_actual, H_actual)

    # --- 3. Perturbation Test for X ---
    delta = 1.0
    batch_idx = np.arange(B)
    # Iterate over every element in the input tensor X.
    for b in range(B):
      for t in range(T):
        for h in range(H):
          X_perturbed = X.at[b, t, h].add(delta)
          perturbed_output = get_logits(
              model, X_perturbed, y, T_actual, H_actual
          )
          output_diff = jnp.abs(perturbed_output - original_output)

          # Make sure the outputs for other batch elements didn't change.
          self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))

          # Indices of test samples for the current batch element.
          test_indices = np.arange(T_actual[b], T)
          # Consider a prediction changed if any of its logits changed.
          prediction_changed = np.any(
              output_diff[b, test_indices, :] != 0, axis=-1
          )
          chex.assert_shape(prediction_changed, (T - T_actual[b],))

          is_padding_h = h >= H_actual[b]
          is_train = t < T_actual[b]

          if is_padding_h:
            logging.info('Perturbing padded H element %d, %d, %d', b, t, h)
            # Assert that NO output changed for any batch element.
            self.assertTrue(
                np.all(output_diff == 0),
                f'Perturbing padded element {b}, {t}, {h} changed the output.',
            )
          elif is_train:
            logging.info('Perturbing training element %d, %d, %d', b, t, h)
            # Perturbing a training feature should affect ALL test predictions.
            self.assertTrue(
                np.all(prediction_changed),
                f'Perturbing train X[{b}, {t}, {h}] did not affect all test'
                ' predictions.'
                f'\n X: {X}'
                f'\n X_perturbed: {X_perturbed}'
                f'\n original_output: {original_output}'
                f'\n perturbed_output: {perturbed_output}',
            )
          else:  # Test element (t >= T_actual[b])
            logging.info('Perturbing test element %d, %d, %d', b, t, h)
            # Perturbing a test feature should ONLY affect its own prediction.
            test_idx = t - T_actual[b]
            self.assertTrue(
                prediction_changed[test_idx],
                f'Perturbing test X[{b}, {t}, {h}] should affect its own'
                ' prediction.',
            )
            mask = np.ones(T - T_actual[b], dtype=bool)
            mask[test_idx] = False
            self.assertTrue(
                np.all(~prediction_changed[mask]),
                f'Perturbing test X[{b}, {t}, {h}] should not affect other'
                ' test predictions.',
            )

    # --- 4. Perturbation Test for y (Labels) ---
    for b in range(B):
      for t in range(T):
        y_perturbed = y.at[b, t].set((y[b, t] + 1) % max_classes)
        perturbed_output = get_logits(model, X, y_perturbed, T_actual, H_actual)
        output_diff = jnp.abs(perturbed_output - original_output)

        # Make sure the outputs for other batch elements didn't change.
        self.assertTrue(np.all(output_diff[batch_idx != b, :, :] == 0))

        test_indices = np.arange(T_actual[b], T)
        prediction_changed = np.any(
            output_diff[b, test_indices, :] != 0, axis=-1
        )
        chex.assert_shape(prediction_changed, (T - T_actual[b],))

        if t < T_actual[b]:
          # Perturbing a training label should affect ALL test predictions.
          self.assertTrue(
              np.all(prediction_changed),
              f'Perturbing train y[{b}, {t}] did not affect all test'
              ' predictions.',
          )
        else:
          # Perturbing a test label should affect NO test predictions.
          self.assertTrue(
              np.all(~prediction_changed),
              f'Perturbing test y[{b}, {t}] should not affect any test'
              ' predictions.',
          )

  @parameterized.product(
      y_embedding_scheme=[
          YEmbeddingScheme.NONE,
          YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
      ],
  )
  def test_padding_invariance(
      self,
      y_embedding_scheme: YEmbeddingScheme = YEmbeddingScheme.NONE,
  ):
    # --- 1. Test Setup ---
    B, T = 2, 16
    H_actual = jnp.array([3, 4])  # Actual number of features
    H1, H2 = 8, 12  # Different padding sizes
    T_actual = jnp.array([4, 5])  # Train sizes
    max_classes = 5
    embed_dim = 6

    key = jax.random.PRNGKey(1)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=3,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=3,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model)

    X_key, y_key = jax.random.split(data_key)
    # Common non-padded data
    X_core = jax.random.normal(X_key, (B, T, np.max(H_actual)))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Create Inputs with Different Padding ---
    X1 = jnp.zeros((B, T, H1))
    X2 = jnp.zeros((B, T, H2))

    for b in range(B):
      X1 = X1.at[b, :, :H_actual[b]].set(X_core[b, :, :H_actual[b]])
      X2 = X2.at[b, :, :H_actual[b]].set(X_core[b, :, :H_actual[b]])
      # Add some noise to padding to make sure it's not just zeros
      X1 = X1.at[b, :, H_actual[b]:].add(
          jax.random.normal(X_key, (T, (H1 - H_actual[b]).item()))
      )
      X2 = X2.at[b, :, H_actual[b]:].add(
          jax.random.normal(X_key, (T, (H2 - H_actual[b]).item()))
      )

    # --- 3. Compute Outputs ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size, d):
      return model(
          X=x_input,
          y=y_input,
          train_size=train_size,
          d=d,
      )

    output1 = get_logits(model, X1, y, T_actual, H_actual)
    output2 = get_logits(model, X2, y, T_actual, H_actual)

    # --- 4. Assertion ---
    # The outputs should be identical since the non-padding part is the same
    np.testing.assert_array_almost_equal(output1, output2, decimal=5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NONE',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'expected_output': [[
              [2.35432, 0.403802, -1.134074],
              [1.628627, 0.455984, -1.079756],
              [2.470968, 0.44621, 0.312044],
              [2.584721, 0.455619, 0.324722],
          ]],
      },
      {
          'testcase_name': 'ADD',
          'y_embedding_scheme': YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
          'expected_output': [[
              [2.1624, 0.405149, -1.114775],
              [1.543, 0.468879, -1.069826],
              [2.62295, 0.459419, 0.371579],
              [2.715013, 0.465245, 0.365446],
          ]],
      },
      {
          'testcase_name': 'activation_relu',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'activation': 'relu',
          'expected_output': [[
              [2.314818, 0.415296, -1.289602],
              [0.24321, 0.901325, -1.016127],
              [3.422867, 0.42615, 1.615581],
              [3.503174, 0.423493, 1.591408],
          ]],
      },
      {
          'testcase_name': 'activation_swiglu',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'activation': 'swiglu',
          'expected_output': [[
              [2.12426, 0.439869, -1.233905],
              [1.812752, 0.483976, -1.207603],
              [-1.323864, 1.370978, -0.101007],
              [-1.358536, 1.385991, -0.088265],
          ]],
      },
      {
          'testcase_name': 'feature_group_true',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'feature_group': True,
          'expected_output': [[
              [0.891614, 0.604129, -0.962587],
              [0.867184, 0.612097, -0.963016],
              [0.596546, 0.412646, 0.349397],
              [0.639643, 0.403368, 0.3539],
          ]],
      },
      {
          'testcase_name': 'no_fourier_features',
          'y_embedding_scheme': YEmbeddingScheme.NONE,
          'use_fourier_features': False,
          'expected_output': [[
              [0.813858, 0.62535, -0.945271],
              [0.74903, 0.641479, -0.927889],
              [1.282514, 0.353903, 0.455361],
              [0.966555, 0.363276, 0.374227],
          ]],
      },
  )
  def test_golden_output(
      self,
      y_embedding_scheme: YEmbeddingScheme,
      expected_output: list[list[list[float]]],
      attention_impl: AttentionImplementation = AttentionImplementation.JAX,
      activation: str = 'gelu',
      feature_group: bool = False,
      use_fourier_features: bool = True,
      icl_num_blocks: int = 1,
  ):
    # --- 1. Test Setup ---
    B, T, H = 1, 4, 3  # Batch size, sequence length, number of features
    train_size = jnp.array([2])
    max_classes = 3
    embed_dim = 4

    key = jax.random.PRNGKey(42)
    init_key, data_key = jax.random.split(key)
    rngs = nnx.Rngs(params=init_key)

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=1,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=1,
        row_num_cls=1,
        icl_num_blocks=icl_num_blocks,
        icl_nhead=1,
        rngs=rngs,
        y_embedding_scheme=y_embedding_scheme,
        icl_attention_impl=attention_impl,
        dtype=jnp.float32,
        activation=activation,
        feature_group=feature_group,
        use_fourier_features=use_fourier_features,

    )
    _init_model_deterministically(model)

    X_key, y_key = jax.random.split(data_key)
    X = jax.random.normal(X_key, (B, T, H))
    y = jax.random.randint(y_key, (B, T), 0, max_classes)

    # --- 2. Compute Output ---
    @nnx.jit
    def get_logits(model, x_input, y_input, train_size):
      return model(
          X=x_input,
          y=y_input,
          d=jnp.array([H]),
          train_size=train_size,
      )

    output = get_logits(model, X, y, train_size)
    logging.info('test_golden_output: %s', output)





  def test_head_sharding_consistency(self):
    # Setup minimal dimensions.
    # Setup mesh dimensions based on available devices.
    num_devices = jax.device_count()
    if num_devices >= 2:
      heads_dim = 2
      data_dim = num_devices // 2
    else:
      heads_dim = 1
      data_dim = 1

    devices = np.array(jax.devices()).reshape(data_dim, heads_dim)
    mesh = jax.sharding.Mesh(devices, ('data', 'heads'))

    # Ensure B and nhead are multiples of mesh dimensions.
    B = data_dim * 1
    T, H = 2, 2
    nhead = heads_dim * 2
    max_classes = 2
    embed_dim = nhead * 4

    key = jax.random.PRNGKey(0)
    init_key, data_key = jax.random.split(key)

    # 1. Baseline model (non-sharded)
    model_baseline = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=nhead,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=nhead,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=nhead,
        rngs=nnx.Rngs(params=init_key),
        # shard_attention_over_heads=False,
        col_attention_impl=AttentionImplementation.JAX,
        row_attention_impl=AttentionImplementation.JAX,
        icl_attention_impl=AttentionImplementation.JAX,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model_baseline)

    # 2. Sharded model
    model_sharded = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        col_nhead=nhead,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=nhead,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=nhead,
        rngs=nnx.Rngs(params=init_key),
        # shard_attention_over_heads=True,
        col_attention_impl=AttentionImplementation.JAX,
        row_attention_impl=AttentionImplementation.JAX,
        icl_attention_impl=AttentionImplementation.JAX,
        dtype=jnp.float32,
    )
    # Ensure both models have the exact same weights.
    nnx.update(model_sharded, nnx.state(model_baseline))
    chex.assert_trees_all_close(
        nnx.state(model_baseline), nnx.state(model_sharded), atol=1e-8
    )

    data_sharding = jax.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(
            'data',
        ),
    )

    def run_model(model, X, y, train_size):
      return model(X=X, y=y, train_size=train_size)

    with mesh:
      X = jax.device_put(jax.random.normal(data_key, (B, T, H)), data_sharding)
      y = jax.device_put(jnp.zeros((B, T), dtype=jnp.int32), data_sharding)
      train_size = jax.device_put(jnp.array([2] * B), data_sharding)
      output_baseline = run_model(model_baseline, X, y, train_size)
      output_sharded = run_model(model_sharded, X, y, train_size)

    # If running on a single device, they MUST be exactly identical.
    # On multiple devices, allow for some numerical drift due to distributed computation.
    atol = 1e-6
    chex.assert_trees_all_close(output_baseline, output_sharded, atol=atol)


class CellEmbedderInformationFlowTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.B = 2
    self.T = 4
    self.H = 8
    self.max_classes = 5
    self.embed_dim = 6
    self.y_embedding_scheme = YEmbeddingScheme.NONE
    self.y_col_embedder_encoder_nhid = 6

  @parameterized.product(
      feature_group=[True, False],
      use_cat_mask=[True, False],
      use_d=[True, False],
  )
  def test_information_flow(self, feature_group, use_cat_mask, use_d):
    key = jax.random.PRNGKey(42)
    init_key, data_key, perturb_key = jax.random.split(key, 3)
    rngs = nnx.Rngs(params=init_key)
    feature_group_size = 3 if feature_group else 1

    model = tabfm_model.CellEmbedder(
        embed_dim=self.embed_dim,
        y_embedding_scheme=self.y_embedding_scheme,
        max_classes=self.max_classes,
        y_col_embedder_encoder_nhid=self.y_col_embedder_encoder_nhid,
        is_classifier=True,
        feature_group=feature_group,
        feature_group_size=feature_group_size,
        rngs=rngs,
    )
    _init_model_deterministically(model)

    X = jax.random.normal(data_key, (self.B, self.T, self.H))
    y = jax.random.randint(data_key, (self.B, self.T,), 0, self.max_classes)

    if use_d:
      # Use different d for each example in batch. Example 0: d=6, Example 1: d=7.
      d = jnp.array([6, 7])
    else:
      d = None

    if use_cat_mask:
      cat_mask = jax.random.bernoulli(data_key, 0.5, (self.B, self.H))
    else:
      cat_mask = None

    # Output baseline
    out = model(X, y, cat_mask=cat_mask, d=d) # Shape (B, T, H, E)

    # 1. Perturb input column h_in within the active range (h_in < min(d))
    h_in = 2
    perturbation = jax.random.normal(perturb_key, (self.B, self.T)) * 0.1
    perturbation = jnp.where(jnp.abs(perturbation) < 1e-4, 0.1, perturbation)
    X_perturbed = X.at[:, :, h_in].add(perturbation)

    # Output perturbed
    out_perturbed = model(X_perturbed, y, cat_mask=cat_mask, d=d)

    # Determine which output indices should change
    # Note: if d is used, modulo is by d, not H.
    def get_expected_changed(b_idx):
      d_val = d[b_idx].item() if use_d else self.H
      if h_in >= d_val:
        return set()
      if not feature_group:
        return {h_in}
      offsets = [(2**i) - 1 for i in range(feature_group_size)]
      return {(h_in - offset) % d_val for offset in offsets}

    # Verify output changes for each example in batch
    for b in range(self.B):
      expected_changed = get_expected_changed(b)
      d_limit = d[b] if use_d else self.H

      for h in range(self.H):
        diff = jnp.abs(out[b, :, h, :] - out_perturbed[b, :, h, :])
        max_diff = jnp.max(diff)

        if h >= d_limit:
          # Padding features should always be 0
          self.assertEqual(jnp.max(jnp.abs(out[b, :, h, :])), 0.0)
          self.assertEqual(max_diff, 0.0, f"Expected padded output column {h} in example {b} to remain zero.")
        elif h in expected_changed:
          self.assertGreater(max_diff, 1e-5, f"Expected output column {h} in example {b} to change, but it didn't.")
        else:
          self.assertEqual(max_diff, 0.0, f"Expected output column {h} in example {b} to remain unchanged, but it changed.")

    # 2. If using d, perturb a feature that is OUTSIDE the active range for at least one example
    if use_d:
      h_in_padded = 6 # >= d[0] (which is 6), so it shouldn't affect example 0
      X_perturbed_padded = X.at[:, :, h_in_padded].add(perturbation)
      out_perturbed_padded = model(X_perturbed_padded, y, cat_mask=cat_mask, d=d)

      # Example 0: h_in_padded >= d[0], so nothing should change
      diff0 = jnp.abs(out[0, :, :, :] - out_perturbed_padded[0, :, :, :])
      self.assertEqual(jnp.max(diff0), 0.0, "Perturbing a feature >= d[0] should not affect example 0.")

      # Example 1: h_in_padded < d[1] (which is 7), so something SHOULD change
      diff1 = jnp.abs(out[1, :, :, :] - out_perturbed_padded[1, :, :, :])
      self.assertGreater(jnp.max(diff1), 1e-5, "Perturbing a feature < d[1] should affect example 1.")


class TabFMPrefillDecodeTest(parameterized.TestCase):
  """Tests the consistency between full forward pass and prefill/decode."""

  def setUp(self):
    super().setUp()
    self.rngs = nnx.Rngs(0)

  @parameterized.product(
      cache_icl_input_only=[True, False],
  )
  def test_prefill_decode_consistency(self, cache_icl_input_only):
    # This test verifies that prefill + decode produces similar results to a
    # full forward pass for the test portion of the data.
    B, T, H = 1, 20, 4
    train_len = 10
    embed_dim = 32
    max_classes = 5

    model = tabfm_model.TabFM(
        loss='cross_entropy',
        max_classes=max_classes,
        embed_dim=embed_dim,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=2,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        rngs=self.rngs,
        cache_icl_input_only=cache_icl_input_only,
        dtype=jnp.float32,
    )
    _init_model_deterministically(model)
    # Scale down weights to avoid chaotic numerical noise amplification on TPU.
    state = nnx.state(model, nnx.Param)
    flat_state = nnx.to_flat_state(state)
    new_state_dict = {}
    for path, var in flat_state:
      new_state_dict[path] = var.value * 0.02
    nnx.update(model, nnx.State.from_flat_path(new_state_dict))

    X = jax.random.normal(self.rngs.params(), (B, T, H))
    y = jax.random.randint(self.rngs.params(), (B, T), 0, max_classes)
    train_size = jnp.array([train_len])

    # 1. Full forward pass — reference output on the test portion.
    out_full = model(X, y, train_size)
    out_full_test = out_full[:, train_len:, :]

    # 2. Prefill + Decode split pass.
    X_train = X[:, :train_len, :]
    y_train = y[:, :train_len]
    X_test = X[:, train_len:, :]

    _, cache = model.prefill(X_train, y_train)

    # Verify the cache has the expected structure.
    icl_cache = cache['icl']
    self.assertIsInstance(icl_cache, tabfm_model.ICLearningCache)

    out_decode = model.decode(X_test, cache)

    # The split prefill+decode path should produce results close to the full pass.
    chex.assert_trees_all_close(out_full_test, out_decode, atol=0.2, rtol=0.1)


if __name__ == '__main__':
  absltest.main()

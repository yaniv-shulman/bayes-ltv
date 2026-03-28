from typing import Dict

import numpy as np
import tensorflow as tf

from models import ltv as target


def test_rbf_kernel_layer_returns_square_kernel_with_positive_diagonal() -> None:
    layer = target.RBFKernelLayer(
        kernel_amplitude=1.0,
        kernel_length_scale=2.0,
        num_timesteps=5,
        eps=1e-4,
    )

    actual: np.ndarray = layer(tf.zeros((1, 1))).numpy()

    assert actual.shape == (5, 5)
    assert np.all(np.diag(actual) > 0.0)
    np.testing.assert_allclose(actual, actual.T, atol=1e-6)


def test_prior_block_covariance_layer_builds_block_diagonal_matrix() -> None:
    kernel: tf.Tensor = tf.constant([[1.0, 0.2], [0.2, 1.5]], dtype=tf.float32)
    layer = target.PriorBlockCovarianceLayer(num_taps=2)

    actual: np.ndarray = layer(kernel).numpy()
    expected: np.ndarray = np.block(
        [
            [kernel.numpy(), np.zeros((2, 2))],
            [np.zeros((2, 2)), kernel.numpy()],
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_get_ltv_model_forward_and_params_shapes() -> None:
    model = target.get_ltv_model(
        input_length=4,
        init_filters=2,
        num_taps=2,
        kernel_amplitude=1.0,
        kernel_length_scale=1.0,
        initial_learning_rate=0.001,
        target_learning_rate=0.001,
        warmup_steps=0,
        epochs=1,
        kl_scaler=1.0,
    )

    inputs: Dict[str, tf.Tensor] = {
        "net_inputs": tf.zeros((1, 4, 1), dtype=tf.float32),
        "conv_inputs": tf.ones((1, 4, 2), dtype=tf.float32),
    }

    actual_pred: tf.Tensor = model(inputs)
    actual_mean, actual_std = model.params(inputs)

    assert tuple(actual_pred.shape) == (1, 4)
    assert tuple(actual_mean.shape) == (1, 4, 2)
    assert tuple(actual_std.shape) == (1, 4, 2)

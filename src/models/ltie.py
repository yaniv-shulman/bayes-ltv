import warnings
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras
from tensorflow_probability.python.layers import default_mean_field_normal_fn

warnings.filterwarnings(
    "ignore",
    message=r".*`layer\.add_variable` is deprecated and will be removed.*",
)


def make_custom_kernel_prior(
    kernel_size: int, alpha: Optional[float] = None
) -> Callable:
    """
    Creates a custom kernel prior function with standard deviation 1/sqrt(kernel_size).

    Args:
        kernel_size: The size of the convolutional kernel.
        alpha: The scale of the prior distribution. If None, std = 1.0. If provided, std = alpha / sqrt(kernel_size).

    Returns:
        A callable that returns a tfp.distributions.Distribution given the kernel shape.
    """
    std: float

    if alpha is None:
        std = 1.0
    else:
        std = alpha / np.sqrt(kernel_size)

    def custom_kernel_prior(dtype, shape, name, trainable, add_variable_fn):
        # We ignore name, trainable, and add_variable_fn here.

        return tfp.distributions.Independent(
            tfp.distributions.Normal(
                loc=tf.zeros(shape, dtype=dtype),
                scale=std * tf.ones(shape, dtype=dtype),
            ),
            reinterpreted_batch_ndims=len(shape),
        )

    return custom_kernel_prior


def get_ltie_model(
    kernel_size: int,
    initial_learning_rate: float,
    target_learning_rate: float,
    warmup_steps: int,
    epochs: int,
    num_examples_per_epoch: int,
    alpha: Optional[float] = None,
) -> tf_keras.Sequential:
    """
    Returns a Bayesian neural network model representing an LTIE channel.

    Args:
        kernel_size: The size of the convolutional kernel.
        initial_learning_rate: The initial learning rate.
        target_learning_rate: The target learning rate.
        warmup_steps: The number of warmup steps.
        epochs: The number of epochs.
        num_examples_per_epoch: The number of examples per epoch.
        alpha: The scale of the prior distribution. If None, std = 1.0. If provided, std = alpha / sqrt(kernel_size).

    Returns:
        The Bayesian neural network model.
    """
    custom_prior_fn = make_custom_kernel_prior(kernel_size, alpha=alpha)

    kernel_posterior_fn = default_mean_field_normal_fn(
        loc_initializer=tf.keras.initializers.RandomNormal(
            stddev=1.0 / np.sqrt(kernel_size), seed=42
        ),
        untransformed_scale_initializer=tf.keras.initializers.RandomNormal(
            mean=-3.0, stddev=0.1, seed=42
        ),
    )

    model: tf_keras.Sequential = tf_keras.Sequential(
        [
            tfp.layers.Convolution1DFlipout(
                filters=1,
                kernel_size=kernel_size,
                padding="valid",
                activation="linear",
                kernel_prior_fn=custom_prior_fn,
                kernel_posterior_fn=kernel_posterior_fn,
                bias_posterior_fn=None,
            ),
            tf_keras.layers.Flatten(),
        ]
    )

    decay_steps: int = epochs - warmup_steps

    lr_warmup_decayed_fn: tf_keras.optimizers.schedules.CosineDecay = (
        tf_keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            warmup_target=target_learning_rate,
            warmup_steps=warmup_steps,
        )
    )

    loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = (
        lambda y, y_hat: tf.sqrt(2 * tf.nn.l2_loss(y - y_hat))
        + model.losses[0] / num_examples_per_epoch
    )

    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=lr_warmup_decayed_fn),
        loss=loss,
    )

    return model

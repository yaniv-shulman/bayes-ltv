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


class ResidualObservationNoiseStdCallback(tf_keras.callbacks.Callback):
    """Update a non-trainable observation-noise scale from batch residuals.

    The callback samples examples from the repeated optimization batch, computes a residual standard deviation, and
    updates the model's non-trainable observation-noise standard deviation via an exponential moving average.
    Repeated batch rows are treated as a practical sampling pool rather than as statistically distinct observations.

    Args:
        x_train: Training inputs used by the current fit call.
        y_train: Training targets aligned with the model outputs.
        sample_size: Number of batch rows to sample for each residual update.
        ema_decay: Exponential moving-average decay in `[0, 1)`.
        min_std: Lower clipping bound for the estimated standard deviation.
        max_std: Optional upper clipping bound for the estimated standard deviation.
        seed: Random seed used for batch-row sampling.

    Raises:
        ValueError: If the sample size, EMA decay, or clipping bounds are
            invalid.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        sample_size: int,
        ema_decay: float,
        min_std: float,
        max_std: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        """
        A callback that updates the observation-noise standard deviation via an exponential moving average.

        Args:
            x_train: The observations used by the current fit call.
            y_train: The training targets aligned with the model outputs.
            sample_size: The number of batch rows to sample for each residual update.
            ema_decay: The exponential moving average decay in `[0, 1)`.
            min_std: The minimum clipping bound for the estimated standard deviation.
            max_std: The maximum clipping bound for the estimated standard deviation.
            seed: The random seed used for batch-row sampling.
        """

        super().__init__()

        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1).")

        if min_std <= 0.0:
            raise ValueError("min_std must be positive.")

        if max_std is not None and max_std < min_std:
            raise ValueError("max_std must be greater than or equal to min_std.")

        self._x_train: np.ndarray = x_train
        self._y_train: np.ndarray = y_train
        self._sample_size: int = sample_size
        self._ema_decay: float = ema_decay
        self._min_std: float = min_std
        self._max_std: Optional[float] = max_std
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Update the observation-noise standard deviation after each epoch.

        Args:
            epoch: Epoch index supplied by Keras.
            logs: Optional Keras logs dictionary.

        Raises:
            ValueError: If the attached model does not expose an observation
                noise standard-deviation variable.
        """
        del epoch, logs

        observation_noise_std_variable: Optional[tf.Variable] = getattr(
            self.model,
            "observation_noise_std_variable",
            None,
        )

        if observation_noise_std_variable is None:
            raise ValueError(
                "ResidualObservationNoiseStdCallback requires the model to "
                "define observation_noise_std_variable."
            )

        sample_size: int = min(self._sample_size, self._x_train.shape[0])

        sampled_indices: np.ndarray = self._rng.choice(
            self._x_train.shape[0],
            size=sample_size,
            replace=False,
        )

        x_sample: np.ndarray = self._x_train[sampled_indices]
        y_sample: np.ndarray = self._y_train[sampled_indices]
        y_pred: np.ndarray = self.model(x_sample, training=False).numpy()
        y_sample_flat: np.ndarray = y_sample.reshape(y_sample.shape[0], -1)
        y_pred_flat: np.ndarray = y_pred.reshape(y_pred.shape[0], -1)

        residual_std: float = float(
            np.sqrt(np.mean(np.square(y_sample_flat - y_pred_flat)))
        )

        updated_std: float = (
            self._ema_decay * float(observation_noise_std_variable.numpy())
            + (1.0 - self._ema_decay) * residual_std
        )

        clipped_std: float = max(updated_std, self._min_std)

        if self._max_std is not None:
            clipped_std = min(clipped_std, self._max_std)

        observation_noise_std_variable.assign(clipped_std)


def get_ltie_model(
    kernel_size: int,
    initial_learning_rate: float,
    target_learning_rate: float,
    warmup_steps: int,
    epochs: int,
    num_independent_examples_per_epoch: Optional[int] = None,
    alpha: Optional[float] = None,
    observation_noise_std: Optional[float] = None,
    estimate_observation_noise_std_from_residuals: bool = False,
) -> tf_keras.Sequential:
    """
    Returns a Bayesian neural network model representing an LTIE channel.

    Args:
        kernel_size: The size of the convolutional kernel.
        initial_learning_rate: The initial learning rate.
        target_learning_rate: The target learning rate.
        warmup_steps: The number of warmup steps.
        epochs: The number of epochs.
        num_independent_examples_per_epoch: The number of statistically
            independent examples per epoch. Replicated rows used only to form a
            larger optimization batch should not be counted here.
        alpha: The scale of the prior distribution. If None, std = 1.0. If provided, std = alpha / sqrt(kernel_size).
        observation_noise_std: The observation noise standard deviation. If provided, the data-fit term is scaled
        according to a Gaussian observation model.
        estimate_observation_noise_std_from_residuals: Whether to maintain a non-trainable observation-noise scale via
            residual-based updates outside backpropagation.

    Returns:
        The Bayesian neural network model.

    Raises:
        ValueError: If the independent-example count is not specified exactly
            once or is not positive.
    """
    if num_independent_examples_per_epoch is None:
        raise ValueError("num_independent_examples_per_epoch must be provided.")

    if num_independent_examples_per_epoch <= 0:
        raise ValueError("num_independent_examples_per_epoch must be positive.")

    if observation_noise_std is not None and observation_noise_std <= 0.0:
        raise ValueError("observation_noise_std must be positive when provided.")

    if (
        observation_noise_std is not None
        and estimate_observation_noise_std_from_residuals
    ):
        raise ValueError(
            "estimate_observation_noise_std_from_residuals must be False when observation_noise_std is provided."
        )

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

    observation_noise_std_variable: Optional[tf.Variable] = None

    if (
        estimate_observation_noise_std_from_residuals
        or observation_noise_std is not None
    ):
        initial_observation_noise_std: float = (
            observation_noise_std if observation_noise_std is not None else 1.0
        )

        observation_noise_std_variable = tf.Variable(
            initial_value=initial_observation_noise_std,
            trainable=False,
            dtype=tf.as_dtype(tf_keras.backend.floatx()),
            name="observation_noise_std",
        )

        model.observation_noise_std_variable = observation_noise_std_variable

    def loss(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
        """Return a single-example ELBO-style loss estimate."""
        y = tf.reshape(y, tf.shape(y_hat))

        residual: tf.Tensor = tf.reshape(
            y - y_hat,
            (tf.shape(y_hat)[0], -1),
        )

        squared_error_per_example: tf.Tensor = tf.reduce_sum(
            tf.square(residual), axis=1
        )

        expected_data_fit: tf.Tensor = tf.reduce_mean(squared_error_per_example)
        observation_scale: tf.Tensor

        if observation_noise_std_variable is None:
            observation_scale = tf.cast(1.0, dtype=y_hat.dtype)
        else:
            observation_scale = tf.cast(
                0.5 / tf.square(observation_noise_std_variable),
                dtype=y_hat.dtype,
            )

        return (
            observation_scale * expected_data_fit
            + model.losses[0] / num_independent_examples_per_epoch
        )

    model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=lr_warmup_decayed_fn),
        loss=loss,
    )

    return model

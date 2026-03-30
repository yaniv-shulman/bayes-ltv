from typing import Optional

import numpy as np
import pytest
import tensorflow as tf

from models import ltie as target


def test_make_custom_kernel_prior_uses_expected_scale() -> None:
    prior_fn = target.make_custom_kernel_prior(kernel_size=16, alpha=2.0)
    distribution = prior_fn(
        dtype=tf.float32,
        shape=(2, 3),
        name="kernel",
        trainable=False,
        add_variable_fn=None,
    )

    actual: np.ndarray = distribution.stddev().numpy()
    expected: float = 2.0 / np.sqrt(16)
    np.testing.assert_allclose(actual, np.full((2, 3), expected))


@pytest.mark.parametrize(
    "sample_size,ema_decay,min_std,max_std,error_pattern",
    [
        (0, 0.9, 0.1, None, "sample_size must be positive"),
        (1, 1.0, 0.1, None, "ema_decay must lie in \\[0, 1\\)"),
        (1, 0.9, 0.0, None, "min_std must be positive"),
        (1, 0.9, 0.2, 0.1, "max_std must be greater than or equal to min_std"),
    ],
)
def test_residual_observation_noise_callback_invalid_init_raises(
    sample_size: int,
    ema_decay: float,
    min_std: float,
    max_std: Optional[float],
    error_pattern: str,
) -> None:
    x_train: np.ndarray = np.zeros((2, 3, 1))
    y_train: np.ndarray = np.zeros((2, 2, 1))

    with pytest.raises(ValueError, match=error_pattern):
        target.ResidualObservationNoiseStdCallback(
            x_train=x_train,
            y_train=y_train,
            sample_size=sample_size,
            ema_decay=ema_decay,
            min_std=min_std,
            max_std=max_std,
            seed=0,
        )


def test_residual_observation_noise_callback_updates_and_clips_std() -> None:
    x_train: np.ndarray = np.zeros((4, 3, 1), dtype=np.float32)
    y_train: np.ndarray = np.ones((4, 2, 1), dtype=np.float32)
    callback = target.ResidualObservationNoiseStdCallback(
        x_train=x_train,
        y_train=y_train,
        sample_size=2,
        ema_decay=0.5,
        min_std=0.1,
        max_std=1.4,
        seed=3,
    )

    class _Model:
        def __init__(self) -> None:
            self.observation_noise_std_variable = tf.Variable(
                2.0, trainable=False, dtype=tf.float32
            )

        def __call__(self, x_sample: np.ndarray, training: bool = False) -> tf.Tensor:
            del training
            return tf.zeros((x_sample.shape[0], 2, 1), dtype=tf.float32)

    model = _Model()
    callback.set_model(model)

    callback.on_epoch_end(epoch=0, logs=None)

    # residual std ~= 1.0, EMA from 2.0 with 0.5 gives 1.5, then clipped to max_std=1.4.
    assert float(model.observation_noise_std_variable.numpy()) == pytest.approx(1.4)


def test_residual_observation_noise_callback_requires_model_variable() -> None:
    callback = target.ResidualObservationNoiseStdCallback(
        x_train=np.zeros((2, 3, 1), dtype=np.float32),
        y_train=np.zeros((2, 2, 1), dtype=np.float32),
        sample_size=1,
        ema_decay=0.9,
        min_std=0.1,
        seed=1,
    )

    class _Model:
        def __call__(self, x_sample: np.ndarray, training: bool = False) -> tf.Tensor:
            del x_sample, training
            return tf.zeros((1, 2, 1), dtype=tf.float32)

    callback.set_model(_Model())

    with pytest.raises(
        ValueError, match="requires the model to define observation_noise_std_variable"
    ):
        callback.on_epoch_end(epoch=0, logs=None)


@pytest.mark.parametrize(
    "kwargs,error_pattern",
    [
        (
            {
                "kernel_size": 3,
                "initial_learning_rate": 0.001,
                "target_learning_rate": 0.001,
                "warmup_steps": 0,
                "epochs": 1,
                "num_independent_examples_per_epoch": None,
            },
            "must be provided",
        ),
        (
            {
                "kernel_size": 3,
                "initial_learning_rate": 0.001,
                "target_learning_rate": 0.001,
                "warmup_steps": 0,
                "epochs": 1,
                "num_independent_examples_per_epoch": 0,
            },
            "must be positive",
        ),
        (
            {
                "kernel_size": 3,
                "initial_learning_rate": 0.001,
                "target_learning_rate": 0.001,
                "warmup_steps": 0,
                "epochs": 1,
                "num_independent_examples_per_epoch": 1,
                "observation_noise_std": -1.0,
            },
            "observation_noise_std must be positive",
        ),
        (
            {
                "kernel_size": 3,
                "initial_learning_rate": 0.001,
                "target_learning_rate": 0.001,
                "warmup_steps": 0,
                "epochs": 1,
                "num_independent_examples_per_epoch": 1,
                "observation_noise_std": 0.5,
                "estimate_observation_noise_std_from_residuals": True,
            },
            "must be False when observation_noise_std is provided",
        ),
    ],
)
def test_get_ltie_model_invalid_inputs_raise(kwargs: dict, error_pattern: str) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        target.get_ltie_model(**kwargs)


def test_get_ltie_model_creates_observation_noise_variable_when_requested() -> None:
    model = target.get_ltie_model(
        kernel_size=3,
        initial_learning_rate=0.001,
        target_learning_rate=0.001,
        warmup_steps=0,
        epochs=1,
        num_independent_examples_per_epoch=1,
        estimate_observation_noise_std_from_residuals=True,
    )

    assert hasattr(model, "observation_noise_std_variable")
    assert float(model.observation_noise_std_variable.numpy()) == pytest.approx(1.0)

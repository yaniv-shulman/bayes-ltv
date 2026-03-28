import numpy as np
import pytest

from experiments.ant import ant_postprocessing as target


def test_posterior_phase_responses_empty_samples_raise() -> None:
    samples_impulse: np.ndarray = np.empty((0, 4))
    h_hat_mean: np.ndarray = np.ones(8, dtype=np.complex128)

    with pytest.raises(ValueError, match="No samples in phase_samples"):
        target.posterior_phase_responses(
            samples_impulse=samples_impulse,
            h_hat_mean=h_hat_mean,
            fs=20.0,
        )


def test_posterior_phase_responses_returns_expected_shapes() -> None:
    samples_impulse: np.ndarray = np.array(
        [[1.0, 0.0, -0.5, 0.25], [0.5, -0.25, 0.125, -0.0625]]
    )
    h_hat_mean: np.ndarray = np.exp(1j * np.linspace(0.0, 1.0, 512))

    (
        actual_samples,
        actual_mean,
        actual_median,
        actual_lower_ci,
        actual_upper_ci,
        actual_model_mean,
    ) = target.posterior_phase_responses(
        samples_impulse=samples_impulse,
        h_hat_mean=h_hat_mean,
        fs=20.0,
    )

    assert actual_samples.shape[0] == samples_impulse.shape[0]
    assert actual_mean.shape == actual_median.shape
    assert actual_mean.shape == actual_lower_ci.shape
    assert actual_mean.shape == actual_upper_ci.shape
    assert actual_model_mean.shape == h_hat_mean.shape


def test_calculate_naive_phase_velocity_sets_invalid_entries_to_nan() -> None:
    phase_sample: np.ndarray = np.array([0.0, -0.1, 0.1])
    actual: np.ndarray = target.calculate_naive_phase_velocity(
        phase_sample=phase_sample,
        distance=1.0,
        w_hat=np.array([1.0, 1.0, 1.0]),
        eps=1e-6,
    )

    assert np.isnan(actual[0])
    assert actual[1] > 0
    assert np.isnan(actual[2])


def test_posterior_phase_velocities_filters_low_velocity_samples() -> None:
    phase_samples: np.ndarray = np.array(
        [
            [-10.0, -10.0, -10.0],  # will be filtered for eps=1.0
            [-0.1, -0.1, -0.1],  # valid
        ]
    )
    w_hat: np.ndarray = np.array([0.5, 1.0, 2.0])
    signal_window_phase_response_mean: np.ndarray = np.array([-0.5, -0.5, -0.5])

    actual = target.posterior_phase_velocities(
        phase_samples=phase_samples,
        distance=1.0,
        eps=1.0,
        min_frequency=0.8,
        w_hat=w_hat,
        signal_window_phase_response_mean=signal_window_phase_response_mean,
    )

    actual_samples: np.ndarray = actual[0]
    assert actual_samples.shape[0] == 1
    assert np.all(np.isnan(actual[1]))


def test_posterior_phase_velocities_no_valid_samples_raise() -> None:
    phase_samples: np.ndarray = np.array([[-10.0, -10.0], [-20.0, -20.0]])
    w_hat: np.ndarray = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="No samples in phase_velocity_samples"):
        target.posterior_phase_velocities(
            phase_samples=phase_samples,
            distance=1.0,
            eps=1.0,
            min_frequency=0.5,
            w_hat=w_hat,
            signal_window_phase_response_mean=np.array([-1.0, -1.0]),
        )

from typing import Callable, List, Tuple

import numpy as np
import pytest

from experiments.ant import synthetic_data as target


def test_single_velocity_returns_expected_shapes() -> None:
    random_generator: np.random.Generator = np.random.default_rng(123)

    actual: List[Tuple[np.ndarray, np.ndarray]] = target.single_velocity(
        num_pairs=2,
        sequence_length=256,
        distance_rx=900.0,
        random_generator=random_generator,
        num_sources=16,
        noise_std=0.0,
    )

    assert len(actual) == 2
    for sig1, sig2 in actual:
        assert sig1.shape == (256,)
        assert sig2.shape == (256,)


def test_single_velocity_sinus_decaying_returns_expected_shapes() -> None:
    random_generator: np.random.Generator = np.random.default_rng(123)

    actual: List[Tuple[np.ndarray, np.ndarray]] = target.single_velocity_sinus_decaying(
        num_pairs=2,
        sequence_length=256,
        distance_rx=900.0,
        random_generator=random_generator,
        num_sources=12,
        pulse_length=20,
        noise_std=0.0,
    )

    assert len(actual) == 2
    for sig1, sig2 in actual:
        assert sig1.shape == (256,)
        assert sig2.shape == (256,)


def test_velocity_curve_sinus_decaying_constant_velocity_returns_callable_and_pairs() -> (
    None
):
    random_generator: np.random.Generator = np.random.default_rng(321)

    actual_pairs, actual_velocity_func = target.velocity_curve_sinus_decaying(
        num_pairs=2,
        sequence_length=256,
        distance_rx=900.0,
        freq_velocity_pairs=3000.0,
        random_generator=random_generator,
        num_sources=10,
        pulse_length=20,
        num_workers=1,
    )

    assert len(actual_pairs) == 2
    assert actual_pairs[0][0].shape == (256,)
    assert actual_pairs[0][1].shape == (256,)
    assert actual_velocity_func(1.0) == pytest.approx(3000.0)
    assert actual_velocity_func(10.0) == pytest.approx(3000.0)


def test_velocity_curve_sinus_decaying_variable_velocity_callable_changes_by_frequency() -> (
    None
):
    random_generator: np.random.Generator = np.random.default_rng(999)
    freq_velocity_pairs: List[Tuple[float, float]] = [
        (1.0, 2500.0),
        (5.0, 3000.0),
        (10.0, 3500.0),
    ]

    actual_pairs, actual_velocity_func = target.velocity_curve_sinus_decaying(
        num_pairs=1,
        sequence_length=256,
        distance_rx=900.0,
        freq_velocity_pairs=freq_velocity_pairs,
        random_generator=random_generator,
        num_sources=10,
        pulse_length=20,
        num_workers=1,
    )

    assert len(actual_pairs) == 1
    low_freq_velocity: float = float(actual_velocity_func(1.0))
    high_freq_velocity: float = float(actual_velocity_func(10.0))
    assert low_freq_velocity != pytest.approx(high_freq_velocity)


def test_velocity_curve_sinus_decaying_invalid_num_workers_raises() -> None:
    random_generator: np.random.Generator = np.random.default_rng(7)

    with pytest.raises(ValueError, match="num_workers must be positive"):
        target.velocity_curve_sinus_decaying(
            num_pairs=1,
            sequence_length=256,
            distance_rx=900.0,
            freq_velocity_pairs=3000.0,
            random_generator=random_generator,
            num_sources=8,
            pulse_length=20,
            num_workers=0,
        )


def test_velocity_curve_sinus_decaying_non_positive_num_pairs_raise() -> None:
    random_generator: np.random.Generator = np.random.default_rng(11)

    with pytest.raises(ValueError, match="num_pairs must be positive"):
        target.velocity_curve_sinus_decaying(
            num_pairs=0,
            sequence_length=256,
            distance_rx=900.0,
            freq_velocity_pairs=3000.0,
            random_generator=random_generator,
            num_sources=8,
            pulse_length=20,
            num_workers=1,
        )


def test_generate_velocity_curve_pair_missing_constant_velocity_raises() -> None:
    base_source_positions: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
    rx1: np.ndarray = np.array([0.0, 0.0])
    rx2: np.ndarray = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="constant_velocity must be provided"):
        target._generate_velocity_curve_sinus_decaying_pair(
            seed=1,
            sequence_length=256,
            base_source_positions=base_source_positions,
            rx1=rx1,
            rx2=rx2,
            variable_velocity=False,
            poly_coeffs=None,
            constant_velocity=None,
            sample_rate=20.0,
            noise_std=0.0,
            pulse_length=20,
            t_pulse=np.arange(20) / 20.0,
            margin=50,
        )


def test_generate_velocity_curve_pair_missing_poly_coeffs_raises() -> None:
    base_source_positions: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])
    rx1: np.ndarray = np.array([0.0, 0.0])
    rx2: np.ndarray = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="poly_coeffs must be provided"):
        target._generate_velocity_curve_sinus_decaying_pair(
            seed=1,
            sequence_length=256,
            base_source_positions=base_source_positions,
            rx1=rx1,
            rx2=rx2,
            variable_velocity=True,
            poly_coeffs=None,
            constant_velocity=None,
            sample_rate=20.0,
            noise_std=0.0,
            pulse_length=20,
            t_pulse=np.arange(20) / 20.0,
            margin=50,
        )

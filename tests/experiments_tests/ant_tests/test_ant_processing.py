from typing import List, Tuple

import numpy as np
import pytest

from experiments.ant import ant_processing as target


def test_calculate_psd_stats_db_empty_pairs_raises() -> None:
    with pytest.raises(ValueError, match="good_pairs list is empty"):
        target.calculate_psd_stats_db(good_pairs=[], fs=20.0)


def test_calculate_psd_stats_db_nperseg_too_large_raises() -> None:
    pair: Tuple[np.ndarray, np.ndarray] = (np.ones(8), np.ones(8))

    with pytest.raises(ValueError, match="nperseg > segment length"):
        target.calculate_psd_stats_db(good_pairs=[pair], fs=20.0, nperseg=16)


def test_calculate_psd_stats_db_returns_expected_shapes() -> None:
    pair_a: Tuple[np.ndarray, np.ndarray] = (
        np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125]),
        np.array([0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.05, -0.05]),
    )

    pair_b: Tuple[np.ndarray, np.ndarray] = (
        np.array([0.9, -0.9, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2]),
        np.array([0.2, -0.2, 0.15, -0.15, 0.1, -0.1, 0.05, -0.05]),
    )

    actual_source, actual_receiver = target.calculate_psd_stats_db(
        good_pairs=[pair_a, pair_b],
        fs=20.0,
        nperseg=8,
    )

    expected_keys: set[str] = {"freq", "psd_mean_db", "psd_std_db"}
    assert set(actual_source.keys()) == expected_keys
    assert set(actual_receiver.keys()) == expected_keys
    assert actual_source["freq"].shape == actual_source["psd_mean_db"].shape
    assert actual_source["freq"].shape == actual_source["psd_std_db"].shape
    assert actual_receiver["freq"].shape == actual_receiver["psd_mean_db"].shape
    assert actual_receiver["freq"].shape == actual_receiver["psd_std_db"].shape


def test_spectral_whiten_pairs_preserves_shapes() -> None:
    pair: Tuple[np.ndarray, np.ndarray] = (
        np.array([1.0, 0.0, -1.0, 0.5]),
        np.array([0.5, -0.25, 0.125, -0.0625]),
    )

    actual: List[Tuple[np.ndarray, np.ndarray]] = target.spectral_whiten_pairs([pair])

    assert len(actual) == 1
    assert actual[0][0].shape == pair[0].shape
    assert actual[0][1].shape == pair[1].shape
    assert np.all(np.isfinite(actual[0][0]))
    assert np.all(np.isfinite(actual[0][1]))


def test_one_bit_quantize_pairs_maps_to_signed_ones() -> None:
    pair: Tuple[np.ndarray, np.ndarray] = (
        np.array([-3.0, 0.0, 2.0]),
        np.array([-1.0, 1.0, 0.0]),
    )

    actual: List[Tuple[np.ndarray, np.ndarray]] = target.one_bit_quantize_pairs([pair])
    expected_sig1: np.ndarray = np.array([-1.0, 1.0, 1.0])
    expected_sig2: np.ndarray = np.array([-1.0, 1.0, 1.0])

    np.testing.assert_array_equal(actual[0][0], expected_sig1)
    np.testing.assert_array_equal(actual[0][1], expected_sig2)


def test_compute_cross_correlation_without_one_bit_matches_manual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.array([1.0, 2.0, -1.0, 0.5]), np.array([0.0, 1.0, -0.5, 2.0])),
        (np.array([-1.0, 0.5, 1.5, -0.5]), np.array([2.0, -1.0, 0.25, 0.0])),
    ]

    monkeypatch.setattr(target, "spectral_whiten_pairs", lambda pairs: pairs)

    # Arrange/Act/Assert
    actual_mean, actual_std = target.compute_cross_correlation(
        pairs=pairs,
        one_bit_quantization=False,
    )
    expected_per_pair: List[np.ndarray] = [
        np.fft.irfft(np.fft.rfft(pair[1]) * np.fft.rfft(pair[0]).conj())
        for pair in pairs
    ]
    expected_arr: np.ndarray = np.array(expected_per_pair)

    np.testing.assert_allclose(actual_mean, expected_arr.mean(axis=0))
    np.testing.assert_allclose(actual_std, expected_arr.std(axis=0))


@pytest.mark.parametrize(
    "num_independent_examples,batch_size_base,expected_batch,expected_repetitions",
    [
        (3, 8, 9, 3),
        (8, 8, 8, 1),
        (9, 8, 9, 1),
    ],
)
def test_compute_uniform_batch_repetitions_success(
    num_independent_examples: int,
    batch_size_base: int,
    expected_batch: int,
    expected_repetitions: int,
) -> None:
    actual_batch, actual_repetitions = target.compute_uniform_batch_repetitions(
        num_independent_examples=num_independent_examples,
        batch_size_base=batch_size_base,
    )

    assert actual_batch == expected_batch
    assert actual_repetitions == expected_repetitions


@pytest.mark.parametrize(
    "num_independent_examples,batch_size_base,error_pattern",
    [
        (0, 8, "num_independent_examples must be positive"),
        (3, 0, "batch_size_base must be positive"),
    ],
)
def test_compute_uniform_batch_repetitions_invalid_inputs_raise(
    num_independent_examples: int,
    batch_size_base: int,
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        target.compute_uniform_batch_repetitions(
            num_independent_examples=num_independent_examples,
            batch_size_base=batch_size_base,
        )


def test_repeat_examples_for_batch_repeats_along_first_axis() -> None:
    array: np.ndarray = np.array([[1.0, 2.0], [3.0, 4.0]])

    actual: np.ndarray = target.repeat_examples_for_batch(
        array=array, num_repetitions=3
    )
    expected: np.ndarray = np.array(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]
    )

    np.testing.assert_array_equal(actual, expected)


def test_repeat_examples_for_batch_non_positive_repetitions_raise() -> None:
    array: np.ndarray = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="num_repetitions must be positive"):
        target.repeat_examples_for_batch(array=array, num_repetitions=0)

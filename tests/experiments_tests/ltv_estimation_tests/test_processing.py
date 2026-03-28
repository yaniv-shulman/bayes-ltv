import numpy as np

from experiments.ltv_estimation import processing as target


def test_generate_non_lti_impulse_response_with_identical_filters_is_invariant() -> (
    None
):
    fir: np.ndarray = np.array([0.2, 0.4, 0.6])

    actual: np.ndarray = target.generate_non_lti_impulse_response(
        f1=fir,
        f2=fir,
        f3=fir,
        length=8,
    )

    expected: np.ndarray = np.tile(fir, (8, 1))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_convolve_non_lti_vectorized_matches_lti_convolution_for_constant_ir() -> None:
    signal: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    fir: np.ndarray = np.array([0.1, 0.2, 0.3])
    ir_series: np.ndarray = np.tile(fir, (len(signal), 1))

    actual: np.ndarray = target.convolve_non_lti_vectorized(
        signal=signal, ir_series=ir_series
    )
    expected: np.ndarray = np.convolve(signal, fir, mode="full")[: len(signal)]

    np.testing.assert_allclose(actual, expected)


def test_prepare_training_data_returns_expected_shapes_and_alignment() -> None:
    source: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    received_noisy: np.ndarray = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    actual_net_input, actual_conv_input, actual_y_true = target.prepare_training_data(
        source=source,
        received_noisy=received_noisy,
        num_taps=2,
        input_length=3,
        num_repeats=2,
    )

    assert actual_net_input.shape == (6, 3, 1)
    assert actual_conv_input.shape == (6, 3, 2)
    assert actual_y_true.shape == (6, 3)
    np.testing.assert_array_equal(actual_y_true[0], np.array([10.0, 11.0, 12.0]))
    np.testing.assert_array_equal(actual_conv_input[0, 0, :], np.array([1.0, 0.0]))
    np.testing.assert_array_equal(actual_conv_input[0, 1, :], np.array([2.0, 1.0]))


def test_stitch_local_fir_estimates_averages_overlaps() -> None:
    predicted_ir_windows: np.ndarray = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )

    actual: np.ndarray = target.stitch_local_fir_estimates(
        predicted_ir_windows=predicted_ir_windows,
        num_original_windows=2,
        source_len=4,
    )

    expected: np.ndarray = np.array(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [6.0, 60.0],
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_sanity_check_non_lti_convolution_returns_bool() -> None:
    actual = target.sanity_check_non_lti_convolution()

    assert isinstance(actual, bool)

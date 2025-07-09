from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from scipy.signal import firwin
from tqdm import tqdm


def generate_non_lti_impulse_response(
    f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, length: int
) -> np.ndarray:
    """
    Generates a non-LTI impulse response by smoothly interpolating between three FIRs.

    Args:
        f1: First FIR.
        f2: Second FIR.
        f3: Third FIR.
        length: Length of the FIR.

    Returns:
        Non-LTI impulse response.
    """
    t: np.ndarray = np.linspace(0, 1, length)

    # Use functions that are guaranteed to be non-negative
    w1: np.ndarray = 0.5 * (1 + np.cos(2 * np.pi * t))  # Peaks at start and end
    w2: np.ndarray = 1 - np.cos(2 * np.pi * t)  # Peaks in the middle
    w3: np.ndarray = np.sin(np.pi * t)  # Peaks in the middle

    # Stack and normalize using softmax for guaranteed convex combination
    weights: np.ndarray = np.stack([w1, w2, w3], axis=-1)
    # Applying a temperature to softmax can control the "sharpness" of transitions
    temperature: float = 0.5
    weights = tf.nn.softmax(weights / temperature, axis=-1).numpy()

    # Interpolate
    ir_series: np.ndarray = (
        weights[:, 0, np.newaxis] * f1
        + weights[:, 1, np.newaxis] * f2
        + weights[:, 2, np.newaxis] * f3
    )

    return ir_series


def convolve_non_lti_vectorized(
    signal: np.ndarray, ir_series: np.ndarray
) -> np.ndarray:
    """
    Performs a non-LTI convolution of a signal with a series of FIR filters.

    Args:
        signal: 1D input signal to convolve.
        ir_series: 2D array where each row is an FIR filter for a specific time step.

    Returns:
        The result of the non-LTI convolution as a 1D array.
    """
    n_signal: int = len(signal)
    n_taps: int = ir_series.shape[1]
    output: np.ndarray = np.zeros(n_signal)
    padded_signal: np.ndarray = np.pad(signal, (n_taps - 1, 0), "constant")
    i: int

    for i in range(n_signal):
        signal_segment: np.ndarray = padded_signal[i : i + n_taps]
        ir_at_i: np.ndarray = ir_series[i, :]
        output[i] = np.dot(signal_segment[::-1], ir_at_i)

    return output


def sanity_check_non_lti_convolution():
    """
    Performs a sanity check on the non-LTI convolution function.
    """
    # --- Example Usage ---
    # You would have your FIR filters defined here
    num_taps_main = 16
    f_linear_phase1 = firwin(
        numtaps=num_taps_main,
        cutoff=[0.05, 0.4, 0.6, 0.95],
        width=0.05,
        pass_zero=False,
    )

    f_linear_phase2 = firwin(
        numtaps=num_taps_main, cutoff=[0.2, 0.93], width=0.05, pass_zero=False
    )

    f_linear_phase3 = firwin(
        numtaps=num_taps_main, cutoff=[0.09, 0.89], width=0.05, pass_zero=False
    )

    print("--- Running Sanity Check for Non-LTI Convolution ---")

    # 1. Define simple, predictable impulse responses
    num_taps = 4
    # A simple ramp-up
    h1 = np.array([0.1, 0.2, 0.3, 0.4])
    # A simple ramp-down
    h2 = np.array([0.4, 0.3, 0.2, 0.1])
    # A constant value
    h3 = np.array([0.25, 0.25, 0.25, 0.25])

    # 2. Generate a short, non-LTI impulse response series
    # Let's use a length where we can easily see the interpolation
    test_length = 10
    test_ir_series = generate_non_lti_impulse_response(h1, h2, h3, test_length)

    print(f"Shape of the test impulse response series: {test_ir_series.shape}")

    # 3. Create a simple delta function as the input signal
    signal_length = 10
    delta_position = 3
    delta_signal = np.zeros(signal_length)
    delta_signal[delta_position] = 1.0

    print(f"Test signal: A delta function at index {delta_position}")
    print(delta_signal)

    # Perform convolution with the corrected function
    output_signal = convolve_non_lti_vectorized(delta_signal, test_ir_series)

    # 5. Define the expected output and check
    # The output should be the impulse response at each time `n`, but shifted by `delta_position`
    # and only appearing where the delta function "activates" it.
    # Specifically, output[n] should be h[n, n - delta_position]
    expected_output = np.zeros_like(output_signal)
    for n in range(delta_position, signal_length):
        tap_index = n - delta_position
        if tap_index < num_taps:
            expected_output[n] = test_ir_series[n, tap_index]

    # Due to the convolution's nature, the output will effectively be the impulse response
    # starting at the delta position.
    expected_simple = np.zeros_like(output_signal)
    len_to_copy = min(num_taps, signal_length - delta_position)

    # The output signal at index `delta_position` should be the first tap of the IR at that time.
    # output[delta_position] = h[delta_position, 0]
    # output[delta_position + 1] = h[delta_position + 1, 1]
    # ... and so on. This is complex.

    # A simpler check: The output signal from convolving with a delta at t=0 should be the IR itself.
    delta_at_zero = np.zeros(signal_length)
    delta_at_zero[0] = 1.0
    output_at_zero = convolve_non_lti_vectorized(delta_at_zero, test_ir_series)

    # The first `num_taps` elements of the output should match the first row of the IR series.
    # This is a key property. output[k] = h[k, k] for a delta at t=0.
    # Let's use a simpler, more intuitive check.

    print("\n--- Simplified Check ---")
    print("Convolving with a delta function at index 0...")

    # Expected output for a delta at index 0
    # output[n] = sum_{k=0}^{num_taps-1} delta[n-k] * h[n, k]
    # The only non-zero term is when k=n, so output[n] = h[n, n]
    expected_diag = np.diag(test_ir_series)

    # However, the standard definition of LTV convolution is y[n] = sum_k s[k]h[n, n-k]
    # If s[k] is a delta at k=0, then y[n] = h[n,n].
    # If s[k] is a delta at k=k_0, then y[n] = h[n, n-k_0].

    # The simplest test case is a TIME-INVARIANT impulse response.
    # If the IR does not change, the non-LTI convolution must equal the standard LTI convolution.
    print("\n--- LTI Invariance Check ---")
    lti_ir = np.tile(h1, (test_length, 1))  # Create a time-invariant IR series

    # Result from your function
    non_lti_result = convolve_non_lti_vectorized(delta_signal, lti_ir)

    # Result from standard numpy.convolve
    lti_result = np.convolve(delta_signal, h1, mode="full")[:test_length]

    print("Result from non-LTI convolution with LTI filter:")
    print(non_lti_result)
    print("\nResult from standard `np.convolve`:")
    print(lti_result)

    # Check if they are close
    is_close = np.allclose(non_lti_result, lti_result)
    if is_close:
        print(
            "\n✅ SUCCESS: The non-LTI convolution correctly matches standard convolution for an LTI case."
        )
    else:
        print(
            "\n❌ FAILURE: The non-LTI convolution does NOT match standard convolution for an LTI case."
        )

    return is_close


def prepare_training_data(
    source: np.ndarray,
    received_noisy: np.ndarray,
    num_taps: int,
    input_length: int,
    num_repeats: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data for a CNN model that estimates the LTV channel.

    Args:
        source: The source signal, a 1D numpy array.
        received_noisy: The received noisy signal, a 1D numpy array.
        num_taps: The number of taps in the FIR filter.
        input_length: The length of the input sequence for the CNN. This is the length of the truncated GP.
        num_repeats: The number of times to repeat the training data to increase the batch size.

    Returns:
        The network input (source signal), convolution input (source signal history for the convolution), and target
        output (received noisy signal).
    """
    training_examples: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []
    num_windows: int = len(source) - input_length + 1
    i: int

    # Loop through the signal to create each training window
    for i in tqdm(range(num_windows), desc="Preparing training data"):
        # The Target (`y_true` window). The output window the model needs to predict. Shape: (input_length,)
        target_window: np.ndarray = received_noisy[i : i + input_length]

        # The Network Input (`net_input` window) This is the window of the source signal that the CNN looks at.
        # It should be aligned with the target window. Shape: (input_length, 1)
        net_input_window: np.ndarray = source[i : i + input_length, np.newaxis]

        # The Convolution Input (`conv_input` window). This is the matrix of source signal history needed for the
        # convolution.
        conv_input_window: np.ndarray = np.zeros((input_length, num_taps))

        # Pad the source signal once at the beginning
        padded_source = np.pad(source, (num_taps - 1, 0), "constant")
        j: int

        for j in range(input_length):
            # The global time index is i + j
            # We need the signal from i+j backwards for num_taps
            conv_input_window[j, :] = padded_source[i + j : i + j + num_taps][::-1]

        # Append the aligned data to our lists
        training_examples.append(
            (
                {"net_inputs": net_input_window, "conv_inputs": conv_input_window},
                target_window,
            )
        )

    # Stack the lists into final numpy arrays
    net_input_list: List[np.ndarray] = [
        item[0]["net_inputs"] for item in training_examples
    ]

    conv_input_list: List[np.ndarray] = [
        item[0]["conv_inputs"] for item in training_examples
    ]

    y_true_list: List[np.ndarray] = [item[1] for item in training_examples]
    net_input: List[np.ndarray] = np.array(net_input_list)
    conv_input: List[np.ndarray] = np.array(conv_input_list)
    y_true: List[np.ndarray] = np.array(y_true_list)
    net_input = np.tile(net_input, (num_repeats, 1, 1))
    conv_input = np.tile(conv_input, (num_repeats, 1, 1))
    y_true = np.tile(y_true, (num_repeats, 1))

    return net_input, conv_input, y_true


def stitch_local_fir_estimates(
    predicted_ir_windows: np.ndarray, num_original_windows: int, source_len: int
) -> np.ndarray:
    """
    Stitch together local FIR estimates into a single, averaged impulse response.
    This function takes the predicted impulse response windows and combines them into a single
    impulse response that matches the length of the original source signal.

    It handles overlaps by averaging the contributions from each window.
    This is useful for cases where the model predicts FIRs in overlapping windows, and we want to
    create a single, coherent impulse response that reflects the overall behavior of the system.

    Args:
        predicted_ir_windows: A 3D numpy array of shape (num_windows, window_length, num_taps)
            where each slice along the first dimension is a predicted FIR for a specific time window.
        num_original_windows: The number of original windows in the source signal.
        source_len: The length of the original source signal.

    Returns:

    """
    num_windows: int
    window_length: int
    num_taps: int
    num_windows, window_length, num_taps = predicted_ir_windows.shape
    # The total length of the final, stitched impulse response series
    total_timesteps: int = num_original_windows + window_length - 1

    # Create arrays to hold the final stitched result and the counts for averaging
    stitched_mean_ir: np.ndarray = np.zeros((total_timesteps, num_taps))
    overlap_counts: np.ndarray = np.zeros((total_timesteps, num_taps))

    # Loop through each predicted window and add it to the final array.
    i: int

    for i in tqdm(
        range(num_windows),
        desc="Stitching overlapping window predictions by averaging...",
    ):
        stitched_mean_ir[i : i + window_length] += predicted_ir_windows[i, :, :]
        overlap_counts[i : i + window_length] += 1

    # Perform the averaging to get the final estimate
    # We add a small epsilon to avoid division by zero where there are no overlaps
    overlap_counts[overlap_counts == 0] = 1e-8
    stitched_mean_ir /= overlap_counts
    # Trim the stitched result to match the length of the original ground truth signal
    stitched_mean_ir = stitched_mean_ir[:source_len]
    return stitched_mean_ir

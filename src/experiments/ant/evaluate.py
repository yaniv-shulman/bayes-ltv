from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import pandas as pd
import tf_keras
from scipy.signal import hilbert
from scipy.special import j0
from tqdm import tqdm

from experiments.ant.ant_processing import (
    compute_cross_correlation,
    spectral_whiten_pairs,
)
from models.ltie import get_ltie_model


def compute_velocity_misfit(
    signal_window: np.ndarray,
    freqs: np.ndarray,
    distance: float,
    min_velocity: float,
    max_velocity: float,
    n_velocities: int,
) -> np.ndarray:
    """
    Compute the velocity misfit for a given signal window.

    The function computes the real FFT of the input signal window using an FFT length that produces
    exactly len(freqs) bins. It then normalizes the FFT and computes a theoretical beam pattern based
    on the Bessel function (j0) for a range of velocity candidates. The misfit is defined as the absolute
    difference between the beam pattern (transposed) and the real part of the normalized FFT.

    Parameters:
        signal_window: 1D array representing the signal window (e.g., an IR sample).
        freqs: 1D array of frequency values. The FFT will be computed to produce exactly len(freqs) frequency bins.
        distance: Receiver distance used in computing the beam pattern.
        min_velocity: Minimum velocity candidate.
        max_velocity: Maximum velocity candidate.
        n_velocities: Number of velocity candidates.

    Returns:
        np.ndarray: A 2D array representing the velocity misfit with shape (n_velocities, len(freqs)).
    """
    # Determine FFT length to get exactly len(freqs) bins.
    n_fft = 2 * (len(freqs) - 1)

    # Compute the FFT with the specified n_fft.
    f_signal = np.fft.rfft(signal_window, n=n_fft)
    # Normalize the FFT to preserve phase information.
    f_signal /= np.abs(f_signal)

    # Create the velocity axis.
    velocity_axis = np.linspace(min_velocity, max_velocity, n_velocities)

    # Compute the beam pattern using the Bessel function.
    # The beam pattern is computed for each frequency (rows) and each velocity candidate (columns).
    bf = j0(2 * np.pi * freqs[:, np.newaxis] * distance / velocity_axis)

    # Normalize the beam pattern using the Hilbert transform along the frequency axis.
    bf /= np.abs(hilbert(bf, axis=0))

    # Compute the misfit as the absolute difference between the beam pattern (transposed)
    # and the real part of the FFT. The beam pattern is transposed to have shape
    # (n_velocities, len(freqs)), matching f_signal.real.
    velocity_misfit = np.abs(bf.T - f_signal.real)
    return velocity_misfit


def compute_velocity_fit_statistics(
    velocity_misfit: np.ndarray,
    freqs: np.ndarray,
    min_velocity: float,
    max_velocity: float,
    n_velocities: int,
    velocity_true_or_func: Union[float, Callable[[float], float]],
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute velocity fit statistics from a velocity misfit map.

    This function analyzes a 2D misfit map that quantifies the difference between a modeled
    or observed response and a theoretical beam forming pattern over a grid defined by candidate
    velocities and frequencies. For each frequency (within an optional specified range), the function:

      1. Selects the candidate velocity (from a candidate velocity axis) that minimizes the misfit.
      2. Computes the error between these estimated velocities and the known true velocity. If a callable
         is provided for true_velocity_or_func, then the true velocity is computed as true_velocity_or_func(f)
         for each frequency.
      3. Returns the estimated velocity per frequency, along with the mean absolute error and the standard
         deviation of the estimation errors.

    Parameters:
        velocity_misfit: A 2D array of misfit values with shape (n_velocities, n_freqs), where each row corresponds to a
            candidate velocity (from the candidate velocity axis) and each column corresponds to a frequency
            as defined in `freqs`).
        freqs: A 1D array of frequency values.
        min_velocity: The minimum candidate velocity in the misfit map.
        max_velocity: The maximum candidate velocity in the misfit map.
        n_velocities: The number of candidate velocities in the misfit map.
        velocity_true_or_func: Either a constant true velocity or a function mapping frequency to the true velocity.
        min_freq: The minimum frequency (inclusive) to consider. If None, the minimum of `freqs` is used.
        max_freq: The maximum frequency (inclusive) to consider. If None, the maximum of `freqs` is used.

    Returns:
        est_vel: A 1D array of estimated velocities for each frequency in the selected range.
        mean_abs_error: The mean absolute error computed as the average absolute difference between the estimated
            velocities and the true velocity.
        std_error: The standard deviation of the velocity estimation errors.
    """
    # Set default frequency range if not provided.
    if min_freq is None:
        min_freq = freqs.min()

    if max_freq is None:
        max_freq = freqs.max()

    # Create a mask to filter frequencies within the desired range.
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    tmp_subset = velocity_misfit[:, freq_mask]
    selected_freqs = freqs[freq_mask]

    # Create candidate velocity axis.
    velocity_axis: np.ndarray = np.linspace(min_velocity, max_velocity, n_velocities)

    # For each frequency in the selected range, find the index of the candidate velocity that minimizes the misfit.
    min_idx = np.argmin(tmp_subset, axis=0)  # shape: (number of selected frequencies,)
    est_vel = velocity_axis[min_idx]

    # Compute the true velocity for each frequency.
    if callable(velocity_true_or_func):
        true_velocities = np.array([velocity_true_or_func(f) for f in selected_freqs])
    else:
        true_velocities = np.full_like(
            est_vel, fill_value=velocity_true_or_func, dtype=float
        )

    # Compute error vector.
    error = est_vel - true_velocities
    mean_abs_error = np.mean(np.abs(error))
    std_error = np.std(error)
    return est_vel, mean_abs_error, std_error


def aggregate_ground_truth_error(
    velocity_misfit: np.ndarray,
    freqs: np.ndarray,
    min_velocity: float,
    max_velocity: float,
    n_velocities: int,
    velocity_true_or_func: Union[float, Callable[[float], float]],
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    Aggregates misfit error information using either a constant true velocity or a frequency-dependent velocity curve.

    If a float is provided, that constant value is used as the true velocity for all frequency bins.
    If a callable is provided, then for each frequency bin the true velocity is computed from the callable,
    and the candidate velocity closest to that value is used to extract the misfit error.
    The errors are then aggregated (using the mean) over the selected frequency bins.

    Args:
        velocity_misfit (np.ndarray): 2D misfit array of shape (n_velocities, n_freq) where each row corresponds
                                      to the misfit for a candidate velocity across frequencies.
        freqs (np.ndarray): 1D array of frequency values.
        min_velocity (float): The minimum candidate velocity in the misfit map.
        max_velocity (float): The maximum candidate velocity in the misfit map.
        n_velocities (int): The number of candidate velocities in the misfit map.
        velocity_true_or_func (float or Callable[[float], float]): Either a constant true velocity
                                      or a function mapping frequency to true velocity.
        min_freq (Optional[float]): Minimum frequency to include in the aggregation.
                                    Defaults to the minimum frequency in `freqs` if None.
        max_freq (Optional[float]): Maximum frequency to include in the aggregation.
                                    Defaults to the maximum frequency in `freqs` if None.

    Returns:
        mean_error (float): The mean misfit error aggregated over the selected frequency bins,
                            using the candidate velocity closest to the true (or frequency-dependent) velocity.
        error_vector (np.ndarray): The misfit error vector at the candidate velocities closest to the ground truth,
                                   one error value per frequency bin.
    """
    # Set default frequency range if not provided.
    if min_freq is None:
        min_freq = freqs.min()
    if max_freq is None:
        max_freq = freqs.max()

    # Create a mask to select only the frequencies within the desired range.
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    tmp_subset = velocity_misfit[:, freq_mask]
    selected_freqs = freqs[freq_mask]

    # Create candidate velocity axis.
    velocity_axis: np.ndarray = np.linspace(min_velocity, max_velocity, n_velocities)

    # If a callable is provided, use frequency-dependent true velocity;
    # otherwise, use the constant true velocity.
    if callable(velocity_true_or_func):
        error_vector = np.array(
            [
                tmp_subset[
                    np.argmin(np.abs(velocity_axis - velocity_true_or_func(f))), i
                ]
                for i, f in enumerate(selected_freqs)
            ]
        )
    else:
        true_velocity = velocity_true_or_func
        idx = np.argmin(np.abs(velocity_axis - true_velocity))
        error_vector = tmp_subset[idx, :]

    mean_error = np.mean(error_vector)
    return mean_error, error_vector


def compute_posterior_velocity_misfit_stats(
    posterior_ir_samples: np.ndarray,
    freqs: np.ndarray,
    distance: float,
    min_velocity: float,
    max_velocity: float,
    n_velocities: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute posterior statistics for the velocity misfit from a set of posterior IR samples.

    This function takes a 2D matrix of posterior IR samples (each row is a sample) and, for each sample,
    computes a velocity misfit matrix by comparing the sample’s normalized Fourier transform with a beamforming
    pattern derived from a Bessel function. The velocity misfit for each sample is a 2D array with shape
    (n_velocities, len(freqs)). The function then aggregates these misfits over all samples and computes summary
    statistics:

      - Mean velocity misfit over samples.
      - Median velocity misfit over samples.
      - Lower confidence interval (2.5th percentile) for the misfit.
      - Upper confidence interval (97.5th percentile) for the misfit.

    Additionally, the velocity axis used in the computation (a linearly spaced vector between the specified
    min_velocity and max_velocity) is returned.

    Parameters:
        posterior_ir_samples (np.ndarray): 2D array of posterior IR samples with shape
                                             (n_samples, sample_length). Each row is a sample.
        freqs (np.ndarray): 1D array of frequency values used in the Fourier transform.
        distance (float): The receiver distance used in computing the beam pattern.
        min_velocity (float): Minimum velocity candidate for the misfit computation.
        max_velocity (float): Maximum velocity candidate for the misfit computation.
        n_velocities (int): Number of velocity candidates between min_velocity and max_velocity.

    Returns:
        misfits (np.ndarray): 3D array of velocity misfit matrices with shape
                              (n_samples, n_velocities, len(freqs)).
        mean_misfit (np.ndarray): Mean velocity misfit over samples, shape (n_velocities, len(freqs)).
        median_misfit (np.ndarray): Median velocity misfit over samples, shape (n_velocities, len(freqs)).
        lower_ci (np.ndarray): 2.5th percentile velocity misfit over samples, shape (n_velocities, len(freqs)).
        upper_ci (np.ndarray): 97.5th percentile velocity misfit over samples, shape (n_velocities, len(freqs)).
        velocity_axis (np.ndarray): 1D array of velocity candidates used in the computation, of length n_velocities.
    """
    # Create velocity axis.
    velocity_axis = np.linspace(min_velocity, max_velocity, n_velocities)

    # List to collect misfit matrices for each sample.
    misfits_list = []

    # Iterate over each posterior IR sample.
    for sample in tqdm(posterior_ir_samples, desc="Computing velocity misfits"):
        # Compute the velocity misfit for this sample.
        misfit = compute_velocity_misfit(
            signal_window=sample,
            freqs=freqs,
            distance=distance,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
            n_velocities=n_velocities,
        )

        misfits_list.append(misfit)

    # Convert list to a 3D NumPy array of shape (n_samples, n_velocities, len(freqs)).
    misfits = np.array(misfits_list)

    # Compute posterior statistics over the sample dimension (axis=0).
    mean_misfit = np.mean(misfits, axis=0)
    median_misfit = np.median(misfits, axis=0)
    lower_ci = np.percentile(misfits, 2.5, axis=0)
    upper_ci = np.percentile(misfits, 97.5, axis=0)

    return misfits, mean_misfit, median_misfit, lower_ci, upper_ci, velocity_axis


def pairs_to_xy(
    filtered_pairs: List[Tuple[np.ndarray, np.ndarray]], swap: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of pairs to x and y arrays. The function creates two arrays, one for the x values and one for the y
    values. The swap parameter determines whether the x and y values are swapped in the output arrays. If swap is False,
    the x values are taken from the first element of each pair and the y values are taken from the second element. If
    swap is True, the x values are taken from the second element of each pair and the y values are taken from the first
    element.

    Args:
        filtered_pairs: A list of (seg1, seg2) tuples representing the pairs of segments.
        swap: A boolean indicating whether to swap the x and y values in the output arrays.

    Returns:
        A tuple of two numpy arrays, one for the x values and one for the y values.
    """
    n_pairs = len(filtered_pairs)
    if n_pairs == 0:
        return np.array([]), np.array([])
    window_samples = len(filtered_pairs[0][0])
    dtype1 = filtered_pairs[0][0].dtype
    dtype2 = filtered_pairs[0][1].dtype
    x = np.zeros((n_pairs, window_samples), dtype=dtype1)
    y = np.zeros((n_pairs, window_samples), dtype=dtype2)
    for i, (seg1, seg2) in enumerate(filtered_pairs):
        if not swap:
            x[i] = seg1
            y[i] = seg2
        else:
            x[i] = seg2
            y[i] = seg1
    return x, y


def run_test(
    selected_pairs: List[Tuple[np.ndarray, np.ndarray]],
    epochs: int,
    kernel_size: int,
    batch_size_base: int,
    initial_learning_rate: float,
    target_learning_rate: float,
    alpha: Optional[float],
    one_bit_quantization: bool,
    spectral_whitening_mir: bool,
) -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tf_keras.Sequential,
    tf_keras.callbacks.History,
]:
    """
    Run a single test using the selected pairs. The function fits the Bayesian model, computes the cross-correlation,
    and returns the best training loss, the cross-correlation, and the mean impulse response.

    Args:
        selected_pairs: A list of (seg1, seg2) tuples representing the pairs of segments.
        epochs: The number of epochs for training.
        kernel_size: The kernel size for the LTIE model.
        batch_size_base: The base batch size for training.
        initial_learning_rate: The initial learning rate for training.
        target_learning_rate: The target learning rate for training.
        alpha: The alpha parameter for the LTIE model.
        one_bit_quantization: A boolean indicating whether to use one-bit quantization.
        spectral_whitening_mir: A boolean indicating whether to use spectral whitening for the impulse response.

    Returns:
        A tuple containing:
            - best_train_loss: The best training loss achieved during training.
            - cc_mean: The mean cross-correlation computed from the selected pairs.
            - cc_std: The standard deviation of the cross-correlation computed from the selected pairs.
            - ir_mean: The mean impulse response estimated by the LTIE model.
            - ir_std: The standard deviation of the impulse response estimated by the LTIE model.
            - model: The trained LTIE model.
            - fit_results: The results of the model fitting process.
    """
    # Compute the cross-correlation (CCF) from the selected pairs.
    cc_mean: np.ndarray
    cc_std: np.ndarray

    cc_mean, cc_std = compute_cross_correlation(
        pairs=selected_pairs, one_bit_quantization=one_bit_quantization
    )

    if spectral_whitening_mir:
        selected_pairs: List[Tuple[np.ndarray, np.ndarray]] = spectral_whiten_pairs(
            pairs=selected_pairs
        )

    # Convert pairs to x and y arrays (for both forward and swapped order)
    x1, y1 = pairs_to_xy(selected_pairs, swap=False)
    x2, y2 = pairs_to_xy(selected_pairs, swap=True)
    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    # Add a channel dimension (for TensorFlow) so that x and y are shape (n_examples, window_length, 1)
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    if one_bit_quantization:
        x = np.where(x >= 0, 1.0, -1.0)
        y = np.where(y >= 0, 1.0, -1.0)

    # Determine batch size and how many repeats of the data are needed
    num_examples = x.shape[0]

    batch_size = int(
        np.ceil(batch_size_base / num_examples) * num_examples
        if num_examples < batch_size_base
        else num_examples
    )

    num_repeats = batch_size // num_examples

    print(
        f"Training with num_examples {num_examples}, batch size {batch_size} and {num_repeats} repeats."
    )

    warmup_steps = int(epochs * 0.04)

    model = get_ltie_model(
        kernel_size=kernel_size,
        initial_learning_rate=initial_learning_rate,
        target_learning_rate=target_learning_rate,
        warmup_steps=warmup_steps,
        epochs=epochs,
        num_examples_per_epoch=batch_size,
        alpha=alpha,
    )

    if num_repeats > 0:
        # Prepare training data by repeating the examples to match the batch size.
        x_repeat = np.repeat(x, num_repeats, axis=0)
    else:
        x_repeat = x

    # Determine the output dimension of the model
    outdim = model(x_repeat).shape[1]
    y_start = kernel_size // 2
    y_end = y_start + outdim

    # In case y does not have enough points, adjust the slicing
    if y.shape[1] < y_end:
        y_slice = y[:, :outdim, :]
    else:
        y_slice = y[:, y_start:y_end, :]

    if num_repeats > 0:
        y_repeat = np.repeat(y_slice, num_repeats, axis=0)
    else:
        y_repeat = y_slice

    # Train the model (set verbose=0 for no output)
    print(f"Training model with x {x_repeat.shape} and y {y_repeat.shape}.")

    fit_results = model.fit(
        x_repeat,
        y_repeat,
        epochs=epochs,
        verbose=0,
        batch_size=batch_size,
    )

    best_train_loss = np.min(fit_results.history["loss"])

    ir_mean = np.flip(model.layers[0].kernel_posterior.mean().numpy().reshape(-1))

    ir_std: np.ndarray = np.flip(
        np.sqrt(model.layers[0].kernel_posterior.variance().numpy()).reshape(-1)
    )

    return best_train_loss, cc_mean, cc_std, ir_mean, ir_std, model, fit_results


def evaluate_test(
    cc: np.ndarray,
    mean_ir: np.ndarray,
    distance_rx: float,
    fs: float,
    velocity_true_or_func: float,
    best_train_loss: float,
    max_eval_velocity: float,
    max_freq: float,
    min_eval_velocity: float,
    min_freq: float,
    n_velocities: int,
    num: int,
    num_freq: int,
) -> dict:
    """
    Evaluate a test by computing the velocity fit statistics using the cross-correlation and the impulse response.

    Args:
        cc: The cross-correlation.
        mean_ir: The mean impulse response.
        distance_rx: The receiver distance.
        fs: The sampling frequency.
        velocity_true_or_func: The true velocity or a function returning velocity at frequency for the velocity fit
            statistics.
        best_train_loss: The best training loss achieved during training.
        max_eval_velocity: The maximum velocity for the velocity fit statistics.
        max_freq: The maximum frequency for the velocity fit statistics.
        min_eval_velocity: The minimum velocity for the velocity fit statistics.
        min_freq: The minimum frequency for the velocity fit statistics.
        n_velocities: The number of velocity candidates for the velocity fit statistics.
        num: The number of pairs used in the test.
        num_freq: The number of frequency bins to use for the evaluation.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    # Define frequency vector (using a fixed number of frequency bins)

    freqs = np.fft.rfftfreq(num_freq, 1 / fs)
    # For the CCF model, use the first num_freq samples of the cross-correlation.
    signal_window_cc = cc[:num_freq]

    misfit_cc = compute_velocity_misfit(
        signal_window=signal_window_cc,
        freqs=freqs,
        distance=distance_rx,
        min_velocity=min_eval_velocity,
        max_velocity=max_eval_velocity,
        n_velocities=n_velocities,
    )

    # For the IR model, take a segment from the estimated impulse response.
    center_idx_ir = (len(mean_ir) - 1) // 2
    signal_window_ir = mean_ir[center_idx_ir : center_idx_ir + num_freq]
    misfit_ir = compute_velocity_misfit(
        signal_window=signal_window_ir,
        freqs=freqs,
        distance=distance_rx,
        min_velocity=min_eval_velocity,
        max_velocity=max_eval_velocity,
        n_velocities=n_velocities,
    )
    # Compute velocity fit statistics over the full frequency range
    est_vel_ccf, mae_ccf, std_ccf = compute_velocity_fit_statistics(
        velocity_misfit=misfit_cc,
        freqs=freqs,
        min_velocity=min_eval_velocity,
        max_velocity=max_eval_velocity,
        n_velocities=n_velocities,
        velocity_true_or_func=velocity_true_or_func,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    est_vel_mir, mae_mir, std_mir = compute_velocity_fit_statistics(
        velocity_misfit=misfit_ir,
        freqs=freqs,
        min_velocity=min_eval_velocity,
        max_velocity=max_eval_velocity,
        n_velocities=n_velocities,
        velocity_true_or_func=velocity_true_or_func,
        min_freq=min_freq,
        max_freq=max_freq,
    )
    # Aggregate the error at the candidate velocity closest to the true velocity.
    mean_error_ccf, _ = aggregate_ground_truth_error(
        velocity_misfit=misfit_cc,
        freqs=freqs,
        min_velocity=min_eval_velocity,
        max_velocity=max_eval_velocity,
        n_velocities=n_velocities,
        velocity_true_or_func=velocity_true_or_func,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    mean_error_mir, _ = aggregate_ground_truth_error(
        velocity_misfit=misfit_ir,
        freqs=freqs,
        min_velocity=min_eval_velocity,
        max_velocity=max_eval_velocity,
        n_velocities=n_velocities,
        velocity_true_or_func=velocity_true_or_func,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    # Store the statistics in a dictionary.
    result = {
        "num_pairs": num,
        "ccf_mae": mae_ccf,
        "ccf_std": std_ccf,
        "ccf_target_error": mean_error_ccf,
        "mir_mae": mae_mir,
        "mir_std": std_mir,
        "mir_target_error": mean_error_mir,
        "best_train_loss": best_train_loss,
    }

    return result


def run_all_tests(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_counts: List[int],
    fs: float,
    distance_rx: float,
    velocity_true_or_func: float | Callable[[float], float],
    batch_size_base: int,
    epochs: int,
    initial_learning_rate: float,
    target_learning_rate: float,
    alpha: Optional[float],
    min_prop_speed: float,
    min_eval_velocity: float,
    max_eval_velocity: float,
    n_velocities: int,
    min_freq: float,
    max_freq: float,
    num_freq: int,
    one_bit_quantization: bool,
    spectral_whitening_mir: bool,
) -> pd.DataFrame:
    """
    Run a series of tests using different numbers of pairs and store the results in a DataFrame. The function
    fits the Bayesian model, computes the cross-correlation, evaluates the velocity misfit using two different
    methods (the full 2D fit and the target velocity error), and stores the results in a pandas DataFrame.

    Args:
        pairs: A list of (seg1, seg2) tuples representing the pairs of segments.
        test_counts: A list of integers representing the number of pairs to use in each test.
        fs: The sampling frequency.
        distance_rx: The receiver distance.
        velocity_true_or_func: The true velocity for the velocity fit statistics.
        batch_size_base: The base batch size for training.
        epochs: The number of epochs for training.
        initial_learning_rate: The initial learning rate for training.
        target_learning_rate: The target learning rate for training.
        alpha: The alpha parameter for the LTIE model.
        min_prop_speed: The minimum propagation speed for kernel size computation.
        min_eval_velocity: The minimum velocity for the velocity fit statistics.
        max_eval_velocity: The maximum velocity for the velocity fit statistics.
        n_velocities: The number of velocity candidates for the velocity fit statistics.
        min_freq: The minimum frequency for the velocity fit statistics.
        max_freq: The maximum frequency for the velocity fit statistics.
        num_freq: The number of frequency bins to use for the evaluation.
        one_bit_quantization: A boolean indicating whether to use one-bit quantization.
        spectral_whitening_mir: A boolean indicating whether to use spectral whitening for the impulse response.
    """

    # Compute kernel size based on the propagation time over the receiver distance.
    prop_time: float = np.ceil(distance_rx / min_prop_speed)
    kernel_size: int = int(np.ceil(fs * prop_time * 2))

    results = []
    for num in tqdm(test_counts):
        print(f"Running test with {num} pairs...")
        # Select the first "num" pair
        selected_pairs = pairs[:num]

        best_train_loss, cc_mean, _, ir_mean, _, model, _ = run_test(
            selected_pairs=selected_pairs,
            epochs=epochs,
            kernel_size=kernel_size,
            batch_size_base=batch_size_base,
            initial_learning_rate=initial_learning_rate,
            target_learning_rate=target_learning_rate,
            alpha=alpha,
            one_bit_quantization=one_bit_quantization,
            spectral_whitening_mir=spectral_whitening_mir,
        )

        result = evaluate_test(
            cc=cc_mean,
            mean_ir=ir_mean,
            distance_rx=distance_rx,
            fs=fs,
            velocity_true_or_func=velocity_true_or_func,
            best_train_loss=best_train_loss,
            max_eval_velocity=max_eval_velocity,
            max_freq=max_freq,
            min_eval_velocity=min_eval_velocity,
            min_freq=min_freq,
            n_velocities=n_velocities,
            num=num,
            num_freq=num_freq,
        )

        del model
        results.append(result)

    # Create and return a pandas DataFrame with all results.
    df_results = pd.DataFrame(results)
    return df_results

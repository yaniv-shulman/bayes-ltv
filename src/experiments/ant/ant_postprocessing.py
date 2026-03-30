"""Postprocess ANT posterior samples into phase and phase-velocity summaries."""

from typing import List, Tuple

import numpy as np
from scipy.signal import freqz
from tqdm import tqdm


def posterior_phase_responses(
    samples_impulse: np.ndarray, h_hat_mean: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute posterior phase-response summaries from impulse samples.

    Args:
        samples_impulse: Posterior impulse-response samples with shape `(n_samples, impulse_length)`.
        h_hat_mean: Complex mean frequency response of the fitted model.
        fs: Sampling frequency in Hz.

    Returns:
        Tuple containing:
            - `phase_samples`: Unwrapped phase responses for all samples with
              shape `(n_samples, n_freqs)`.
            - `phase_mean_posterior`: Mean posterior phase per frequency.
            - `phase_median_posterior`: Median posterior phase per frequency.
            - `phase_lower_ci`: 2.5th percentile posterior phase per frequency.
            - `phase_upper_ci`: 97.5th percentile posterior phase per frequency.
            - `model_phase_response_mean`: Unwrapped phase of `h_hat_mean`.

    Raises:
        ValueError: If `samples_impulse` has no samples.
    """
    phase_samples: List[np.ndarray] = []

    for i in tqdm(range(samples_impulse.shape[0]), desc="Calculating phase responses"):
        # Sample a kernel from the posterior
        sample_impulse = samples_impulse[i, :]

        # Compute frequency response
        _, h_hat_sample = freqz(sample_impulse, fs=fs)
        phase_sample = np.unwrap(np.angle(h_hat_sample))
        phase_samples.append(phase_sample)

    if len(phase_samples) == 0:
        raise ValueError("No samples in phase_samples.")

    # Convert to NumPy array for easier stats: shape (n_samples, n_freqs)
    phase_samples = np.array(phase_samples)
    # Compute posterior summary statistics across samples
    phase_mean_posterior = np.mean(phase_samples, axis=0)
    phase_median_posterior = np.median(phase_samples, axis=0)
    phase_lower_ci = np.percentile(phase_samples, 2.5, axis=0)
    phase_upper_ci = np.percentile(phase_samples, 97.5, axis=0)
    model_phase_response_mean = np.unwrap(np.angle(h_hat_mean))

    return (
        phase_samples,
        phase_mean_posterior,
        phase_median_posterior,
        phase_lower_ci,
        phase_upper_ci,
        model_phase_response_mean,
    )


def calculate_naive_phase_velocity(
    phase_sample: np.ndarray, distance: float, w_hat: np.ndarray, eps: float
) -> np.ndarray:
    """
    Estimate naive phase velocity for one phase-response sample.

    Args:
        phase_sample: Unwrapped phase-response sample across frequencies.
        distance: Receiver separation distance in meters.
        w_hat: Frequency axis in Hz.
        eps: Small positive threshold used to avoid near-zero division.

    Returns:
        Per-frequency naive phase-velocity estimate where invalid entries are
        encoded as `np.nan`.

    Raises:
        ValueError: If `eps` is not positive.
        ValueError: If `phase_sample` and `w_hat` shapes do not match.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    if phase_sample.shape != w_hat.shape:
        raise ValueError("phase_sample and w_hat must have matching shapes.")

    with np.errstate(divide="ignore", invalid="ignore"):
        phase_sample_safe = np.where(np.abs(phase_sample) < eps, np.nan, phase_sample)
        phase_velocity_sample = (2 * np.pi * w_hat * distance) / (-phase_sample_safe)

        phase_velocity_sample = np.where(
            phase_velocity_sample < 0, np.nan, phase_velocity_sample
        )

    return phase_velocity_sample


def posterior_phase_velocities(
    phase_samples: np.ndarray,
    distance: float,
    eps: float,
    min_frequency: float,
    w_hat: np.ndarray,
    signal_window_phase_response_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Summarize posterior phase velocities over all impulse samples.

    Args:
        phase_samples: Unwrapped phase samples with shape `(n_samples, n_freqs)`.
        distance: Receiver separation distance in meters.
        eps: Positive threshold used for numerical checks.
        min_frequency: Minimum frequency (Hz) used for strict sample validation.
        w_hat: Frequency axis in Hz with shape `(n_freqs,)`.
        signal_window_phase_response_mean: Mean phase response used for the deterministic reference velocity estimate.

    Returns:
        Tuple containing:
            - `phase_velocity_samples`: Sample-wise velocity estimates with
              shape `(n_samples, n_freqs)`.
            - `phase_velocity_mean_posterior`: Mean posterior velocity.
            - `phase_velocity_median_posterior`: Median posterior velocity.
            - `phase_velocity_lower_ci`: 2.5th percentile posterior velocity.
            - `phase_velocity_upper_ci`: 97.5th percentile posterior velocity.
            - `model_phase_velocity_mean`: Velocity estimate from
              `signal_window_phase_response_mean`.

    Raises:
        ValueError: If no frequencies exceed `min_frequency`.
        ValueError: If any sample has non-finite or sub-`eps` velocities above
            `min_frequency`.
        ValueError: If no phase-velocity samples are produced.
    """
    if phase_samples.ndim != 2:
        raise ValueError("phase_samples must be a 2D array.")

    if w_hat.ndim != 1:
        raise ValueError("w_hat must be a 1D array.")

    if phase_samples.shape[1] != w_hat.shape[0]:
        raise ValueError("phase_samples second dimension must match len(w_hat).")

    if signal_window_phase_response_mean.shape != w_hat.shape:
        raise ValueError("signal_window_phase_response_mean must match w_hat shape.")

    freq_mask: np.ndarray = w_hat > min_frequency

    if not np.any(freq_mask):
        raise ValueError("No frequencies exceed min_frequency.")

    phase_velocity_samples: List[np.ndarray] = []

    for i in tqdm(range(phase_samples.shape[0]), desc="Calculating phase velocities"):
        phase_sample = phase_samples[i, :]

        phase_velocity_sample = calculate_naive_phase_velocity(
            phase_sample=phase_sample,
            distance=distance,
            w_hat=w_hat,
            eps=eps,
        )

        selected_phase_velocities: np.ndarray = phase_velocity_sample[freq_mask]

        if not np.all(np.isfinite(selected_phase_velocities)):
            raise ValueError(
                f"Phase velocity sample {i} contains non-finite values above min_frequency."
            )

        if (selected_phase_velocities < eps).any():
            raise ValueError(
                f"Phase velocity sample {i} falls below eps above min_frequency."
            )

        phase_velocity_samples.append(phase_velocity_sample)

    if len(phase_velocity_samples) == 0:
        raise ValueError("No samples in phase_velocity_samples.")

    phase_velocity_samples = np.array(phase_velocity_samples)
    num_freq: int = phase_velocity_samples.shape[1]
    phase_velocity_mean_posterior = np.full(num_freq, np.nan)
    phase_velocity_median_posterior = np.full(num_freq, np.nan)
    phase_velocity_lower_ci = np.full(num_freq, np.nan)
    phase_velocity_upper_ci = np.full(num_freq, np.nan)

    for i in range(num_freq):
        finite_values: np.ndarray = phase_velocity_samples[
            np.isfinite(phase_velocity_samples[:, i]), i
        ]

        if finite_values.size == 0:
            continue

        phase_velocity_mean_posterior[i] = np.mean(finite_values)
        phase_velocity_median_posterior[i] = np.median(finite_values)
        phase_velocity_lower_ci[i] = np.percentile(finite_values, 2.5)
        phase_velocity_upper_ci[i] = np.percentile(finite_values, 97.5)

    model_phase_velocity_mean = calculate_naive_phase_velocity(
        phase_sample=signal_window_phase_response_mean,
        distance=distance,
        w_hat=w_hat,
        eps=eps,
    )

    return (
        phase_velocity_samples,
        phase_velocity_mean_posterior,
        phase_velocity_median_posterior,
        phase_velocity_lower_ci,
        phase_velocity_upper_ci,
        model_phase_velocity_mean,
    )

import numpy as np
from scipy.signal import freqz
from tqdm import tqdm


def posterior_phase_responses(samples_impulse, h_hat_mean, fs):
    phase_samples = []

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


def calculate_naive_phase_velocity(phase_sample, distance, w_hat, eps):
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_sample_safe = np.where(np.abs(phase_sample) < eps, np.nan, phase_sample)
        phase_velocity_sample = (2 * np.pi * w_hat * distance) / (-phase_sample_safe)

        phase_velocity_sample = np.where(
            phase_velocity_sample < 0, np.nan, phase_velocity_sample
        )
    return phase_velocity_sample


def posterior_phase_velocities(
    phase_samples,
    distance,
    eps,
    min_frequency,
    w_hat,
    signal_window_phase_response_mean,
):
    phase_velocity_samples = []

    for i in tqdm(range(phase_samples.shape[0]), desc="Calculating phase velocities"):
        phase_sample = phase_samples[i, :]

        phase_velocity_sample = calculate_naive_phase_velocity(
            phase_sample=phase_sample,
            distance=distance,
            w_hat=w_hat,
            eps=eps,
        )

        if (phase_velocity_sample[w_hat > min_frequency] < eps).any():
            print("Skipping sample", i)
            continue

        phase_velocity_samples.append(phase_velocity_sample)

    if len(phase_velocity_samples) == 0:
        raise ValueError("No samples in phase_velocity_samples.")

    phase_velocity_samples = np.array(phase_velocity_samples)
    phase_velocity_mean_posterior = np.nanmean(phase_velocity_samples, axis=0)
    phase_velocity_median_posterior = np.nanmedian(phase_velocity_samples, axis=0)
    phase_velocity_lower_ci = np.nanpercentile(phase_velocity_samples, 2.5, axis=0)
    phase_velocity_upper_ci = np.nanpercentile(phase_velocity_samples, 97.5, axis=0)

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

from concurrent.futures import ProcessPoolExecutor
import os
from itertools import repeat
from typing import Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def single_velocity(
    num_pairs: int,
    sequence_length: int,
    distance_rx: float,
    random_generator: np.random.Generator,
    num_sources: int = 10000,
    radius: float = 10000.0,
    velocity: float = 3000,
    sample_rate: float = 20.0,
    noise_std: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic pairs of signals simulating wave propagation from multiple sources. This function creates a
    specified number of signal pairs, each representing the received signal at two spatially separated receivers.
    Sources are initially positioned  uniformly on a circle with a given radius and then perturbed radially by random
    factors. The function computes the travel time from each source to both receivers based on the specified wave
    velocity, and then determines the sample delay between the receivers using the provided sample rate. Each source
    contributes a +/- unit impulse to the receiver signals. Optionally, Gaussian noise with a specified standard
    deviation can be added to the signals.

    Parameters:
        num_pairs (int): Number of signal pairs to generate.
        sequence_length (int): Length of the signal sequence for each receiver.
        distance_rx (float): Distance between the two receivers. The first receiver is at (0, 0)
                             and the second is at (distance_rx, 0).
        random_generator (np.random.Generator): Random number generator for reproducibility.
        num_sources (int, optional): Number of sources distributed on the circle. Default is 10000.
        radius (float, optional): Radius of the circle on which sources are initially placed.
                                  Default is 10000.0.
        velocity (float, optional): Propagation velocity of the wave in meters per second.
                                    Default is 3000.
        sample_rate (float, optional): Sampling rate in Hz used to discretize time delays.
                                       Default is 20.0.
        noise_std (float, optional): Standard deviation of additive Gaussian noise. If set to 0,
                                     no noise is added. Default is 0.0.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of tuples where each tuple contains two NumPy
        arrays corresponding to the simulated signals at receiver 1 and receiver 2, respectively.
    """
    rx1: np.ndarray = np.array([0, 0])  # Receiver at the origin
    rx2: np.ndarray = np.array([distance_rx, 0])  # Second receiver at [distance, 0]

    # Generate source positions in a circle
    angles: np.ndarray = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    i: int

    for i in range(num_pairs):
        pairs.append((np.zeros(sequence_length), np.zeros(sequence_length)))

    for i in tqdm(range(len(pairs))):
        # Compute source positions on a circle and add radial randomness
        source_positions: np.ndarray = np.array(
            [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        )

        source_positions = (
            source_positions.T * (1 + random_generator.random(num_sources) * 100)
        ).T

        # Calculate distances and travel times to the two receivers
        d1: np.ndarray = np.linalg.norm(rx1 - source_positions, axis=1)
        t1: np.ndarray = d1 / velocity
        d2: np.ndarray = np.linalg.norm(rx2 - source_positions, axis=1)
        t2: np.ndarray = d2 / velocity

        # Initialize raw signal arrays for both receivers
        x1: np.ndarray = np.zeros(sequence_length * 2)
        x2: np.ndarray = np.zeros(sequence_length * 2)

        # Compute the delay (in sample indices) between the two receivers
        dt: np.ndarray = np.round((t2 - t1) * sample_rate).astype(int)

        # Adjust the margin based on velocity; baseline of 100 when velocity == 3000
        margin: int = int(100 * (3000 / velocity))
        rt: np.ndarray = random_generator.integers(
            margin, sequence_length * 2 - margin, num_sources
        )

        # Add contributions from each source into the signal arrays
        j: int

        for j in range(num_sources):
            x1[rt[j]] += 1
            x1[rt[j] + 1] -= 1
            x2[rt[j] + dt[j]] += 1
            x2[rt[j] + 1 + dt[j]] -= 1

        # Retain the raw observed signals by extracting a segment from each
        start: int = len(x1) // 2 - sequence_length // 2
        end: int = start + sequence_length
        pairs[i][0][:] = x1[start:end]
        pairs[i][1][:] = x2[start:end]

        # Add noise to the signals
        if noise_std > 0:
            pairs[i] = (
                pairs[i][0] + random_generator.normal(0, noise_std, sequence_length),
                pairs[i][1] + random_generator.normal(0, noise_std, sequence_length),
            )

    return pairs


def single_velocity_sinus_decaying(
    num_pairs: int,
    sequence_length: int,
    distance_rx: float,
    random_generator: np.random.Generator,
    num_sources: int = 10000,
    radius: float = 10000.0,
    velocity: float = 3000,
    sample_rate: float = 20.0,
    noise_std: float = 0.0,
    pulse_length: int = 50,
    frequency: float = 5.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simulates signals at two receivers. Each source emits a decaying sinusoidal pulse
    with random amplitude and phase. The decay is implemented via an exponential envelope.

    Args:
        num_pairs: Number of signal pairs to generate.
        sequence_length: Length of the final signal segment (samples).
        distance_rx: Distance between the two receivers.
        random_generator: Random number generator.
        num_sources: Number of sources.
        radius: Radius of the circle on which sources are initially placed.
        velocity: Wave velocity in m/s.
        sample_rate: Sampling rate in Hz.
        noise_std: Standard deviation of the additive noise. If 0, no noise is added.
        pulse_length: Number of samples over which each sinusoidal pulse is applied.
        frequency: Frequency of the sinusoidal pulse in Hz.

    Returns:
        List of tuples, each containing the pair of signals (as numpy arrays).
    """
    rx1: np.ndarray = np.array([0, 0])  # Receiver at the origin
    rx2: np.ndarray = np.array([distance_rx, 0])  # Second receiver at [distance_rx, 0]

    # Generate source positions on a circle (by angles)
    angles: np.ndarray = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)

    # Prepare list to store pairs of signals
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    i: int

    for i in range(num_pairs):
        pairs.append((np.zeros(sequence_length), np.zeros(sequence_length)))

    # Precompute the time vector for the sinusoidal pulse (in seconds)
    t_pulse: np.ndarray = np.arange(pulse_length) / sample_rate

    for i in tqdm(range(len(pairs))):
        # Compute source positions on a circle and add radial randomness
        source_positions: np.ndarray = np.array(
            [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
        )

        source_positions = (
            source_positions.T * (1 + random_generator.random(num_sources) * 100)
        ).T

        # Calculate distances and travel times to the two receivers
        d1: np.ndarray = np.linalg.norm(rx1 - source_positions, axis=1)
        t1: np.ndarray = d1 / velocity
        d2: np.ndarray = np.linalg.norm(rx2 - source_positions, axis=1)
        t2: np.ndarray = d2 / velocity

        # Initialize raw signal arrays for both receivers (longer to allow for margin)
        x1: np.ndarray = np.zeros(sequence_length * 2)
        x2: np.ndarray = np.zeros(sequence_length * 2)

        # Compute the delay (in sample indices) between the two receivers
        dt: np.ndarray = np.round((t2 - t1) * sample_rate).astype(int)

        # Adjust the margin based on velocity and pulse_length to avoid edge effects.
        margin: int = pulse_length + int(100 * (3000 / velocity))

        rt: np.ndarray = random_generator.integers(
            margin, sequence_length * 2 - margin, num_sources
        )

        # For each source, add a decaying sinusoidal pulse with random amplitude and phase.
        for j in range(num_sources):
            amplitude: float = random_generator.uniform(-1.5, 1.5)
            frequency: float = random_generator.uniform(0.1, 10.0)
            # phase: float = random_generator.uniform(0, 2 * np.pi)
            phase: float = 0.0

            # Set time constant tau such that the envelope decays to ~exp(-2) at the end of the pulse.
            tau: float = random_generator.uniform(0.5, 2.5)
            # Exponential decay envelope
            envelope: np.ndarray = np.exp(-t_pulse / tau)

            # Generate the sinusoid and apply the exponential decay envelope.
            sinus: np.ndarray = (
                amplitude * np.sin(2 * np.pi * frequency * t_pulse + phase) * envelope
            )

            # Add the sinusoidal pulse to receiver 1 at time index rt[j]
            x1[rt[j] : rt[j] + pulse_length] += sinus
            # Add the same sinusoidal pulse to receiver 2, shifted by the delay dt[j]
            x2[rt[j] + dt[j] : rt[j] + dt[j] + pulse_length] += sinus

        # Extract the central segment of the signal as the final observed signal.
        start: int = len(x1) // 2 - sequence_length // 2
        end: int = start + sequence_length
        sig1: np.ndarray = x1[start:end]
        sig2: np.ndarray = x2[start:end]

        # Optionally add noise to the signals
        if noise_std > 0:
            sig1 = sig1 + random_generator.normal(0, noise_std, sequence_length)
            sig2 = sig2 + random_generator.normal(0, noise_std, sequence_length)

        pairs[i] = (sig1, sig2)

    return pairs


def velocity_curve_sinus_decaying(
    num_pairs: int,
    sequence_length: int,
    distance_rx: float,
    freq_velocity_pairs,  # can be a float or a list of (frequency, velocity) tuples
    random_generator: np.random.Generator,
    num_sources: int = 10000,
    radius: float = 10000.0,
    sample_rate: float = 20.0,
    noise_std: float = 0.0,
    pulse_length: int = 50,
    num_workers: Optional[int] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Callable[[float], float]]:
    """
    Simulates signals at two receivers. Each source emits a decaying sinusoidal pulse
    with random amplitude, random frequency (if variable velocity is used, this frequency
    determines the travel velocity via a fitted curve), and an exponential decay envelope.

    Args:
        num_pairs: Number of signal pairs to generate.
        sequence_length: Length of the final signal segment (samples).
        distance_rx: Distance between the two receivers.
        random_generator: Random number generator.
        num_sources: Number of sources.
        radius: Radius of the circle on which sources are initially placed.
        freq_velocity_pairs: Either a constant velocity (float/int) or a list of (frequency, velocity) tuples.
        sample_rate: Sampling rate in Hz.
        noise_std: Standard deviation of the additive noise. If 0, no noise is added.
        pulse_length: Number of samples over which each sinusoidal pulse is applied.
        num_workers: Number of worker processes used to generate pairs. If
            `None`, the function uses up to `os.cpu_count()` workers.

    Returns:
        List of tuples, each containing the pair of signals (as numpy arrays).
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")

    rx1: np.ndarray = np.array([0, 0])
    rx2: np.ndarray = np.array([distance_rx, 0])

    # Determine velocity handling.
    if isinstance(freq_velocity_pairs, (list, tuple)):
        freqs, vels = zip(*freq_velocity_pairs)
        poly_coeffs = np.polyfit(freqs, vels, deg=2)
        velocity_func = lambda f: np.polyval(poly_coeffs, f)
        min_velocity = min(vels)
        variable_velocity = True
    else:
        velocity_func = lambda f: freq_velocity_pairs
        variable_velocity = False

    # Use a margin that is conservative (based on the slowest velocity if variable)
    ref_velocity = min_velocity if variable_velocity else freq_velocity_pairs
    margin: int = pulse_length + int(100 * (3000 / ref_velocity))

    angles: np.ndarray = np.linspace(0, 2 * np.pi, num_sources, endpoint=False)
    base_source_positions: np.ndarray = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles))
    )
    t_pulse: np.ndarray = np.arange(pulse_length) / sample_rate

    pair_seeds: np.ndarray = random_generator.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=num_pairs,
        dtype=np.uint32,
    )

    resolved_num_workers: int = (
        min(os.cpu_count() or 1, num_pairs) if num_workers is None else num_workers
    )

    if resolved_num_workers <= 0:
        raise ValueError("num_workers must be positive when provided.")

    if resolved_num_workers == 1:
        pairs = [
            _generate_velocity_curve_sinus_decaying_pair(
                seed=int(pair_seed),
                sequence_length=sequence_length,
                base_source_positions=base_source_positions,
                rx1=rx1,
                rx2=rx2,
                variable_velocity=variable_velocity,
                poly_coeffs=poly_coeffs if variable_velocity else None,
                constant_velocity=(
                    None if variable_velocity else float(freq_velocity_pairs)
                ),
                sample_rate=sample_rate,
                noise_std=noise_std,
                pulse_length=pulse_length,
                t_pulse=t_pulse,
                margin=margin,
            )
            for pair_seed in tqdm(pair_seeds, total=num_pairs)
        ]
    else:
        max_workers: int = min(resolved_num_workers, num_pairs)
        chunksize: int = max(1, num_pairs // (max_workers * 4))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pairs = list(
                tqdm(
                    executor.map(
                        _generate_velocity_curve_sinus_decaying_pair,
                        map(int, pair_seeds),
                        repeat(sequence_length),
                        repeat(base_source_positions),
                        repeat(rx1),
                        repeat(rx2),
                        repeat(variable_velocity),
                        repeat(poly_coeffs if variable_velocity else None),
                        repeat(
                            None if variable_velocity else float(freq_velocity_pairs)
                        ),
                        repeat(sample_rate),
                        repeat(noise_std),
                        repeat(pulse_length),
                        repeat(t_pulse),
                        repeat(margin),
                        chunksize=chunksize,
                    ),
                    total=num_pairs,
                )
            )

    return pairs, velocity_func


def _generate_velocity_curve_sinus_decaying_pair(
    seed: int,
    sequence_length: int,
    base_source_positions: np.ndarray,
    rx1: np.ndarray,
    rx2: np.ndarray,
    variable_velocity: bool,
    poly_coeffs: Optional[np.ndarray],
    constant_velocity: Optional[float],
    sample_rate: float,
    noise_std: float,
    pulse_length: int,
    t_pulse: np.ndarray,
    margin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic receiver pair for curved-velocity ANT data."""
    random_generator: np.random.Generator = np.random.default_rng(seed)
    num_sources: int = base_source_positions.shape[0]
    radial_scale: np.ndarray = 1.0 + random_generator.random(num_sources) * 100.0
    source_positions: np.ndarray = base_source_positions * radial_scale[:, np.newaxis]

    d1: np.ndarray = np.linalg.norm(rx1 - source_positions, axis=1)
    d2: np.ndarray = np.linalg.norm(rx2 - source_positions, axis=1)

    if not variable_velocity:
        if constant_velocity is None:
            raise ValueError(
                "constant_velocity must be provided for fixed-velocity generation."
            )
        t1: np.ndarray = d1 / constant_velocity
        t2: np.ndarray = d2 / constant_velocity
        dt: np.ndarray = np.round((t2 - t1) * sample_rate).astype(int)
    else:
        if poly_coeffs is None:
            raise ValueError(
                "poly_coeffs must be provided for variable-velocity generation."
            )

    x1: np.ndarray = np.zeros(sequence_length * 2)
    x2: np.ndarray = np.zeros(sequence_length * 2)

    amplitudes: np.ndarray = random_generator.uniform(-1.5, 1.5, num_sources)
    freqs: np.ndarray = random_generator.uniform(0.1, 10.0, num_sources)
    taus: np.ndarray = random_generator.uniform(0.5, 2.5, num_sources)

    for j in range(num_sources):
        envelope: np.ndarray = np.exp(-t_pulse / taus[j])
        sinus: np.ndarray = (
            amplitudes[j] * np.sin(2 * np.pi * freqs[j] * t_pulse) * envelope
        )

        if variable_velocity:
            current_velocity: float = float(np.polyval(poly_coeffs, freqs[j]))
            t1_current: float = d1[j] / current_velocity
            t2_current: float = d2[j] / current_velocity
            dt_j: int = round((t2_current - t1_current) * sample_rate)
        else:
            dt_j = int(dt[j])

        valid_rt: Optional[int] = None
        for _ in range(10):
            candidate: int = int(
                random_generator.integers(
                    margin,
                    sequence_length * 2 - margin - pulse_length,
                )
            )
            if (
                candidate + dt_j >= 0
                and candidate + dt_j + pulse_length <= sequence_length * 2
            ):
                valid_rt = candidate
                break
        if valid_rt is None:
            continue

        x1[valid_rt : valid_rt + pulse_length] += sinus
        x2[valid_rt + dt_j : valid_rt + dt_j + pulse_length] += sinus

    start: int = len(x1) // 2 - sequence_length // 2
    end: int = start + sequence_length
    sig1: np.ndarray = x1[start:end]
    sig2: np.ndarray = x2[start:end]

    if noise_std > 0:
        sig1 = sig1 + random_generator.normal(0, noise_std, sequence_length)
        sig2 = sig2 + random_generator.normal(0, noise_std, sequence_length)

    return sig1, sig2


if __name__ == "__main__":
    single_velocity(num_pairs=1000, sequence_length=1200, distance_rx=9000)

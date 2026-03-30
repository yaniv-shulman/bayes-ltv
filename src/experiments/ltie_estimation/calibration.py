import argparse
import logging
from pathlib import Path
from statistics import NormalDist
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
from scipy.signal import firwin

from experiments.context import data_dir, out_dir
from models.ltie import get_ltie_model

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERVAL_LEVELS: Tuple[float, ...] = (0.90, 0.95)


def compute_uniform_batch_repetitions(
    num_independent_examples: int,
    batch_size_base: int,
) -> Tuple[int, int]:
    """Compute a uniform repetition plan for a repeated optimization batch.

    Args:
        num_independent_examples: Number of statistically independent examples.
        batch_size_base: Requested minimum batch size for optimization.

    Returns:
        The concrete batch size and the uniform repetition count per example.

    Raises:
        ValueError: If the example count or batch-size target is not positive.
    """
    if num_independent_examples <= 0:
        raise ValueError("num_independent_examples must be positive.")

    if batch_size_base <= 0:
        raise ValueError("batch_size_base must be positive.")

    if num_independent_examples < batch_size_base:
        batch_size: int = int(
            np.ceil(batch_size_base / num_independent_examples)
            * num_independent_examples
        )
    else:
        batch_size = num_independent_examples

    return batch_size, batch_size // num_independent_examples


def load_default_source_signal(source_column: int = 1) -> np.ndarray:
    """Load the default LTIE source signal from the Earthquakes dataset.

    Args:
        source_column: Column index of the source trace to load.

    Returns:
        The selected source signal.
    """

    data_file: Path = data_dir.joinpath("Earthquakes_TRAIN.txt")

    data_frame: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=data_file,
        header=None,
        sep="  ",
        engine="python",
    )

    return data_frame[source_column].values


def build_default_linear_phase_fir(numtaps: int = 16) -> np.ndarray:
    """Build the default linear-phase FIR used in the LTIE notebook.

    Args:
        numtaps: Number of FIR taps.

    Returns:
        The synthetic ground-truth FIR coefficients.
    """
    return firwin(
        numtaps=numtaps,
        cutoff=[0.25, 0.40, 0.50, 0.95],
        width=0.05,
        pass_zero=False,
    )


def fit_ltie_posterior_statistics(
    sources: np.ndarray,
    received_noisy: np.ndarray,
    batch_size_base: int,
    epochs: int,
    initial_learning_rate: float,
    target_learning_rate: float,
    alpha: float,
    observation_noise_std: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Fit the LTIE model and return posterior tap moments.

    Args:
        sources: Known source signals with shape `(num_examples, signal_length)`.
        received_noisy: Noisy received signals with shape
            `(num_examples, received_length)`.
        batch_size_base: Requested minimum optimization batch size.
        epochs: Number of optimization epochs.
        initial_learning_rate: Initial optimizer learning rate.
        target_learning_rate: Peak optimizer learning rate.
        alpha: Prior scaling parameter.
        observation_noise_std: Observation noise standard deviation used to
            scale the Gaussian data-fit term.

    Returns:
        The posterior mean and posterior standard deviation for each FIR tap,
        the concrete batch size, and the uniform repetition count per example.
    """
    num_independent_examples: int = sources.shape[0]
    kernel_size: int = sources.shape[1] - received_noisy.shape[1] + 1
    warmup_steps: int = int(epochs * 0.04)
    batch_size: int
    num_repetitions: int
    batch_size, num_repetitions = compute_uniform_batch_repetitions(
        num_independent_examples=num_independent_examples,
        batch_size_base=batch_size_base,
    )

    model: tf_keras.Sequential = get_ltie_model(
        kernel_size=kernel_size,
        initial_learning_rate=initial_learning_rate,
        target_learning_rate=target_learning_rate,
        warmup_steps=warmup_steps,
        epochs=epochs,
        num_independent_examples_per_epoch=num_independent_examples,
        alpha=alpha,
        observation_noise_std=observation_noise_std,
    )

    x_base: np.ndarray = sources[..., np.newaxis]
    x: np.ndarray = np.repeat(x_base, num_repetitions, axis=0)
    outdim: int = received_noisy.shape[1]

    y_base: np.ndarray = received_noisy[..., np.newaxis][:, :outdim, :]
    y: np.ndarray = np.repeat(
        y_base,
        num_repetitions,
        axis=0,
    )

    model.fit(
        x,
        y,
        epochs=epochs,
        verbose=False,
        batch_size=batch_size,
        shuffle=False,
    )

    posterior_mean: np.ndarray = np.flip(
        model.layers[0].kernel_posterior.mean().numpy().reshape(-1)
    )

    posterior_std: np.ndarray = np.flip(
        np.sqrt(model.layers[0].kernel_posterior.variance().numpy()).reshape(-1)
    )

    return posterior_mean, posterior_std, batch_size, num_repetitions


def summarize_pointwise_interval_coverage(
    fir_ground_truth: np.ndarray,
    posterior_means: np.ndarray,
    posterior_stds: np.ndarray,
    interval_levels: Tuple[float, ...],
) -> pd.DataFrame:
    """Summarize pointwise coverage of nominal posterior intervals.

    Args:
        fir_ground_truth: Ground-truth FIR coefficients with shape `(num_taps,)`.
        posterior_means: Posterior means with shape `(num_replications, num_taps)`.
        posterior_stds: Posterior standard deviations with shape
            `(num_replications, num_taps)`.
        interval_levels: Nominal interval levels in `(0, 1)`.

    Returns:
        A summary table containing nominal coverage, empirical coverage,
        and average interval width for each interval level.

    Raises:
        ValueError: If the array shapes are inconsistent or an interval level
            lies outside `(0, 1)`.
    """
    if posterior_means.shape != posterior_stds.shape:
        raise ValueError("posterior_means and posterior_stds must have the same shape.")

    if posterior_means.ndim != 2:
        raise ValueError("posterior_means and posterior_stds must be 2D arrays.")

    if posterior_means.shape[1] != fir_ground_truth.shape[0]:
        raise ValueError(
            "The number of FIR taps must match the second dimension of the posterior arrays."
        )

    level: float
    for level in interval_levels:
        if not 0.0 < level < 1.0:
            raise ValueError("Interval levels must lie strictly between 0 and 1.")

    total_taps: int = posterior_means.shape[0] * posterior_means.shape[1]
    rows: List[dict] = []

    for level in interval_levels:
        z_score: float = NormalDist().inv_cdf(0.5 + 0.5 * level)
        half_width: np.ndarray = z_score * posterior_stds
        lower: np.ndarray = posterior_means - half_width
        upper: np.ndarray = posterior_means + half_width
        covered: np.ndarray = (fir_ground_truth.reshape(1, -1) >= lower) & (
            fir_ground_truth.reshape(1, -1) <= upper
        )

        rows.append(
            {
                "interval_level": level,
                "nominal_coverage": level,
                "empirical_coverage": float(np.sum(covered) / total_taps),
                "mean_interval_width": float(np.mean(2.0 * half_width)),
                "num_replications": posterior_means.shape[0],
                "num_taps": posterior_means.shape[1],
            }
        )

    return pd.DataFrame(rows).sort_values("interval_level").reset_index(drop=True)


def run_pointwise_ltie_calibration(
    num_replications: int,
    noise_std: float,
    batch_size: int,
    epochs: int,
    seed: int,
    interval_levels: Tuple[float, ...],
    num_independent_examples_per_epoch: int = 1,
    source_column: int = 1,
    initial_learning_rate: float = 0.0,
    target_learning_rate: float = 0.5,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Run a repeated-noise LTIE pointwise calibration check.

    Args:
        num_replications: Number of repeated noisy observations to fit.
        noise_std: Standard deviation of the additive Gaussian noise.
        batch_size: Requested minimum optimization batch size.
        epochs: Number of optimization epochs per replication.
        seed: Base random seed for noise draws and TensorFlow initialization.
        interval_levels: Nominal interval levels to evaluate.
        num_independent_examples_per_epoch: Number of statistically
            independent noisy observation pairs used in each fitted model.
        source_column: Column index of the Earthquakes trace to use.
        initial_learning_rate: Initial optimizer learning rate.
        target_learning_rate: Peak optimizer learning rate.
        alpha: Prior scaling parameter.

    Returns:
        A summary table with pointwise interval coverage statistics.
    """
    source: np.ndarray = load_default_source_signal(source_column=source_column)
    fir_ground_truth: np.ndarray = build_default_linear_phase_fir()
    sources: np.ndarray = np.repeat(
        source.reshape(1, -1),
        num_independent_examples_per_epoch,
        axis=0,
    )

    received_clean: np.ndarray = np.convolve(
        source,
        fir_ground_truth,
        mode="valid",
    ).squeeze()

    posterior_means: np.ndarray = np.empty(
        (num_replications, fir_ground_truth.shape[0])
    )

    posterior_stds: np.ndarray = np.empty((num_replications, fir_ground_truth.shape[0]))
    actual_batch_sizes: np.ndarray = np.empty(num_replications, dtype=int)
    num_repetitions_arr: np.ndarray = np.empty(num_replications, dtype=int)

    rng: np.random.Generator = np.random.default_rng(seed)
    replication: int

    for replication in range(num_replications):
        tf_keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + replication)

        received_noisy: np.ndarray = received_clean.reshape(1, -1) + rng.normal(
            0.0,
            noise_std,
            size=(num_independent_examples_per_epoch, received_clean.shape[0]),
        )

        posterior_mean: np.ndarray
        posterior_std: np.ndarray
        actual_batch_size: int
        num_repetitions: int

        (
            posterior_mean,
            posterior_std,
            actual_batch_size,
            num_repetitions,
        ) = fit_ltie_posterior_statistics(
            sources=sources,
            received_noisy=received_noisy,
            batch_size_base=batch_size,
            epochs=epochs,
            initial_learning_rate=initial_learning_rate,
            target_learning_rate=target_learning_rate,
            alpha=alpha,
            observation_noise_std=noise_std,
        )

        posterior_means[replication, :] = posterior_mean
        posterior_stds[replication, :] = posterior_std
        actual_batch_sizes[replication] = actual_batch_size
        num_repetitions_arr[replication] = num_repetitions

    summary: pd.DataFrame = summarize_pointwise_interval_coverage(
        fir_ground_truth=fir_ground_truth,
        posterior_means=posterior_means,
        posterior_stds=posterior_stds,
        interval_levels=interval_levels,
    )
    summary["noise_std"] = noise_std
    summary["epochs"] = epochs
    summary["batch_size_base"] = batch_size
    summary["actual_batch_size"] = int(actual_batch_sizes[0])
    summary["num_repetitions_per_example"] = int(num_repetitions_arr[0])
    summary["num_independent_examples_per_epoch"] = num_independent_examples_per_epoch
    summary["source_column"] = source_column

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the calibration script.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run a repeated-noise LTIE pointwise calibration check."
    )
    parser.add_argument("--num-replications", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--num-independent-examples-per-epoch",
        type=int,
        default=1,
    )
    parser.add_argument("--source-column", type=int, default=1)
    parser.add_argument(
        "--interval-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_INTERVAL_LEVELS),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=out_dir.joinpath(
            "ltie_estimation",
            "ltie_pointwise_calibration_summary.csv",
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the LTIE calibration check and write the summary to disk."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args: argparse.Namespace = parse_args()
    output_file: Path = args.output_file
    output_dir: Path = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: pd.DataFrame = run_pointwise_ltie_calibration(
        num_replications=args.num_replications,
        noise_std=args.noise_std,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        interval_levels=tuple(args.interval_levels),
        num_independent_examples_per_epoch=args.num_independent_examples_per_epoch,
        source_column=args.source_column,
    )
    summary.to_csv(output_file, index=False)

    LOGGER.info("Wrote pointwise calibration summary to %s", output_file)
    LOGGER.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from plotly import graph_objects as go
from scipy.signal import freqz
from scipy.signal import group_delay

from experiments.plots_utils_common import (
    received_color_plotly,
    received_noisy_color_plotly,
    fir_ground_truth_color_plotly,
    fir_mean_est_color_plotly,
    gt_color_matplotlib,
    received_noisy_color_matplotlib,
    est_color_matplotlib,
)


def plot_fir_ground_truth_plotly(
    fir_ground_truth: np.ndarray, plots_file: Path, include_title_in_plots: bool
) -> None:

    fig: go.Figure = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(fir_ground_truth)),
            y=fir_ground_truth,
            line=dict(color=fir_ground_truth_color_plotly),
            name="Ground Truth FIR",
        )
    )

    fig.update_layout(
        go.Layout(
            title="Ground Truth FIR" if include_title_in_plots else None,
            height=600,
            width=800,
        )
    )

    fig.write_image(plots_file)
    fig.show()


def plot_fir_fit_and_ground_truth_plotly(
    fir_mean_est: np.ndarray,
    fir_std_est: np.ndarray,
    fir_ground_truth: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:

    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.arange(fir_ground_truth.shape[0]),
                y=fir_mean_est + 2 * fir_std_est,
                mode="lines",
                line=dict(color=f"light{fir_mean_est_color_plotly}"),
                name="mean + 2 std",
            ),
            go.Scatter(
                x=np.arange(fir_ground_truth.shape[0]),
                y=fir_mean_est - 2 * fir_std_est,
                mode="lines",
                line=dict(color=f"light{fir_mean_est_color_plotly}"),
                name="mean - 2 std",
                fill="tonexty",
            ),
            go.Scatter(
                x=np.arange(fir_ground_truth.shape[0]),
                y=fir_ground_truth,
                line=dict(color=fir_ground_truth_color_plotly),
                name="ground truth",
            ),
            go.Scatter(
                x=np.arange(fir_ground_truth.shape[0]),
                y=fir_mean_est,
                line=dict(color=fir_mean_est_color_plotly),
                name="estimated mean",
            ),
        ]
    )

    fig.update_layout(
        go.Layout(
            title="Estimated and Ground Truth FIR" if include_title_in_plots else None,
            height=600,
            width=800,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79),
        )
    )

    fig.write_image(plots_file)
    fig.show()


def plot_fir_fit_and_ground_truth_matplotlib(
    fir_mean_est: np.ndarray,
    fir_std_est: np.ndarray,
    fir_ground_truth: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    band_alpha: float = 0.2,
    width: float = 12.0,
    height: float = 8.0,
    dpi: float = 300.0,
) -> None:
    """
    Plot estimated FIR mean ±2 std (shaded), the mean line, and ground truth
    """
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(width * cm, height * cm),
        dpi=dpi,
    )

    x = np.arange(fir_ground_truth.shape[0])
    upper = fir_mean_est + 2 * fir_std_est
    lower = fir_mean_est - 2 * fir_std_est

    # Shaded ±2σ band
    ax.fill_between(
        x,
        lower,
        upper,
        color=est_color_matplotlib,
        alpha=band_alpha,
        label="Mean ± 2 σ",
    )

    # Ground truth
    ax.plot(
        x,
        fir_ground_truth,
        color=gt_color_matplotlib,
        linewidth=1.0,
        label="Ground truth",
    )

    # Estimated mean
    ax.plot(
        x,
        fir_mean_est,
        color=est_color_matplotlib,
        linewidth=1.0,
        linestyle="--",
        label="Estimated mean",
    )

    ax.set_xlabel("Tap index")
    ax.set_ylabel("Coefficient value")

    if include_title_in_plots:
        ax.set_title("Estimated FIR ±2 STD vs Ground Truth")

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plots_file, dpi=300)
    plt.show()


def posterior_samples_fir_plotly(
    samples: np.ndarray,
    fir_ground_truth: np.ndarray,
    fir_mean_est: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    fig = go.Figure()
    i: int

    for i in range(samples.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=np.arange(fir_ground_truth.shape[0]),
                y=samples[i],
                line=dict(color=fir_mean_est_color_plotly),
                name=None if i > 0 else "Samples from Posterior",
                showlegend=False if i > 0 else True,
                opacity=0.01,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(fir_ground_truth.shape[0]),
            y=fir_ground_truth,
            line=dict(color=fir_ground_truth_color_plotly),
            name="ground truth",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(fir_ground_truth.shape[0]),
            y=fir_mean_est,
            line=dict(color=fir_mean_est_color_plotly),
            name="estimated mean",
        )
    )

    fig.update_layout(
        go.Layout(
            title=(
                "Samples from the Posterior of Impulse Response"
                if include_title_in_plots
                else None
            ),
            height=600,
            width=800,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.73),
        )
    )

    fig.write_image(plots_file)
    fig.show()


def posterior_samples_fir_matplotlib(
    samples: np.ndarray,
    fir_ground_truth: np.ndarray,
    fir_mean_est: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    sample_alpha: float = 0.01,
    width: float = 12.0,
    height: float = 8.0,
    dpi: float = 300.0,
) -> None:
    """
    Plot samples from the posterior of the FIR kernel (faint lines), plus the estimated mean (dashed) and ground truth
    (solid).
    """
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(width * cm, height * cm), dpi=dpi)

    x: np.ndarray = np.arange(fir_ground_truth.shape[0])

    # Draw posterior samples
    for i in range(samples.shape[0]):
        # sample kernel, reshape to 1D
        w: np.ndarray = samples[i]

        ax.plot(
            x,
            w,
            color=est_color_matplotlib,
            alpha=sample_alpha,
            linewidth=0.5,
            label="Posterior samples" if i == 0 else None,
        )

    # Ground truth
    ax.plot(
        x,
        fir_ground_truth,
        color=gt_color_matplotlib,
        linewidth=1.0,
        label="Ground truth",
    )

    # Estimated mean
    ax.plot(
        x,
        fir_mean_est,
        color=est_color_matplotlib,
        linewidth=1.0,
        linestyle="--",
        label="Estimated mean",
    )

    ax.set_xlabel("Tap index")
    ax.set_ylabel("Coefficient value")

    if include_title_in_plots:
        ax.set_title("Samples from the Posterior of Impulse Response")

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plots_file, dpi=300)
    plt.show()


def plot_frequency_response_plotly(
    samples: np.ndarray,
    w_ground_truth: np.ndarray,
    h_ground_truth: np.ndarray,
    w_mean_est: np.ndarray,
    h_mean_est: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    fig: go.Figure = go.Figure()
    i: int

    for i in range(samples.shape[0]):
        w_sample: np.ndarray
        h_sample: np.ndarray
        w_sample, h_sample = freqz(samples[i])

        fig.add_trace(
            go.Scatter(
                x=w_sample,
                y=20 * np.log10(abs(h_sample)),
                line=dict(color=fir_mean_est_color_plotly),
                name=None if i > 0 else "Samples from Posterior",
                showlegend=False if i > 0 else True,
                opacity=0.01,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=w_ground_truth,
            y=20 * np.log10(abs(h_ground_truth)),
            line=dict(color=fir_ground_truth_color_plotly),
            name="Ground Truth",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=w_mean_est,
            y=20 * np.log10(abs(h_mean_est)),
            line=dict(color=fir_mean_est_color_plotly),
            name="Estimated Mean",
        )
    )

    fig.update_layout(
        go.Layout(
            title=(
                "Frequency Response of Mean Estimated IR and Samples from the Posterior"
                if include_title_in_plots
                else None
            ),
            height=600,
            width=800,
            legend=dict(yanchor="top", y=0.17, xanchor="left", x=0.73),
        )
    )

    fig.write_image(plots_file)
    fig.show()


def plot_frequency_response_matplotlib(
    samples: np.ndarray,
    w_ground_truth: np.ndarray,
    h_ground_truth: np.ndarray,
    w_mean_est: np.ndarray,
    h_mean_est: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    sample_alpha: float = 0.01,
    width: float = 12.0,
    height: float = 8.0,
    dpi: float = 300.0,
) -> None:
    """
    Plot frequency responses of posterior samples (faint lines), plus ground truth and mean estimate.
    """
    cm: float = 1 / 2.54
    fig, ax = plt.subplots(figsize=(width * cm, height * cm), dpi=dpi)
    i: int

    for i in range(samples.shape[0]):
        w_sample: np.ndarray
        h_sample: np.ndarray
        w_sample, h_sample = freqz(samples[i])
        mag_db: np.ndarray = 20 * np.log10(np.abs(h_sample))

        ax.plot(
            w_sample,
            mag_db,
            color=est_color_matplotlib,
            alpha=sample_alpha,
            linewidth=0.5,
            label="Posterior samples" if i == 0 else None,
        )

    # Ground truth
    ax.plot(
        w_ground_truth,
        20 * np.log10(np.abs(h_ground_truth)),
        color=gt_color_matplotlib,
        linewidth=1.0,
        label="Ground truth",
    )

    # Estimated mean
    ax.plot(
        w_mean_est,
        20 * np.log10(np.abs(h_mean_est)),
        color=est_color_matplotlib,
        linewidth=1.0,
        linestyle="--",
        label="Estimated mean",
    )

    ax.set_xlabel("Frequency [rad/sample]")
    ax.set_ylabel("Magnitude [dB]")

    if include_title_in_plots:
        ax.set_title("Frequency Response of Estimated IR and Posterior Samples")

    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plots_file, dpi=300)
    plt.show()


def plot_group_delay_plotly(
    taps: np.ndarray,  # shape → (n_samples, numtaps)
    fir_ground_truth: Union[np.ndarray, list],  # (numtaps,)
    fir_mean_est: Union[np.ndarray, list],  # (numtaps,)
    plots_file: Path,
    include_title_in_plots: bool = True,
    opacity_samples: float = 0.02,
) -> None:
    """
    Plot group delay for:
      • every posterior-sample FIR (faint lines)
      • the posterior mean / MAP estimate (solid, thick)
      • the ground-truth FIR (solid, thick, different colour)

    Args:
    taps : ndarray (n_samples, numtaps) Each row is one FIR coefficient vector from your posterior samples.
    fir_ground_truth : array-like (numtaps,) The “true” FIR you generated data with (single vector).
    fir_mean_est : array-like (numtaps,) Posterior mean / MAP estimate (single vector).
    plots_file : Path Where to save the static image (e.g. PNG, PDF…).
    include_title_in_plots : bool, optional If True, add a title to the figure.  Default = True.
    opacity_samples : float, optional Alpha for the many posterior-sample traces.  Default = 0.02.
    """
    fig = go.Figure()

    # ── posterior-sample traces ────────────────────────────────────────────
    for i in range(taps.shape[0]):
        w_s, gd_s = group_delay((taps[i], 1.0))  # a=1 for FIR
        fig.add_trace(
            go.Scatter(
                x=w_s,
                y=gd_s,
                mode="lines",
                line=dict(color=fir_mean_est_color_plotly),
                name="Posterior samples" if i == 0 else None,
                showlegend=(i == 0),
                opacity=opacity_samples,
            )
        )

    # ── posterior-mean / MAP estimate ─────────────────────────────────────
    w_mean, gd_mean = group_delay((np.asarray(fir_mean_est), 1.0))
    fig.add_trace(
        go.Scatter(
            x=w_mean,
            y=gd_mean,
            mode="lines",
            line=dict(color=fir_mean_est_color_plotly, width=2),
            name="Estimated mean",
        )
    )

    # ── ground-truth filter ───────────────────────────────────────────────
    w_gt, gd_gt = group_delay((np.asarray(fir_ground_truth), 1.0))
    fig.add_trace(
        go.Scatter(
            x=w_gt,
            y=gd_gt,
            mode="lines",
            line=dict(color=fir_ground_truth_color_plotly, width=2),
            name="Ground truth",
        )
    )

    # ── layout & output ───────────────────────────────────────────────────
    fig.update_layout(
        title=(
            "Group Delay of Estimated IR and Posterior Samples"
            if include_title_in_plots
            else None
        ),
        height=600,
        width=800,
        xaxis_title="Frequency [rad/sample]",
        yaxis_title="Group delay [samples]",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        yaxis=dict(range=[-100, 100]),
    )

    fig.write_image(str(plots_file))  # static image (png/svg/pdf… depending on suffix)
    fig.show()  # interactive view in notebook / browser


def plot_denoised_posterior_received_signal_plotly(
    denoised_samples: np.ndarray,
    received: np.ndarray,
    received_noisy: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    fig: go.Figure = go.Figure()
    i: int

    for i in range(denoised_samples.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(received)),
                y=denoised_samples[i],
                line=dict(color="green"),
                name=None if i > 0 else "samples from posterior",
                showlegend=False if i > 0 else True,
                opacity=0.01,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(received)),
            y=received,
            line=dict(color=received_color_plotly),
            name="Ground Truth Received (no noise)",
        )
    )

    fig.update_layout(
        go.Layout(
            title=(
                "Samples from Posterior of Received Signal Compared to Ground Truth and Noisy Received Signal"
                if include_title_in_plots
                else None
            ),
            height=450,
            width=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(received_noisy)),
            y=received_noisy,
            line=dict(color=received_noisy_color_plotly),
            name="Observed Received (with noise)",
            opacity=0.5,
        )
    )

    fig.write_image(plots_file)
    fig.show()


def plot_denoised_posterior_received_signal_matplotlib(
    denoised_samples: np.ndarray,
    received: np.ndarray,
    received_noisy: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    sample_alpha: float = 0.01,
    width: float = 12.0,
    height: float = 4.5,
    dpi: float = 300.0,
) -> None:
    """
    Plot samples from the posterior of the received signal (faint lines), plus the ground truth received signal and the
    noisy observed signal.
    """
    cm: float = 1 / 2.54
    fig, ax = plt.subplots(figsize=(width * cm, height * cm), dpi=dpi)

    x: np.ndarray = np.arange(len(received))
    x_noisy: np.ndarray = np.arange(len(received_noisy))
    i: int

    # Posterior samples
    for i in range(denoised_samples.shape[0]):
        ax.plot(
            x,
            denoised_samples[i, : len(received)],
            color=est_color_matplotlib,
            alpha=sample_alpha,
            linewidth=0.5,
            label="Samples from posterior" if i == 0 else None,
        )

    # Ground truth (no noise)
    ax.plot(
        x,
        received,
        color=gt_color_matplotlib,
        linewidth=1.0,
        label="Ground Truth Received (no noise)",
    )

    # Observed noisy signal
    ax.plot(
        x_noisy,
        received_noisy,
        color=received_noisy_color_matplotlib,
        linewidth=1.0,
        alpha=0.5,
        label="Observed Received (with noise)",
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")

    if include_title_in_plots:
        ax.set_title(
            "Samples from Posterior of Received Signal Compared to Ground Truth and Noisy Observations"
        )

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plots_file, dpi=300)
    plt.show()


def plot_estimated_ccf_and_posterior_samples_plotly(
    samples: np.ndarray,
    source_auto_corr: np.ndarray,
    actual_c_pad: np.ndarray,
    actual_c_noisy_pad: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    skip_idx: int = 100,
) -> None:
    i: int

    for i in range(0, 400, skip_idx):
        fig = go.Figure()

        j: int
        for j in range(samples.shape[0]):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(skip_idx),
                    y=np.convolve(
                        source_auto_corr,
                        samples[j],
                    )[i : i + skip_idx],
                    line=dict(color=fir_mean_est_color_plotly),
                    name=None if j > 0 else "Samples from Posterior",
                    showlegend=False if j > 0 else True,
                    opacity=0.01,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=np.arange(skip_idx),
                y=actual_c_pad[i : i + skip_idx],
                line=dict(color=received_color_plotly),
                name="CCF Ground Truth (no noise)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(skip_idx),
                y=actual_c_noisy_pad[i : i + skip_idx],
                line=dict(color=received_noisy_color_plotly),
                name="CCF Observed (with noise)",
            )
        )

        fig.update_layout(
            go.Layout(
                title=(
                    f"Samples from Posterior of CCF Compared to Ground Truth and Observed Noisy {i} to {i + 100}"
                    if include_title_in_plots
                    else None
                ),
                height=450,
                width=1200,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79),
            )
        )

        fig.write_image(plots_file)
        fig.show()


def plot_estimated_ccf_and_posterior_samples_matplotlib(
    samples: np.ndarray,
    source_auto_corr: np.ndarray,
    actual_c_pad: np.ndarray,
    actual_c_noisy_pad: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    skip_idx: int = 100,
    sample_alpha: float = 0.01,
    width: float = 12.0,
    height: float = 4.5,
    dpi: float = 300.0,
) -> None:
    """
    For each window [i : i+skip_idx] (i stepping 0..400):
      - plot `num_samples` posterior-sample cross-correlations (faint lines),
      - overlay the true CCF and the noisy observed CCF.
    """
    cm: float = 1 / 2.54  # cm→inch
    i: int

    for i in range(0, 400, skip_idx):
        # create new figure for this window
        fig, ax = plt.subplots(figsize=(width * cm, height * cm), dpi=dpi)
        x: np.ndarray = np.arange(skip_idx)
        j: int

        # posterior-sample CCFs
        for j in range(samples.shape[0]):
            conv: np.ndarray = np.convolve(source_auto_corr, samples[j])
            y: np.ndarray = conv[i : i + skip_idx]
            ax.plot(
                x,
                y,
                color=est_color_matplotlib,
                alpha=sample_alpha,
                linewidth=0.5,
                label="Posterior samples" if j == 0 else None,
            )

        # ground-truth CCF (no noise)
        ax.plot(
            x,
            actual_c_pad[i : i + skip_idx],
            color=gt_color_matplotlib,
            linewidth=1.0,
            label="CCF Ground Truth (no noise)",
        )

        # observed noisy CCF
        ax.plot(
            x,
            actual_c_noisy_pad[i : i + skip_idx],
            color=received_noisy_color_matplotlib,
            linewidth=1.0,
            alpha=0.5,
            label="CCF Observed (with noise)",
        )

        # labels & title
        ax.set_xlabel("Lag index")
        ax.set_ylabel("Cross-correlation")

        if include_title_in_plots:
            ax.set_title(
                f"Posterior CCF Samples vs Truth & Noise (lags {i}–{i+skip_idx})"
            )

        ax.legend(loc="upper right", frameon=False)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout(pad=0.2)

        # save and close
        save_path = plots_file.with_name(f"{plots_file.stem}_{i}{plots_file.suffix}")
        fig.savefig(save_path, dpi=300)
        plt.show()

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    }
)


source_color_matplotlib: str = "C0"
receiver_color_matplotlib: str = "C1"
gt_color_matplotlib: str = "C2"
est_ir_color_matplotlib: str = "C3"
est_ccf_color_matplotlib: str = "C4"


def plot_pair_segment_plotly(
    seg1: np.ndarray,
    seg2: np.ndarray,
    seg_idx: int,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    fig: go.Figure = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(seg1)),
            y=seg1,
            line=dict(color="red"),
            name=f"Source Pair {seg_idx}",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(seg2)),
            y=seg2,
            line=dict(color="purple"),
            name=f"Receiver Pair {seg_idx}",
        )
    )

    fig.update_layout(
        title="Pair Segment Example" if include_title_in_plots else None,
        height=450,
        width=1200,
        template="plotly_white",
    )

    fig.write_image(plot_file)
    fig.show()


def plot_pair_segment_matplotlib(
    seg1: np.ndarray,
    seg2: np.ndarray,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """
    Plot two paired segments (source vs receiver) side-by-side,
    saving a publication-quality figure similar to the Plotly version.
    """
    cm = 1 / 2.54  # inches per cm
    fig, ax = plt.subplots(figsize=(12 * cm, 4.5 * cm), dpi=300)

    x1 = np.arange(len(seg1))
    x2 = np.arange(len(seg2))

    ax.plot(x1, seg1, color=source_color_matplotlib, linewidth=1.0, label=f"Source")

    ax.plot(x2, seg2, color=receiver_color_matplotlib, linewidth=1.0, label=f"Receiver")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")

    if include_title_in_plots:
        ax.set_title("Pair Segment Example")

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)


def plot_pair_psd_plotly(
    source_stats_db, receiver_stats_db, include_title_in_plots, plot_file
):
    fig = go.Figure()

    # Plot Source PSD: Mean line and ± STD band in dB
    fig.add_trace(
        go.Scatter(
            x=source_stats_db["freq"],
            y=source_stats_db["psd_mean_db"],
            mode="lines",
            name="Source PSD Mean (dB)",
            line=dict(color="red"),
        )
    )

    # Shaded region for source ± STD in dB
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([source_stats_db["freq"], source_stats_db["freq"][::-1]]),
            y=np.concatenate(
                [
                    source_stats_db["psd_mean_db"] - source_stats_db["psd_std_db"],
                    (source_stats_db["psd_mean_db"] + source_stats_db["psd_std_db"])[
                        ::-1
                    ],
                ]
            ),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Source ± STD (dB)",
        )
    )

    # Plot Receiver PSD: Mean line and ± STD band in dB
    fig.add_trace(
        go.Scatter(
            x=receiver_stats_db["freq"],
            y=receiver_stats_db["psd_mean_db"],
            mode="lines",
            name="Receiver PSD Mean (dB)",
            line=dict(color="purple"),
        )
    )

    # Shaded region for receiver ± STD in dB
    fig.add_trace(
        go.Scatter(
            x=np.concatenate(
                [receiver_stats_db["freq"], receiver_stats_db["freq"][::-1]]
            ),
            y=np.concatenate(
                [
                    receiver_stats_db["psd_mean_db"] - receiver_stats_db["psd_std_db"],
                    (
                        receiver_stats_db["psd_mean_db"]
                        + receiver_stats_db["psd_std_db"]
                    )[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Receiver ± STD (dB)",
        )
    )

    fig.update_layout(
        title=(
            "PSD Statistics (in dB) for Source and Receiver"
            if include_title_in_plots
            else None
        ),
        xaxis_title="Frequency (Hz)",
        yaxis_title="PSD (dB)",
        template="plotly_white",
    )

    fig.write_image(plot_file)
    fig.show()


def plot_estimated_ir_plotly(
    mean_ir: np.ndarray,
    std_ir: np.ndarray,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    kernel_size: int = len(mean_ir)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.arange(kernel_size),
                y=mean_ir + std_ir,
                mode="lines",
                line=dict(color="lightgreen"),
                name="mean + std",
            ),
            go.Scatter(
                x=np.arange(kernel_size),
                y=mean_ir - std_ir,
                mode="lines",
                line=dict(color="lightgreen"),
                name="mean - std",
                fill="tonexty",
            ),
            go.Scatter(
                x=np.arange(kernel_size),
                y=mean_ir,
                line=dict(color="green"),
                name="estimated mean",
            ),
        ]
    )

    fig.update_layout(
        title="Estimated impulse response" if include_title_in_plots else None,
        height=800,
        width=1200,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.79),
        template="plotly_white",
    )

    fig.write_image(plot_file)
    fig.show()


def plot_estimated_ir_matplotlib(
    mean_ir: np.ndarray,
    std_ir: np.ndarray,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """Estimated impulse response ±1σ band + mean."""
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12 * cm, 4.5 * cm), dpi=300)

    x = np.arange(len(mean_ir))
    # ±1σ envelope
    ax.fill_between(
        x,
        mean_ir - std_ir,
        mean_ir + std_ir,
        color=est_ir_color_matplotlib,
        alpha=0.5,
        label="Mean ± Std",
    )
    # mean line
    ax.plot(
        x,
        mean_ir,
        color=est_ir_color_matplotlib,
        linewidth=1.0,
        label="Estimated mean",
    )

    ax.set_xlabel("Tap index")
    ax.set_ylabel("Coefficient value")
    if include_title_in_plots:
        ax.set_title("Estimated Impulse Response")

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)


def plot_freq_response_plotly(
    w_hat,
    amplitude_model_h_hat_mean,
    include_title_in_plots: bool,
    plot_file: Path,
    freq_responses_samples: Optional[np.ndarray] = None,
    amplitude_mean_posterior: Optional[np.ndarray] = None,
    amplitude_median_posterior: Optional[np.ndarray] = None,
    amplitude_lower_ci: Optional[np.ndarray] = None,
    amplitude_upper_ci: Optional[np.ndarray] = None,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()

    if freq_responses_samples is not None:
        # Plot each posterior sample with very low opacity.
        for i in range(freq_responses_samples.shape[0]):
            freq_response = freq_responses_samples[i, :]
            fig.add_trace(
                go.Scatter(
                    x=w_hat,
                    y=freq_response,
                    mode="lines",
                    line=dict(color="green"),
                    name="Posterior samples" if i == 0 else None,
                    showlegend=(i == 0),
                    opacity=0.01,
                )
            )

    # Plot the model's estimated mean (kernel mean) in orange.
    fig.add_trace(
        go.Scatter(
            x=w_hat,
            y=amplitude_model_h_hat_mean,
            mode="lines",
            line=dict(color="green"),
            name="Estimated Mean (model kernel mean)",
        )
    )

    if amplitude_mean_posterior is not None:
        # Plot the posterior sample based mean.
        fig.add_trace(
            go.Scatter(
                x=w_hat,
                y=amplitude_mean_posterior,
                mode="lines",
                line=dict(color="green"),
                name="Posterior Mean (samples)",
            )
        )

    if amplitude_median_posterior is not None:
        # Plot the posterior sample based median.
        fig.add_trace(
            go.Scatter(
                x=w_hat,
                y=amplitude_median_posterior,
                mode="lines",
                line=dict(color="lightgreen"),
                name="Posterior Median (samples)",
            )
        )

    if amplitude_lower_ci is not None:
        # Plot the lower bound of the 95% credible interval.
        fig.add_trace(
            go.Scatter(
                x=w_hat,
                y=amplitude_lower_ci,
                mode="lines",
                line=dict(color="black", dash="dot"),
                name="2.5% CI",
            )
        )

    if amplitude_upper_ci is not None:
        # Plot the upper bound of the 95% credible interval.
        fig.add_trace(
            go.Scatter(
                x=w_hat,
                y=amplitude_upper_ci,
                mode="lines",
                line=dict(color="black", dash="dot"),
                name="97.5% CI",
            )
        )

    # Update layout.
    fig.update_layout(
        title=(
            "Frequency Response of the Estimated Impulse Response"
            if include_title_in_plots
            else None
        ),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude (dB)",
        height=800,
        width=1200,
        legend=dict(yanchor="top", y=0.17, xanchor="left", x=0.73),
        template="plotly_white",
    )

    # Save the plot to file and show it.
    fig.write_image(plot_file)
    fig.show()


def plot_ccf_plotly(
    cc_mean: np.ndarray,
    cc_std: np.ndarray,
    num_freq: int,
    fs: float,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """
    Plot the cross-correlation function ±1σ band using Plotly.

    Args:
        cc_mean:                   The mean cross-correlation array.
        cc_std:               The standard-deviation array for the cross-correlation.
        num_freq:             The number of points to plot from cc and cc_std.
        fs:                   The sampling frequency (to convert index to time in seconds).
        include_title_in_plots: If True, includes a title in the plot.
        plot_file:            The file path to save the plot image.
    """
    # Compute the time axis.
    t = np.arange(num_freq) / fs

    # Make sure cc_std is at least as long as cc
    if cc_std.shape[0] < num_freq:
        raise ValueError(
            f"cc_std must have at least {num_freq} points, got {cc_std.shape[0]}"
        )

    # Build ±1σ envelope
    upper = cc_mean[:num_freq] + cc_std[:num_freq]
    lower = cc_mean[:num_freq] - cc_std[:num_freq]

    # Create the figure
    fig = go.Figure()

    # Plot the ±1σ filled band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,255,0.2)",
            line=dict(color="blue"),
            name="CCF ±1σ",
        )
    )

    # Plot the mean CCF on top
    fig.add_trace(
        go.Scatter(
            x=t,
            y=cc_mean[:num_freq],
            mode="lines",
            line=dict(color="blue"),
            name="CCF mean",
        )
    )

    # Layout
    fig.update_layout(
        title="Cross-Correlation Function ±1σ" if include_title_in_plots else None,
        xaxis_title="Time [s]",
        yaxis_title="Amplitude",
        width=1200,
        height=800,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        template="plotly_white",
    )

    # Save and show
    fig.write_image(str(plot_file))
    fig.show()


def plot_ccf_matplotlib(
    cc_mean: np.ndarray,
    cc_std: np.ndarray,
    num_freq: int,
    fs: float,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """Cross-correlation function ±1σ band + mean."""
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12 * cm, 4.5 * cm), dpi=300)

    t = np.arange(num_freq) / fs
    upper = cc_mean[:num_freq] + cc_std[:num_freq]
    lower = cc_mean[:num_freq] - cc_std[:num_freq]

    # ±1σ envelope
    ax.fill_between(
        t,
        lower,
        upper,
        color=est_ccf_color_matplotlib,
        alpha=0.5,
        label="CCF ±1σ",
    )
    # mean line
    ax.plot(
        t,
        cc_mean[:num_freq],
        color=est_ccf_color_matplotlib,
        linewidth=1.0,
        label="CCF mean",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    if include_title_in_plots:
        ax.set_title("Cross-Correlation Function ±1σ")

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)


def plot_relative_uncertainty_plotly(
    ir_mean: np.ndarray,
    ir_std: np.ndarray,
    cc_mean: np.ndarray,
    cc_std: np.ndarray,
    fs: float,
    include_title: bool,
    plot_file: Path,
) -> None:
    """
    Plot the relative (fractional) ±1σ uncertainty for both the estimated impulse response
    and the cross-correlation on a shared time axis, overlaying both bands for direct comparison.

    Args:
        ir_mean:    Mean impulse-response samples.
        ir_std:     1σ of impulse-response samples.
        cc_mean:    Mean cross-correlation samples.
        cc_std:     1σ of cross-correlation samples.
        fs:         Sampling frequency (Hz) for converting indices to time in seconds.
        include_title: Whether to include a title in the plot.
        plot_file:  Path to save the generated image.
    """
    # Build time axes
    t_ir = np.arange(ir_mean.size) / fs
    t_cc = np.arange(cc_mean.size) / fs

    # Compute fractional uncertainty (σ relative to each curve's peak)
    peak_ir = np.max(np.abs(ir_mean))
    peak_cc = np.max(np.abs(cc_mean))
    ir_frac_std = ir_std / peak_ir
    cc_frac_std = cc_std / peak_cc

    # Determine common y-axis range (0 to max_frac)
    max_frac = max(ir_frac_std.max(), cc_frac_std.max()) * 1.1  # add 10% headroom

    # Create figure
    fig = go.Figure()

    # IR fractional uncertainty band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_ir, t_ir[::-1]]),
            y=np.concatenate([ir_frac_std, ir_frac_std[::-1]]),
            fill="toself",
            fillcolor="rgba(0,128,0,0.2)",
            line=dict(color="green"),
            name="IR ±1σ (frac)",
        )
    )

    # CCF fractional uncertainty band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_cc, t_cc[::-1]]),
            y=np.concatenate([cc_frac_std, cc_frac_std[::-1]]),
            fill="toself",
            fillcolor="rgba(0,0,255,0.2)",
            line=dict(color="blue"),
            name="CCF ±1σ (frac)",
        )
    )

    # Layout adjustments
    fig.update_layout(
        title="Relative Uncertainty: IR vs CCF" if include_title else None,
        xaxis_title="Time [s]",
        yaxis_title="Fractional ±1σ",
        yaxis=dict(range=[0, max_frac]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        width=1200,
        height=800,
    )

    # Save and show
    fig.write_image(str(plot_file))
    fig.show()


def plot_relative_uncertainty_matplotlib(
    ir_mean: np.ndarray,
    ir_std: np.ndarray,
    cc_mean: np.ndarray,
    cc_std: np.ndarray,
    fs: float,
    include_title: bool,
    plot_file: Path,
) -> None:
    """
    Plot the relative (fractional) ±1σ uncertainty for both the estimated impulse response
    and the cross-correlation on a shared time axis, overlaying both bands for direct comparison.

    Args:
        ir_mean:    Mean impulse-response samples.
        ir_std:     1σ of impulse-response samples.
        cc_mean:    Mean cross-correlation samples.
        cc_std:     1σ of cross-correlation samples.
        fs:         Sampling frequency (Hz) for converting indices to time in seconds.
        include_title: Whether to include a title in the plot.
        plot_file:  Path to save the generated image.
    """
    # Build time axes
    t_ir = np.arange(ir_mean.size) / fs
    t_cc = np.arange(cc_mean.size) / fs

    # Compute fractional uncertainty (σ relative to each curve's peak)
    peak_ir = np.max(np.abs(ir_mean))
    peak_cc = np.max(np.abs(cc_mean))
    ir_frac_std = ir_std / peak_ir
    cc_frac_std = cc_std / peak_cc

    # Determine common y-axis range (0 to max_frac)
    max_frac = max(ir_frac_std.max(), cc_frac_std.max()) * 1.1  # add 10% headroom

    # Start plotting
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12 * cm, 8 * cm), dpi=300)

    # IR fractional ±1σ band
    ax.fill_between(
        t_ir,
        ir_frac_std,
        color=est_ir_color_matplotlib,
        alpha=0.2,
        step=None,
        label="IR ±1σ (frac)",
    )

    # CCF fractional ±1σ band
    ax.fill_between(
        t_cc,
        cc_frac_std,
        color=est_ccf_color_matplotlib,
        alpha=0.2,
        step=None,
        label="CCF ±1σ (frac)",
    )

    # Zero baseline
    ax.axhline(0, color="black", linewidth=0.5)

    # Labels and title
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Fractional ±1σ")
    if include_title:
        ax.set_title("Relative Uncertainty: IR vs CCF")

    ax.set_ylim(0, max_frac)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout(pad=0.2)
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)


def plot_tests_target_velocity_error_plotly(
    results_df: pd.DataFrame,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """
    Plot the cross-correlation function using Plotly.

    Args:
        cc: The cross-correlation array.
        num_freq: The number of frequency/time points to plot.
        fs: The sampling frequency (to convert index to time in seconds).
        include_title_in_plots: If True, includes a title in the plot.
        plot_file: The file path to save the plot image.
    """
    # Create the figure and add a trace for the ccf.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df["num_pairs"],
            y=results_df["ccf_target_error"],
            mode="lines",
            line=dict(color="blue"),
            name="CCF",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=results_df["num_pairs"],
            y=results_df["mir_target_error"],
            mode="lines",
            line=dict(color="green"),
            name="Mean IR",
        )
    )

    # Update the layout with labels and an optional title.
    fig.update_layout(
        title="Target velocity mean mismatch" if include_title_in_plots else None,
        xaxis_title="num pairs",
        yaxis_title="Mean mismatch",
        width=1200,
        height=800,
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
        template="plotly_white",
    )

    # Save the plot to file and display it.
    fig.write_image(plot_file)
    fig.show()


def plot_tests_target_velocity_error_matplotlib(
    results_df: pd.DataFrame,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """
    Plot target velocity mean mismatch vs number of pairs using Matplotlib,
    replicating the Plotly version in publication-quality style.
    """
    # Figure size: 12 cm × 8 cm
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12 * cm, 8 * cm), dpi=300)

    x = results_df["num_pairs"]

    ax.plot(
        x,
        results_df["ccf_target_error"],
        color=est_ccf_color_matplotlib,
        linewidth=1.0,
        label="CCF",
    )
    ax.plot(
        x,
        results_df["mir_target_error"],
        color=est_ir_color_matplotlib,
        linewidth=1.0,
        label="Mean IR",
    )

    ax.set_xlabel("num pairs")
    ax.set_ylabel("Mean mismatch")

    if include_title_in_plots:
        ax.set_title("Target velocity mean mismatch")

    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)


def plot_velocity_curve(
    velocity_func,
    include_title_in_plots,
    plot_file,
    freqs=None,
    vels=None,
):
    """
    Plots the velocity curve using Plotly.

    Args:
        velocity_func: A function that maps frequency to velocity.
        freqs: Optional list/array of frequencies (for data points).
        vels: Optional list/array of velocities corresponding to freqs.
    """
    # Create a range of frequency values from 0 to 10 Hz.
    f_range = np.linspace(0, 10, 200)
    velocities = velocity_func(f_range)

    # Create a Plotly figure.
    fig = go.Figure()

    # Add the fitted velocity curve.
    fig.add_trace(
        go.Scatter(
            x=f_range,
            y=velocities,
            mode="lines",
            line=dict(color="orange"),
            name="Fitted Velocity Curve",
        )
    )

    # Optionally, add the original data points.
    if freqs is not None and vels is not None:
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=vels,
                mode="markers",
                marker=dict(color="black"),
                name="Data Points",
            )
        )

    # Update layout settings.
    fig.update_layout(
        title="Velocity Curve from 0 to 10 Hz" if include_title_in_plots else None,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Velocity (m/s)",
        template="plotly_white",
    )

    # Save the plot to file and show it.
    fig.write_image(plot_file)
    fig.show()


def plot_misfit_with_velocity_plotly(
    misfit,
    freqs,
    min_eval_velocity,
    max_eval_velocity,
    velocity_true_or_func,
    phase_velocity_mean_ir,
    w_hat,
    title,
    include_title_in_plots,
    plot_file,
) -> None:
    """
    Plots a misfit heatmap with an overlaid velocity function curve.
    Only the velocity function appears in the legend.

    Args:
        misfit (np.ndarray): 2D array of misfit values.
        freqs (array-like): Frequency range (e.g., [min, max]).
        min_eval_velocity (float): Minimum velocity value for y-axis.
        max_eval_velocity (float): Maximum velocity value for y-axis.
        velocity_true_or_func (callable): Function mapping frequency to velocity.
        title (str): Title for the plot.
        include_title_in_plots (bool): If True, includes titles for the subplots.
        plot_file: File path to save the resulting plot image.
    """
    # Create frequency values for velocity function overlay.
    velocities: np.ndarray

    if isinstance(velocity_true_or_func, float):
        velocities = np.full_like(freqs, velocity_true_or_func)
    else:
        velocities = velocity_true_or_func(freqs)

    # Define x and y axes for the heatmap.
    x_axis = np.linspace(freqs[0], freqs[-1], misfit.shape[1])
    y_axis = np.linspace(min_eval_velocity, max_eval_velocity, misfit.shape[0])

    fig = go.Figure()

    # Add heatmap trace (without legend).
    fig.add_trace(
        go.Heatmap(
            z=misfit,
            x=x_axis,
            y=y_axis,
            colorscale="Greys",
            showscale=True,
            colorbar=dict(
                title="Misfit",
                len=0.5,  # set the colorbar length to 50% of the plot height
                y=0.5,  # center the colorbar vertically
            ),
            showlegend=False,
        )
    )

    # Overlay the velocity function curve.
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=velocities,
            mode="lines",
            line=dict(color="orange"),
            name="True Velocity",
            legendgroup="velocity",
            showlegend=True,
        )
    )

    # Optionally overlay the phase velocity curve.
    if phase_velocity_mean_ir is not None:
        fig.add_trace(
            go.Scatter(
                x=w_hat[w_hat > 0.3],
                y=phase_velocity_mean_ir[w_hat > 0.3],
                mode="lines",
                name="Naive Phase Velocity Mean IR",
                line=dict(color="green"),
                legendgroup="phase",
            )
        )

    # Update layout with axis titles and square-ish dimensions.
    fig.update_layout(
        title=title if include_title_in_plots else None,
        xaxis_title="Frequency [Hz]",
        yaxis_title="Velocity [m/s]",
        width=800,
        height=800,
        template="plotly_white",
        legend=dict(
            orientation="h",  # horizontal legend
            yanchor="top",
            y=-0.2,  # position the legend below the plot
            xanchor="center",
            x=0.5,
        ),
    )

    # Save the plot to file and show it.
    fig.write_image(plot_file)
    fig.show()


def plot_misfit_with_velocity_matplotlib(
    misfit: np.ndarray,
    freqs: np.ndarray,
    min_eval_velocity: float,
    max_eval_velocity: float,
    velocity_true_or_func,
    phase_velocity_mean_ir: Optional[np.ndarray],
    w_hat: Optional[np.ndarray],
    title: str,
    include_title_in_plots: bool,
    plot_file: Path,
) -> None:
    """
    Plots a misfit heatmap with an overlaid velocity function curve using Matplotlib.
    Only the velocity function and optional phase velocity appear in the legend.

    Args:
        misfit:                2D array of misfit values (shape: [n_velocities, n_frequencies]).
        freqs:                 1D array of frequency points corresponding to misfit columns.
        min_eval_velocity:     Minimum velocity for y-axis range.
        max_eval_velocity:     Maximum velocity for y-axis range.
        velocity_true_or_func: Float or callable mapping freqs -> velocity values.
        phase_velocity_mean_ir: 1D array of phase velocity estimates (same length as w_hat).
        w_hat:                 1D array of frequencies corresponding to phase_velocity_mean_ir.
        title:                 Plot title.
        include_title_in_plots: Whether to include the title.
        plot_file:             Path to save the figure.
    """
    # Determine velocities array
    if isinstance(velocity_true_or_func, (int, float)):
        velocities = np.full_like(freqs, velocity_true_or_func, dtype=float)
    else:
        velocities = velocity_true_or_func(freqs)

    # Create axes for heatmap
    n_v, n_f = misfit.shape
    x = np.linspace(freqs[0], freqs[-1], n_f)
    y = np.linspace(min_eval_velocity, max_eval_velocity, n_v)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot heatmap
    # Use extent to map array indices to (x, y) coordinates, origin lower so y increases upward
    im = ax.imshow(
        misfit,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="Greys",
    )
    # Colorbar: half the height, centered
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Misfit")

    # Overlay true velocity
    ax.plot(
        freqs, velocities, color=gt_color_matplotlib, linewidth=2, label="True Velocity"
    )

    # Overlay phase velocity if provided
    if phase_velocity_mean_ir is not None:
        mask = w_hat > 0.3
        ax.plot(
            w_hat[mask],
            phase_velocity_mean_ir[mask],
            color=est_ir_color_matplotlib,
            linewidth=2,
            label="Naive Phase Velocity Mean IR",
        )

    # Labels and title
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Velocity [m/s]")
    if include_title_in_plots:
        ax.set_title(title)

    # Legend below plot, horizontal
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()


def plot_error_vectors(
    error_vec_ccf: np.ndarray,
    error_vec_mir: np.ndarray,
    freqs: Optional[np.ndarray],
    include_title_in_plots: bool,
    plot_file: Path,
):
    """
    Plots error_vec_ccf and error_vec_mir as grouped bars side by side using Plotly.

    Args:
        error_vec_ccf (np.ndarray or list): Error vector for the CCF model.
        error_vec_mir (np.ndarray or list): Error vector for the MIR model.
        freqs (np.ndarray or list, optional): Frequency values corresponding to each error value.
                                              If not provided, the indices of the error vectors are used.
        include_title_in_plots (bool): If True, includes a title in the plot.
        plot_file (str): File path to save the resulting plot image.
    """
    # Use indices if no frequency values are provided.
    if freqs is None:
        freqs = np.arange(len(error_vec_ccf))

    # Create grouped bar chart with specified colors.
    fig = go.Figure(
        data=[
            go.Bar(name="CCF Error", x=freqs, y=error_vec_ccf, marker_color="blue"),
            go.Bar(name="MIR Error", x=freqs, y=error_vec_mir, marker_color="green"),
        ]
    )

    # Update layout for grouped bars.
    fig.update_layout(
        title="CC and MIR Error by Frequency" if include_title_in_plots else None,
        xaxis_title="Frequency" if freqs is not None else "Frequency Bin",
        yaxis_title="Error Value",
        barmode="group",
        template="plotly_white",
    )

    # Save the plot to file and show it.
    fig.write_image(plot_file)
    fig.show()


def plot_error_vectors_matplotlib(
    error_vec_ccf: np.ndarray,
    error_vec_mir: np.ndarray,
    freqs: np.ndarray = None,
    include_title_in_plots: bool = True,
    plot_file: Path = Path("error_vectors.png"),
) -> None:
    """
    Plot grouped bar charts of CCF and MIR error vectors side by side using Matplotlib.

    If freqs is None or its length does not match the error vectors,
    the x-axis will default to integer indices of the error arrays.
    """
    # Ensure freqs matches error vector length
    n = len(error_vec_ccf)
    if freqs is None or len(freqs) != n:
        freqs = np.arange(n)

    # Determine bar width based on spacing
    if len(freqs) > 1:
        dx = np.min(np.diff(freqs))
    else:
        dx = 1.0
    width = dx * 0.4

    # Figure size: 12 cm × 8 cm
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12 * cm, 8 * cm), dpi=300)

    # Positions for grouped bars
    x_ccf = freqs - width / 2
    x_mir = freqs + width / 2

    # Plot bars
    ax.bar(
        x_ccf,
        error_vec_ccf,
        width=width,
        color=est_ccf_color_matplotlib,
        label="CCF Error",
    )
    ax.bar(
        x_mir,
        error_vec_mir,
        width=width,
        color=est_ir_color_matplotlib,
        label="MIR Error",
    )

    # Labels and title
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Error Value")
    if include_title_in_plots:
        ax.set_title("CC and MIR Error by Frequency")

    # Legend and grid
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, axis="y")

    fig.tight_layout(pad=0.2)
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)

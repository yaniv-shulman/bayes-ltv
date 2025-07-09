from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from experiments.plots_utils_common import (
    gt_color_matplotlib,
    fir_ground_truth_color_plotly,
    fir_mean_est_color_plotly,
    est_color_matplotlib,
)


def plot_fir_ground_truth_components_plotly(
    f_linear_phase1: np.ndarray,
    f_linear_phase2: np.ndarray,
    f_linear_phase3: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(f_linear_phase1)),
            y=f_linear_phase1,
            line=dict(color="violet"),
            name="Impulse Response 1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(f_linear_phase2)),
            y=f_linear_phase2,
            line=dict(color="purple"),
            name="Impulse Response 2",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(f_linear_phase1)),
            y=f_linear_phase3,
            line=dict(color="plum"),
            name="Impulse Response 3",
        )
    )

    fig.update_layout(
        go.Layout(
            title="Ground Truth FIR Components" if include_title_in_plots else None,
            height=600,
            width=800,
        )
    )

    fig.write_image(plots_file)
    fig.show()


def plot_fir_ground_truth_components_matplotlib(
    f_linear_phase1: np.ndarray,
    f_linear_phase2: np.ndarray,
    f_linear_phase3: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    """
    Plot three ground-truth FIR components as variants of a base color (gt_color),
    in a Matplotlib figure matching your publication style.
    """

    # Helper to lighten a color by blending with white
    def lighten_color(color, amount=0.5):
        c = np.array(mcolors.to_rgb(color))
        return tuple(c + (1 - c) * amount)

    # Generate three shades: base, +30% white, +60% white
    colors = [
        gt_color_matplotlib,
        lighten_color(gt_color_matplotlib, 0.3),
        lighten_color(gt_color_matplotlib, 0.6),
    ]

    # Figure & axes
    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(12 * cm, 8 * cm), dpi=300)

    x = np.arange(len(f_linear_phase1))

    ax.plot(
        x, f_linear_phase1, color=colors[0], linewidth=1.0, label="Impulse Response 1"
    )
    ax.plot(
        x, f_linear_phase2, color=colors[1], linewidth=1.0, label="Impulse Response 2"
    )
    ax.plot(
        x, f_linear_phase3, color=colors[2], linewidth=1.0, label="Impulse Response 3"
    )

    ax.set_xlabel("Tap index")
    ax.set_ylabel("Coefficient value")

    if include_title_in_plots:
        ax.set_title("Ground Truth FIR Components")

    ax.legend(loc="upper left", frameon=False, fontsize=5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout(pad=0.2)
    fig.savefig(plots_file, dpi=300)
    plt.close(fig)


def plot_ltv_fir_ground_truth_3d(
    ltv_ir_ground_truth: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    """
    Plot the LTV FIR ground truth in a 3D plot using Plotly.

    Args:
        ltv_ir_ground_truth: 2D array where each row is a time step and each column is a filter tap.
        plots_file: The file path to save the plot.
        include_title_in_plots: bool, whether to include a title in the plot.
    """
    # Create a time axis for plotting
    num_time_steps = ltv_ir_ground_truth.shape[0]

    # Plotting the 3D response using Plotly
    fig = go.Figure()

    # Add traces for each tap in the FIR filter
    for i in range(ltv_ir_ground_truth.shape[1]):
        fig.add_trace(
            go.Scatter3d(
                x=np.arange(ltv_ir_ground_truth.shape[0]),
                y=np.full(num_time_steps, i),
                z=ltv_ir_ground_truth[:, i],
                mode="lines",
                name=f"Tap {i + 1}",
            )
        )

    # Set plot layout
    fig.update_layout(
        title="LTV Impulse Response Over Time" if include_title_in_plots else None,
        scene=dict(
            xaxis_title="Time Step",
            yaxis_title="Filter Tap Index",
            zaxis_title="Amplitude",
        ),
        width=800,
        height=600,
    )

    fig.write_image(plots_file)
    fig.show()


def plot_ltv_fir_ground_truth_3d_matplotlib(
    ltv_ir_ground_truth: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
) -> None:
    """
    3D plot of LTV FIR ground truth, using shades of base_color,
    with extra bottom margin so axis labels aren’t cut off.
    """

    # Helper to lighten a color by blending with white
    def lighten_color(color, amount=0.5):
        c = np.array(mcolors.to_rgb(color))
        return tuple(c + (1 - c) * amount)

    num_time_steps, num_taps = ltv_ir_ground_truth.shape

    # Generate a gradient of shades from base_color
    colors = [
        lighten_color(gt_color_matplotlib, i / max(1, num_taps - 1) * 0.7)
        for i in range(num_taps)
    ]

    # Figure setup
    cm = 1 / 2.54
    fig = plt.figure(figsize=(9 * cm, 8 * cm), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    t = np.arange(num_time_steps)
    for i in range(num_taps):
        ax.plot(
            t,
            np.full(num_time_steps, i),
            ltv_ir_ground_truth[:, i],
            color=colors[i],
            linewidth=1.0,
            label=f"Taps" if i == 0 else None,
        )

    # Axis labels
    ax.set_xlabel("Time Step", labelpad=10)
    ax.set_ylabel("Filter Tap Index", labelpad=10)
    ax.set_zlabel("Amplitude", labelpad=10)

    # Title
    if include_title_in_plots:
        ax.set_title("LTV Impulse Response Over Time")

    # Legend (only first for clarity)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Tight layout + extra bottom margin
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(bottom=0.15)  # push down the bottom of the axes

    fig.savefig(plots_file, dpi=300)
    plt.close(fig)


def plot_fir_fit_and_ground_truth_plotly(
    fir_ground_truth: np.ndarray,
    fir_mean_est: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    window_index: int = None,
) -> None:
    """
    Plot FIR ground truth and estimated IRs in 3D using Plotly.
    This function creates a 3D plot comparing the ground truth FIR coefficients with the estimated coefficients from a
    fitting algorithm. The plot includes lines for each filter tap index, with the x-axis representing time steps,
    the y-axis representing filter tap indices, and the z-axis representing the amplitude of the coefficients.

    Args:
        fir_ground_truth: 2D array of shape (T, M) where T is the number of time steps and M is the number of filter taps.`
        fir_mean_est: 2D array of shape (T, M) representing the estimated FIR coefficients.
        plots_file: Path to save the plotly figure.
        include_title_in_plots: bool, whether to include a title in the plot.
        window_index: int, optional, index of the window for which the plot is generated. If provided, it will be
            included in the title.
    """
    T, M = fir_ground_truth.shape
    x = np.arange(T)
    fig = go.Figure()

    # Ground truth lines
    for m in range(M):
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full(T, m),
                z=fir_ground_truth[:, m],
                mode="lines",
                line=dict(color=fir_ground_truth_color_plotly, width=2),
                name="Ground Truth" if m == 0 else None,
                showlegend=(m == 0),
            )
        )

    # Estimated lines
    for m in range(M):
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=np.full(T, m),
                z=fir_mean_est[:, m],
                mode="lines",
                line=dict(color=fir_mean_est_color_plotly, width=2),
                name="Estimated IR" if m == 0 else None,
                showlegend=(m == 0),
            )
        )

    # Title
    if include_title_in_plots:
        title = "IR Comparison"
        if window_index is not None:
            title += f" (window {window_index})"
        fig.update_layout(title=title)

    # Axes & layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Time Step",
            yaxis_title="Filter Tap Index",
            zaxis_title="Amplitude",
        ),
        height=600,
        width=800,
        legend=dict(x=0, y=1),
    )

    fig.write_image(str(plots_file))
    fig.show()


def plot_fir_fit_and_ground_truth_matplotlib(
    fir_ground_truth: np.ndarray,
    fir_mean_est: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    window_index: int = None,
) -> None:
    """
    Matplotlib 3D line plot comparing FIR ground truth and estimated coefficients.
    """
    # Figure size: 12 cm × 8 cm at 300 dpi
    cm = 1 / 2.54
    fig = plt.figure(figsize=(9 * cm, 8 * cm), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    T, M = fir_ground_truth.shape
    t = np.arange(T)

    # Ground truth lines
    for m in range(M):
        ax.plot(
            t,
            np.full(T, m),
            fir_ground_truth[:, m],
            color=gt_color_matplotlib,
            linewidth=1.0,
            label="Ground Truth" if m == 0 else None,
        )

    # Estimated mean lines (dashed)
    for m in range(M):
        ax.plot(
            t,
            np.full(T, m),
            fir_mean_est[:, m],
            color=est_color_matplotlib,
            linewidth=0.8,
            alpha=0.6,
            label="Estimated IR" if m == 0 else None,
        )

    # Axis labels
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Filter Tap Index")
    ax.set_zlabel("Amplitude")

    # Title
    if include_title_in_plots:
        title = "IR Comparison"
        if window_index is not None:
            title += f" (window {window_index})"
        ax.set_title(title)

    # Legend
    ax.legend(loc="upper left", frameon=False)

    # Layout and save
    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(bottom=0.15)  # push down the bottom of the axes
    fig.savefig(plots_file, dpi=300)
    plt.close(fig)

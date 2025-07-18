from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from plotly import graph_objects as go

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

source_color_plotly: str = "blue"
received_color_plotly: str = "orange"
received_noisy_color_plotly: str = "red"
fir_ground_truth_color_plotly: str = "purple"
fir_mean_est_color_plotly: str = "green"

source_color_matplotlib: str = "C0"
gt_color_matplotlib: str = "C1"
received_noisy_color_matplotlib: str = "C2"
est_color_matplotlib: str = "C4"


def plot_source_plotly(
    source: np.ndarray, plots_file: Path, include_title_in_plots: bool
) -> None:

    fig: go.Figure = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(source)),
            y=source,
            line=dict(color=source_color_plotly),
            name="source",
        )
    )

    fig.update_layout(
        go.Layout(
            title="Source Signal" if include_title_in_plots else None,
            height=450,
            width=1200,
        )
    )

    fig.write_image(plots_file)
    fig.show()


def plot_all_signals_plotly(
    source: np.ndarray,
    received: np.ndarray,
    received_noisy: np.ndarray,
    plots_file: Optional[Path],
    include_title_in_plots: bool,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(received)),
            y=source[: len(received)],
            line=dict(color=source_color_plotly),
            name="Source",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(received)),
            y=received,
            line=dict(color=received_color_plotly),
            name="Received",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(received_noisy)),
            y=received_noisy,
            line=dict(color=received_noisy_color_plotly),
            name="Received Noisy",
        )
    )

    fig.update_layout(
        go.Layout(
            title=(
                "Source Received and Received Noisy Data"
                if include_title_in_plots
                else None
            ),
            height=450,
            width=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.9),
        ),
    )

    if plots_file is not None:
        fig.write_image(plots_file)

    fig.show()


def plot_all_signals_matplotlib(
    source: np.ndarray,
    received: np.ndarray,
    received_noisy: np.ndarray,
    plots_file: Path,
    include_title_in_plots: bool,
    width: float = 12.0,
    height: float = 4.5,
    dpi: float = 300.0,
) -> None:
    """
    Plot source, received, and received_noisy signals using matplotlib,

    Args:
        source (np.ndarray): Source signal
        received (np.ndarray): Received signal
        received_noisy (np.ndarray): Received noisy signal
        plots_file (Path): Path to the plots file
        include_title_in_plots (bool): Include title in plot
        width (float): Width of the plot in cm
        height (float): Height of the plot in cm
        dpi (float): DPI of the plot
    """
    cm_to_inch = 1 / 2.54

    fig, ax = plt.subplots(
        figsize=(width * cm_to_inch, height * cm_to_inch),
        dpi=dpi,
    )

    x_rec = np.arange(len(received))
    x_rec_noisy = np.arange(len(received_noisy))

    ax.plot(
        x_rec,
        source[: len(received)],
        color=source_color_matplotlib,
        label="Source",
        linewidth=1.0,
    )
    ax.plot(
        x_rec,
        received,
        color=gt_color_matplotlib,
        label="Received",
        linewidth=1.0,
    )
    ax.plot(
        x_rec_noisy,
        received_noisy,
        color=received_noisy_color_matplotlib,
        label="Received Noisy",
        linewidth=1.0,
    )

    # Labels
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")

    # Optional title
    if include_title_in_plots:
        ax.set_title("Source, Received, and Received Noisy Data")

    # Legend in upper right, with minimal frame
    ax.legend(loc="upper right", frameon=False)

    # Optional grid for clarity
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Tight layout so labels/titles don't get cut off
    fig.tight_layout(pad=0.2)

    # Save and close
    fig.savefig(plots_file, format=plots_file.suffix.lstrip("."), dpi=300)
    plt.show()


def plot_training_loss_plotly(fit_results, include_title_in_plots, plot_file):
    # Access history dictionary
    history = fit_results.history

    # Create a list of epochs (starting from 1)
    epochs = list(range(1, len(history["loss"]) + 1))

    # Create a figure
    fig = go.Figure()

    # Plot training loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["loss"],
            mode="lines",
            line=dict(color="green"),
            name="Training Loss",
        )
    )

    # If validation loss is available, plot it as well
    if "val_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines",
                line=dict(color="green"),
                name="Validation Loss",
            )
        )

    # Customize layout
    fig.update_layout(
        title="Model Loss over Epochs" if include_title_in_plots else None,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
    )

    fig.write_image(plot_file)
    fig.show()

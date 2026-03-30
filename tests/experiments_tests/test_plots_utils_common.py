from pathlib import Path

import numpy as np
import pytest

from experiments import plots_utils_common as target


def test_plot_source_plotly_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.go.Figure, "write_image", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.go.Figure, "show", lambda self, *args, **kwargs: None)

    target.plot_source_plotly(
        source=np.array([1.0, 0.0, -1.0]),
        plots_file=test_artifacts_dir.joinpath("common_plot_source_plotly.png"),
        include_title_in_plots=False,
    )


def test_plot_all_signals_matplotlib_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.plt.Figure, "savefig", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.plt, "close", lambda *args, **kwargs: None)

    target.plot_all_signals_matplotlib(
        source=np.array([1.0, 2.0, 3.0, 4.0]),
        received=np.array([0.5, 1.5, 2.5]),
        received_noisy=np.array([0.4, 1.4, 2.4]),
        plots_file=test_artifacts_dir.joinpath(
            "common_plot_all_signals_matplotlib.png"
        ),
        include_title_in_plots=False,
    )


def test_plot_training_loss_plotly_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.go.Figure, "write_image", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.go.Figure, "show", lambda self, *args, **kwargs: None)

    class _FitResults:
        def __init__(self) -> None:
            self.history = {"loss": [3.0, 2.0, 1.0], "val_loss": [3.5, 2.5, 1.5]}

    fit_results = _FitResults()

    target.plot_training_loss_plotly(
        fit_results=fit_results,
        include_title_in_plots=False,
        plot_file=test_artifacts_dir.joinpath("common_plot_training_loss_plotly.png"),
    )

from pathlib import Path

import numpy as np
import pytest

from experiments.ltie_estimation import plots_utils as target


def test_plot_fir_ground_truth_plotly_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.go.Figure, "write_image", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.go.Figure, "show", lambda self, *args, **kwargs: None)

    target.plot_fir_ground_truth_plotly(
        fir_ground_truth=np.array([1.0, 0.5, -0.2]),
        plots_file=test_artifacts_dir.joinpath("ltie_plot_fir_ground_truth.png"),
        include_title_in_plots=False,
    )


def test_plot_fir_fit_and_ground_truth_matplotlib_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.plt.Figure, "savefig", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.plt, "close", lambda *args, **kwargs: None)

    target.plot_fir_fit_and_ground_truth_matplotlib(
        fir_mean_est=np.array([1.0, 0.4, -0.3]),
        fir_std_est=np.array([0.1, 0.1, 0.05]),
        fir_ground_truth=np.array([1.0, 0.5, -0.2]),
        plots_file=test_artifacts_dir.joinpath("ltie_plot_fir_fit_matplotlib.png"),
        include_title_in_plots=False,
    )

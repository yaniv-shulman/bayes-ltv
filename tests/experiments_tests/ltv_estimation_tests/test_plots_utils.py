from pathlib import Path

import numpy as np
import pytest

from experiments.ltv_estimation import plots_utils as target


def test_plot_fir_ground_truth_components_plotly_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.go.Figure, "write_image", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.go.Figure, "show", lambda self, *args, **kwargs: None)

    target.plot_fir_ground_truth_components_plotly(
        f_linear_phase1=np.array([1.0, 0.5, -0.1]),
        f_linear_phase2=np.array([0.8, 0.3, -0.2]),
        f_linear_phase3=np.array([0.7, 0.1, -0.3]),
        plots_file=test_artifacts_dir.joinpath("ltv_plot_fir_gt_components_plotly.png"),
        include_title_in_plots=False,
    )


def test_plot_fir_ground_truth_components_matplotlib_smoke(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.plt.Figure, "savefig", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.plt, "close", lambda *args, **kwargs: None)

    target.plot_fir_ground_truth_components_matplotlib(
        f_linear_phase1=np.array([1.0, 0.5, -0.1]),
        f_linear_phase2=np.array([0.8, 0.3, -0.2]),
        f_linear_phase3=np.array([0.7, 0.1, -0.3]),
        plots_file=test_artifacts_dir.joinpath(
            "ltv_plot_fir_gt_components_matplotlib.png"
        ),
        include_title_in_plots=False,
    )

from pathlib import Path

import numpy as np
import pytest

from experiments.ant import plots_utils as target


def test_plot_error_vectors_mismatched_lengths_raise(test_artifacts_dir: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        target.plot_error_vectors(
            error_vec_ccf=np.array([1.0, 2.0]),
            error_vec_mir=np.array([1.0]),
            freqs=np.array([1.0, 2.0]),
            include_title_in_plots=False,
            plot_file=test_artifacts_dir.joinpath("ant_plot_mismatch_lengths.png"),
        )


def test_plot_error_vectors_freq_length_mismatch_raises(
    test_artifacts_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="freqs length must match"):
        target.plot_error_vectors(
            error_vec_ccf=np.array([1.0, 2.0]),
            error_vec_mir=np.array([3.0, 4.0]),
            freqs=np.array([1.0]),
            include_title_in_plots=False,
            plot_file=test_artifacts_dir.joinpath("ant_plot_freq_mismatch_plotly.png"),
        )


def test_plot_error_vectors_matplotlib_freq_length_mismatch_raises(
    test_artifacts_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="freqs length must match"):
        target.plot_error_vectors_matplotlib(
            error_vec_ccf=np.array([1.0, 2.0]),
            error_vec_mir=np.array([3.0, 4.0]),
            freqs=np.array([1.0]),
            include_title_in_plots=False,
            plot_file=test_artifacts_dir.joinpath(
                "ant_plot_freq_mismatch_matplotlib.png"
            ),
        )


def test_plot_error_vectors_and_matplotlib_success_paths(
    monkeypatch: pytest.MonkeyPatch,
    test_artifacts_dir: Path,
) -> None:
    monkeypatch.setattr(
        target.go.Figure, "write_image", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.go.Figure, "show", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(
        target.plt.Figure, "savefig", lambda self, *args, **kwargs: None
    )
    monkeypatch.setattr(target.plt, "close", lambda *args, **kwargs: None)

    target.plot_error_vectors(
        error_vec_ccf=np.array([1.0, 2.0, 3.0]),
        error_vec_mir=np.array([1.5, 1.0, 0.5]),
        freqs=None,
        include_title_in_plots=False,
        plot_file=test_artifacts_dir.joinpath("ant_plot_success_plotly.png"),
    )

    target.plot_error_vectors_matplotlib(
        error_vec_ccf=np.array([1.0, 2.0, 3.0]),
        error_vec_mir=np.array([1.5, 1.0, 0.5]),
        freqs=np.array([1.0, 2.0, 3.0]),
        include_title_in_plots=False,
        plot_file=test_artifacts_dir.joinpath("ant_plot_success_matplotlib.png"),
    )

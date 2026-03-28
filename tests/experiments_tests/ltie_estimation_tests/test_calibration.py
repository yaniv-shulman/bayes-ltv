from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from experiments.ltie_estimation import calibration as target


def test_compute_uniform_batch_repetitions_success() -> None:
    actual_batch, actual_repeats = target.compute_uniform_batch_repetitions(
        num_independent_examples=3,
        batch_size_base=8,
    )

    assert actual_batch == 9
    assert actual_repeats == 3


@pytest.mark.parametrize(
    "num_independent_examples,batch_size_base,error_pattern",
    [
        (0, 8, "num_independent_examples must be positive"),
        (1, 0, "batch_size_base must be positive"),
    ],
)
def test_compute_uniform_batch_repetitions_invalid_inputs_raise(
    num_independent_examples: int,
    batch_size_base: int,
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        target.compute_uniform_batch_repetitions(
            num_independent_examples=num_independent_examples,
            batch_size_base=batch_size_base,
        )


def test_build_default_linear_phase_fir_returns_requested_numtaps() -> None:
    actual: np.ndarray = target.build_default_linear_phase_fir(numtaps=12)

    assert actual.shape == (12,)
    assert np.all(np.isfinite(actual))


def test_summarize_pointwise_interval_coverage_success() -> None:
    fir_ground_truth: np.ndarray = np.array([0.5, -0.5, 0.25])
    posterior_means: np.ndarray = np.array([[0.45, -0.45, 0.3], [0.55, -0.52, 0.2]])
    posterior_stds: np.ndarray = np.array([[0.1, 0.1, 0.1], [0.12, 0.1, 0.08]])

    actual: pd.DataFrame = target.summarize_pointwise_interval_coverage(
        fir_ground_truth=fir_ground_truth,
        posterior_means=posterior_means,
        posterior_stds=posterior_stds,
        interval_levels=(0.9, 0.95),
    )

    assert list(actual["interval_level"]) == [0.9, 0.95]
    assert np.all(
        (actual["empirical_coverage"] >= 0.0) & (actual["empirical_coverage"] <= 1.0)
    )
    assert np.all(actual["mean_interval_width"] > 0.0)


def test_summarize_pointwise_interval_coverage_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        target.summarize_pointwise_interval_coverage(
            fir_ground_truth=np.array([1.0, 2.0]),
            posterior_means=np.ones((2, 2)),
            posterior_stds=np.ones((2, 3)),
            interval_levels=(0.9,),
        )


def test_summarize_pointwise_interval_coverage_invalid_interval_raises() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        target.summarize_pointwise_interval_coverage(
            fir_ground_truth=np.array([1.0, 2.0]),
            posterior_means=np.ones((2, 2)),
            posterior_stds=np.ones((2, 2)),
            interval_levels=(1.0,),
        )


def test_fit_ltie_posterior_statistics_with_mocked_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTensor:
        def __init__(self, data: np.ndarray) -> None:
            self._data: np.ndarray = data

        def numpy(self) -> np.ndarray:
            return self._data

    class _FakeKernelPosterior:
        def mean(self) -> _FakeTensor:
            return _FakeTensor(np.array([1.0, 2.0, 3.0]))

        def variance(self) -> _FakeTensor:
            return _FakeTensor(np.array([1.0, 4.0, 9.0]))

    class _FakeLayer:
        def __init__(self) -> None:
            self.kernel_posterior: _FakeKernelPosterior = _FakeKernelPosterior()

    class _FakeModel:
        def __init__(self) -> None:
            self.layers: List[_FakeLayer] = [_FakeLayer()]
            self.fit_calls: int = 0

        def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            verbose: bool,
            batch_size: int,
            shuffle: bool,
        ) -> None:
            del x, y, epochs, verbose, batch_size, shuffle
            self.fit_calls += 1

    fake_model: _FakeModel = _FakeModel()
    monkeypatch.setattr(target, "get_ltie_model", lambda **kwargs: fake_model)

    sources: np.ndarray = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]
    )
    received_noisy: np.ndarray = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    actual_mean, actual_std, actual_batch_size, actual_repeats = (
        target.fit_ltie_posterior_statistics(
            sources=sources,
            received_noisy=received_noisy,
            batch_size_base=4,
            epochs=2,
            initial_learning_rate=0.001,
            target_learning_rate=0.001,
            alpha=1.0,
            observation_noise_std=0.5,
        )
    )

    np.testing.assert_array_equal(actual_mean, np.array([3.0, 2.0, 1.0]))
    np.testing.assert_array_equal(actual_std, np.array([3.0, 2.0, 1.0]))
    assert actual_batch_size == 4
    assert actual_repeats == 2
    assert fake_model.fit_calls == 1


def test_run_pointwise_ltie_calibration_with_mocked_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        target,
        "load_default_source_signal",
        lambda source_column: np.arange(12, dtype=float),
    )
    monkeypatch.setattr(
        target, "build_default_linear_phase_fir", lambda: np.array([1.0, 0.0, -1.0])
    )

    def fake_fit_ltie_posterior_statistics(
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        del kwargs
        return np.array([1.0, 0.0, -1.0]), np.array([0.1, 0.1, 0.1]), 8, 2

    monkeypatch.setattr(
        target, "fit_ltie_posterior_statistics", fake_fit_ltie_posterior_statistics
    )

    actual: pd.DataFrame = target.run_pointwise_ltie_calibration(
        num_replications=3,
        noise_std=0.5,
        batch_size=8,
        epochs=2,
        seed=7,
        interval_levels=(0.9,),
        num_independent_examples_per_epoch=1,
        source_column=1,
    )

    assert len(actual) == 1
    assert float(actual["noise_std"].iloc[0]) == pytest.approx(0.5)
    assert int(actual["actual_batch_size"].iloc[0]) == 8
    assert int(actual["num_repetitions_per_example"].iloc[0]) == 2


def test_parse_args_parses_custom_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--num-replications",
            "5",
            "--noise-std",
            "0.25",
            "--output-file",
            str(Path("out/custom.csv")),
        ],
    )

    actual = target.parse_args()

    assert actual.num_replications == 5
    assert actual.noise_std == pytest.approx(0.25)
    assert actual.output_file == Path("out/custom.csv")

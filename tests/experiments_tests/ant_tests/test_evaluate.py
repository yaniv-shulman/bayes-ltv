from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

from experiments.ant import evaluate as target


def test_compute_velocity_misfit_returns_expected_shape() -> None:
    signal_window: np.ndarray = np.array([1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125])
    freqs: np.ndarray = np.fft.rfftfreq(10, d=1.0)

    actual: np.ndarray = target.compute_velocity_misfit(
        signal_window=signal_window,
        freqs=freqs,
        distance=1000.0,
        min_velocity=1500.0,
        max_velocity=3500.0,
        n_velocities=5,
    )

    assert actual.shape == (5, len(freqs))
    assert np.all(np.isfinite(actual))


def test_compute_velocity_fit_statistics_constant_velocity() -> None:
    velocity_misfit: np.ndarray = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.4],
            [0.3, 0.2, 0.1],
        ]
    )
    freqs: np.ndarray = np.array([1.0, 2.0, 3.0])

    actual_velocities, actual_mae, actual_std = target.compute_velocity_fit_statistics(
        velocity_misfit=velocity_misfit,
        freqs=freqs,
        min_velocity=100.0,
        max_velocity=300.0,
        n_velocities=3,
        velocity_true_or_func=200.0,
    )

    expected_velocities: np.ndarray = np.array([100.0, 200.0, 300.0])
    expected_error: np.ndarray = expected_velocities - 200.0
    np.testing.assert_allclose(actual_velocities, expected_velocities)
    assert actual_mae == pytest.approx(np.mean(np.abs(expected_error)))
    assert actual_std == pytest.approx(np.std(expected_error))


def test_aggregate_ground_truth_error_returns_selected_freqs_when_requested() -> None:
    velocity_misfit: np.ndarray = np.array(
        [
            [0.1, 0.9, 0.8],
            [0.2, 0.1, 0.4],
            [0.3, 0.2, 0.1],
        ]
    )
    freqs: np.ndarray = np.array([1.0, 2.0, 3.0])

    actual_mean, actual_vector, actual_selected_freqs = (
        target.aggregate_ground_truth_error(
            velocity_misfit=velocity_misfit,
            freqs=freqs,
            min_velocity=100.0,
            max_velocity=300.0,
            n_velocities=3,
            velocity_true_or_func=200.0,
            min_freq=2.0,
            max_freq=3.0,
            return_selected_freqs=True,
        )
    )

    expected_vector: np.ndarray = np.array([0.1, 0.4])
    np.testing.assert_allclose(actual_vector, expected_vector)
    np.testing.assert_allclose(actual_selected_freqs, np.array([2.0, 3.0]))
    assert actual_mean == pytest.approx(np.mean(expected_vector))


def test_aggregate_ground_truth_error_callable_velocity() -> None:
    velocity_misfit: np.ndarray = np.array(
        [
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.7],
            [0.2, 0.9, 0.1],
        ]
    )
    freqs: np.ndarray = np.array([1.0, 2.0, 3.0])

    def velocity_curve(freq: float) -> float:
        if freq < 2.0:
            return 100.0
        if freq < 3.0:
            return 200.0
        return 300.0

    actual_mean, actual_vector = target.aggregate_ground_truth_error(
        velocity_misfit=velocity_misfit,
        freqs=freqs,
        min_velocity=100.0,
        max_velocity=300.0,
        n_velocities=3,
        velocity_true_or_func=velocity_curve,
    )

    expected_vector: np.ndarray = np.array([0.6, 0.2, 0.1])
    np.testing.assert_allclose(actual_vector, expected_vector)
    assert actual_mean == pytest.approx(np.mean(expected_vector))


def test_compute_posterior_velocity_misfit_stats_shapes() -> None:
    freqs: np.ndarray = np.fft.rfftfreq(10, d=1.0)
    posterior_ir_samples: np.ndarray = np.array(
        [
            [1.0, -0.3, 0.1, -0.05, 0.02, -0.01],
            [0.8, -0.25, 0.15, -0.06, 0.01, -0.01],
        ]
    )

    (
        actual_misfits,
        actual_mean,
        actual_median,
        actual_lower_ci,
        actual_upper_ci,
        actual_velocity_axis,
    ) = target.compute_posterior_velocity_misfit_stats(
        posterior_ir_samples=posterior_ir_samples,
        freqs=freqs,
        distance=1000.0,
        min_velocity=1500.0,
        max_velocity=3500.0,
        n_velocities=4,
    )

    assert actual_misfits.shape == (2, 4, len(freqs))
    assert actual_mean.shape == (4, len(freqs))
    assert actual_median.shape == (4, len(freqs))
    assert actual_lower_ci.shape == (4, len(freqs))
    assert actual_upper_ci.shape == (4, len(freqs))
    assert actual_velocity_axis.shape == (4,)


def test_pairs_to_xy_empty_pairs_returns_empty_arrays() -> None:
    actual_x, actual_y = target.pairs_to_xy(filtered_pairs=[], swap=False)

    assert actual_x.size == 0
    assert actual_y.size == 0


def test_pairs_to_xy_swap_controls_assignment() -> None:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.array([1.0, 2.0]), np.array([10.0, 20.0])),
        (np.array([3.0, 4.0]), np.array([30.0, 40.0])),
    ]

    actual_x_no_swap, actual_y_no_swap = target.pairs_to_xy(
        filtered_pairs=pairs, swap=False
    )
    actual_x_swap, actual_y_swap = target.pairs_to_xy(filtered_pairs=pairs, swap=True)

    np.testing.assert_array_equal(actual_x_no_swap, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(
        actual_y_no_swap, np.array([[10.0, 20.0], [30.0, 40.0]])
    )
    np.testing.assert_array_equal(actual_x_swap, np.array([[10.0, 20.0], [30.0, 40.0]]))
    np.testing.assert_array_equal(actual_y_swap, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_run_all_tests_uses_run_test_and_evaluate_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[int] = []

    def fake_run_test(**kwargs):
        selected_pairs: List[Tuple[np.ndarray, np.ndarray]] = kwargs["selected_pairs"]
        calls.append(len(selected_pairs))
        return (
            0.5,
            np.ones(8),
            np.zeros(8),
            np.ones(16),
            np.zeros(16),
            object(),
            object(),
        )

    def fake_evaluate_test(**kwargs) -> Dict[str, float]:
        return {
            "num_pairs": float(kwargs["num"]),
            "ccf_mae": 0.1,
            "ccf_std": 0.2,
            "ccf_target_error": 0.3,
            "mir_mae": 0.4,
            "mir_std": 0.5,
            "mir_target_error": 0.6,
            "best_train_loss": kwargs["best_train_loss"],
        }

    monkeypatch.setattr(target, "run_test", fake_run_test)
    monkeypatch.setattr(target, "evaluate_test", fake_evaluate_test)

    pairs: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.ones(12), np.ones(12)) for _ in range(5)
    ]
    actual_df = target.run_all_tests(
        pairs=pairs,
        test_counts=[1, 3, 5],
        fs=20.0,
        distance_rx=500.0,
        velocity_true_or_func=3000.0,
        batch_size_base=4,
        epochs=1,
        initial_learning_rate=0.001,
        target_learning_rate=0.001,
        alpha=None,
        min_prop_speed=1500.0,
        min_eval_velocity=1000.0,
        max_eval_velocity=4000.0,
        n_velocities=5,
        min_freq=0.1,
        max_freq=8.0,
        num_freq=8,
        one_bit_quantization=False,
        spectral_whitening_mir=False,
    )

    assert calls == [1, 3, 5]
    assert len(actual_df) == 3
    assert list(actual_df["num_pairs"]) == [1.0, 3.0, 5.0]


def test_run_test_does_not_whiten_mir_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.array([1.0, 0.0, -1.0, 0.5]), np.array([0.5, -0.25, 0.125, -0.0625]))
    ]

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

    class _FakeFitResults:
        def __init__(self) -> None:
            self.history: Dict[str, List[float]] = {"loss": [3.0, 1.0]}

    class _FakeModel:
        def __init__(self) -> None:
            self.layers: List[_FakeLayer] = [_FakeLayer()]
            self.fit_kwargs: Dict[str, object] = {}

        def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
            del training
            return np.zeros((x.shape[0], 2, 1), dtype=float)

        def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            verbose: int,
            batch_size: int,
            shuffle: bool,
            callbacks: List[object],
        ) -> _FakeFitResults:
            del x, y, epochs, verbose, batch_size, shuffle
            self.fit_kwargs["callbacks"] = callbacks
            return _FakeFitResults()

    fake_model: _FakeModel = _FakeModel()

    monkeypatch.setattr(
        target,
        "spectral_whiten_pairs",
        lambda pairs: (_ for _ in ()).throw(
            AssertionError("unexpected whitening call")
        ),
    )
    monkeypatch.setattr(
        target,
        "compute_cross_correlation",
        lambda pairs, one_bit_quantization: (np.ones(4), np.zeros(4)),
    )
    monkeypatch.setattr(target, "get_ltie_model", lambda **kwargs: fake_model)

    actual = target.run_test(
        selected_pairs=pairs,
        epochs=1,
        kernel_size=3,
        batch_size_base=2,
        initial_learning_rate=0.001,
        target_learning_rate=0.001,
        alpha=None,
        one_bit_quantization=False,
        spectral_whitening_mir=False,
    )

    assert actual[0] == pytest.approx(1.0)
    np.testing.assert_allclose(actual[1], np.ones(4))
    np.testing.assert_allclose(actual[3], np.array([3.0, 2.0, 1.0]))
    np.testing.assert_allclose(actual[4], np.array([3.0, 2.0, 1.0]))
    assert fake_model.fit_kwargs["callbacks"] == []


def test_run_test_adds_residual_noise_callback_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.array([1.0, 0.0, -1.0, 0.5]), np.array([0.5, -0.25, 0.125, -0.0625]))
    ]

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

    class _FakeFitResults:
        def __init__(self) -> None:
            self.history: Dict[str, List[float]] = {"loss": [2.0, 0.5]}

    class _FakeModel:
        def __init__(self) -> None:
            self.layers: List[_FakeLayer] = [_FakeLayer()]
            self.fit_callbacks: List[object] = []

        def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
            del training
            return np.zeros((x.shape[0], 2, 1), dtype=float)

        def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            verbose: int,
            batch_size: int,
            shuffle: bool,
            callbacks: List[object],
        ) -> _FakeFitResults:
            del x, y, epochs, verbose, batch_size, shuffle
            self.fit_callbacks = callbacks
            return _FakeFitResults()

    class _DummyCallback:
        def __init__(self, **kwargs) -> None:
            self.kwargs: Dict[str, object] = kwargs

    fake_model: _FakeModel = _FakeModel()

    monkeypatch.setattr(
        target,
        "compute_cross_correlation",
        lambda pairs, one_bit_quantization: (np.ones(4), np.zeros(4)),
    )
    monkeypatch.setattr(target, "get_ltie_model", lambda **kwargs: fake_model)
    monkeypatch.setattr(target, "ResidualObservationNoiseStdCallback", _DummyCallback)

    target.run_test(
        selected_pairs=pairs,
        epochs=1,
        kernel_size=3,
        batch_size_base=2,
        initial_learning_rate=0.001,
        target_learning_rate=0.001,
        alpha=None,
        one_bit_quantization=False,
        spectral_whitening_mir=False,
        estimate_observation_noise_std_from_residuals=True,
    )

    assert len(fake_model.fit_callbacks) == 1
    assert isinstance(fake_model.fit_callbacks[0], _DummyCallback)

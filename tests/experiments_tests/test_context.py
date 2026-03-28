import importlib
from pathlib import Path

import pytest

import experiments.context as target


def test_context_uses_irbr_repo_dir_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir: Path = Path("/tmp/bayes-ltv-test-root")
    monkeypatch.setenv("IRBR_REPO_DIR", str(repo_dir))

    actual = importlib.reload(target)

    assert actual.repo_dir == repo_dir
    assert actual.data_dir == repo_dir.joinpath("data")
    assert actual.out_dir == repo_dir.joinpath("out")


def test_context_raises_when_irbr_repo_dir_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IRBR_REPO_DIR", raising=False)

    with pytest.raises(TypeError):
        importlib.reload(target)

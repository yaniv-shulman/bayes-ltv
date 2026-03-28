import os
import sys
from pathlib import Path

import pytest


REPO_DIR: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = REPO_DIR.joinpath("src")
TEST_ARTIFACTS_DIR: Path = REPO_DIR.joinpath("out", "tests")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("IRBR_REPO_DIR", str(REPO_DIR))


@pytest.fixture
def test_artifacts_dir() -> Path:
    TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_ARTIFACTS_DIR

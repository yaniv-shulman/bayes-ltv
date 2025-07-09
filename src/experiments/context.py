import os
from pathlib import Path

repo_dir: Path = Path(os.getenv("IRBR_REPO_DIR"))
data_dir: Path = repo_dir.joinpath("data")
out_dir: Path = repo_dir.joinpath("out")

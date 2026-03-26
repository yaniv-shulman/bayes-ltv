#!/bin/bash

IRBR_REPO_DIR=$(git rev-parse --show-toplevel)
PYTHONPATH="${IRBR_REPO_DIR}/src"

export IRBR_REPO_DIR
export NVIDIA_VISIBLE_DEVICES=all
export PYTHONPATH

poetry env use 3.11
poetry install --no-root
$(eval poetry env activate)

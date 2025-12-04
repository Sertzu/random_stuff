#!/usr/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --system-site-packages .venv

uv sync --project pyproject.toml

uv run test_perf.py
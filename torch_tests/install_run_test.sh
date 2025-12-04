#!/usr/bin/bash

nvidia-smi

uv venv --system-site-packages .venv

uv sync --project pyproject.toml

uv run test_perf.py
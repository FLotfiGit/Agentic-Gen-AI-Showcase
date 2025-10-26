#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install pre-commit
pre-commit install

echo "pre-commit installed. To run all hooks now: pre-commit run --all-files"
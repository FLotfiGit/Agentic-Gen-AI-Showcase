#!/usr/bin/env bash
set -euo pipefail

# Create venv if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip and install deps
pip install --upgrade pip
pip install -r requirements.txt

# Create outputs dir
mkdir -p outputs

echo "Setup complete. Activate with: source .venv/bin/activate"

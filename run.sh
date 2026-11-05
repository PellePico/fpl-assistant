#!/usr/bin/env bash
# Start FPL Assistant on http://localhost:8501
# First run creates a virtual environment and installs the packages.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "First run: setting up (takes a minute)..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

.venv/bin/streamlit run app.py

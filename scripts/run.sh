#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to activate venv and launch Gradio dashboard
VENV_DIR=".venv"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Error: Virtual environment not found at ${VENV_DIR}"
    echo "Please run ./scripts/setup.sh first"
    exit 1
fi

# Use the venv's Python directly instead of trying to activate
# This works reliably in non-interactive scripts
exec ${VENV_DIR}/bin/python3 scripts/gradio_app.py "$@"

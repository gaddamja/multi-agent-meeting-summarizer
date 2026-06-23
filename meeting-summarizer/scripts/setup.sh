#!/usr/bin/env bash
set -euo pipefail

# Create virtualenv and install dependencies (does NOT auto-install PyTorch by default)
VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

echo "Creating venv at ${VENV_DIR}..."
${PYTHON} -m venv ${VENV_DIR}
. ${VENV_DIR}/bin/activate
pip install --upgrade pip setuptools wheel

echo
echo "IMPORTANT: Install PyTorch before installing the rest of the requirements."
echo "Visit https://pytorch.org/get-started/locally/ and choose the right command for your platform."
echo "Example (macOS CPU):"
echo "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo
read -p "Press Enter to continue and install remaining Python packages (or Ctrl-C to abort)..."

pip install -r requirements.txt

echo
echo "Setup complete. Activate the venv with:"
echo "  source ${VENV_DIR}/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

# Complete end-to-end setup for Meeting Summarizer
# Handles: FFmpeg check, PyTorch install, venv creation, dependencies, HF token setup
VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "  Meeting Summarizer - Complete Setup"
echo "================================================"
echo

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Detected OS: ${OS}"
echo

# Step 1: Check FFmpeg
echo "Step 1/5: Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ FFmpeg is already installed${NC}"
else
    echo -e "${YELLOW}⚠ FFmpeg not found. Installing...${NC}"
    if [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${RED}✗ Homebrew not found. Please install from https://brew.sh/${NC}"
            exit 1
        fi
        brew install ffmpeg
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            echo -e "${RED}✗ Please install FFmpeg manually for your Linux distribution${NC}"
            exit 1
        fi
    elif [[ "$OS" == "windows" ]]; then
        echo -e "${YELLOW}⚠ Please install FFmpeg manually from https://ffmpeg.org/download.html or use: choco install ffmpeg${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ FFmpeg installed successfully${NC}"
fi
echo

# Step 2: Create virtual environment
echo "Step 2/5: Creating virtual environment..."
if [[ -d "${VENV_DIR}" ]]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists at ${VENV_DIR}${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ${VENV_DIR}
        ${PYTHON} -m venv ${VENV_DIR}
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo -e "${YELLOW}⚠ Using existing virtual environment${NC}"
    fi
else
    ${PYTHON} -m venv ${VENV_DIR}
    echo -e "${GREEN}✓ Virtual environment created at ${VENV_DIR}${NC}"
fi
. ${VENV_DIR}/bin/activate
echo

# Step 3: Install PyTorch
echo "Step 3/5: Installing PyTorch..."
if python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch is already installed${NC}"
else
    echo "Installing PyTorch for your platform..."
    if [[ "$OS" == "macos" ]]; then
        # Check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo "Detected Apple Silicon (M1/M2/M3) - installing with MPS support"
            pip install torch torchaudio
        else
            echo "Detected Intel Mac - installing CPU-only version"
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    elif [[ "$OS" == "linux" ]]; then
        # Check for CUDA
        if command -v nvidia-smi &> /dev/null; then
            echo "Detected NVIDIA GPU - installing CUDA version"
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            echo "No NVIDIA GPU detected - installing CPU-only version"
            pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    elif [[ "$OS" == "windows" ]]; then
        echo "Installing PyTorch for Windows (CPU-only by default)"
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    echo -e "${GREEN}✓ PyTorch installed successfully${NC}"
fi
echo

# Step 4: Install dependencies
echo "Step 4/5: Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo

# Step 5: Hugging Face token setup
echo "Step 5/5: Hugging Face Token Setup"
echo "-----------------------------------"
echo "Speaker diarization requires a Hugging Face token."
echo
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo -e "${GREEN}✓ HF_TOKEN environment variable is already set${NC}"
else
    echo "To set up your Hugging Face token:"
    echo "1. Get a token: https://huggingface.co/settings/tokens"
    echo "2. Accept the pyannote license: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo
    read -p "Enter your HF_TOKEN (or press Enter to skip): " HF_TOKEN_INPUT
    if [[ -n "$HF_TOKEN_INPUT" ]]; then
        export HF_TOKEN="$HF_TOKEN_INPUT"
        echo
        echo "Add this to your ~/.bashrc or ~/.zshrc to persist:"
        echo "  echo 'export HF_TOKEN=\"$HF_TOKEN\"' >> ~/.bashrc"
        echo -e "${GREEN}✓ HF_TOKEN set for this session${NC}"
    else
        echo -e "${YELLOW}⚠ Skipped HF_TOKEN setup. You'll need to set it before using speaker diarization.${NC}"
    fi
fi
echo

# Summary
echo "================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================================"
echo
echo "To get started:"
echo "  1. Activate virtual environment:"
echo "       source ${VENV_DIR}/bin/activate"
echo
echo "  2. Launch the Gradio dashboard:"
echo "       python scripts/gradio_app.py"
echo
echo "  Or run individual agents:"
echo "       python scripts/run_transcription.py <audio_file>"
echo "       python scripts/run_summary.py --audio <audio_file>"
echo "       python scripts/run_action_items.py <transcript.json>"
echo
echo "================================================"

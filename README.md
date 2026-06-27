# Meeting Transcription — LangGraph-style Multi-Agent Prototype

This project provides a prototype multi-agent system that converts meeting audio (MP3/WAV) into actionable outputs. It includes a Transcription Agent that uses OpenAI Whisper (local) for speech-to-text and `pyannote.audio` for speaker diarization, producing timestamped, speaker-labelled transcripts.

## Installation

### 1. System Dependencies

Install FFmpeg (required for audio decoding):

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html or use `choco install ffmpeg`

### 2. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or on Windows: .venv\Scripts\activate
```

### 3. Install PyTorch (MUST do this first)

Install PyTorch using official instructions from https://pytorch.org/get-started/locally/

**Example for macOS (CPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Example for macOS with GPU (Apple Silicon):**
```bash
pip install torch torchaudio
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up Hugging Face Token (Required for speaker diarization)

1. Go to https://huggingface.co/settings/tokens and create a new token (read access is sufficient)
2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the model's license
3. Export the token in your shell:

```bash
export HF_TOKEN="hf_your_token_here"
```

**Tip:** Add to your `~/.zshrc` or `~/.bashrc` to persist across sessions:
```bash
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### 1) Transcribe audio only
```bash
python scripts/run_transcription.py /path/to/meeting.mp3 --model small --output out.json
```

### 2) Summarize an existing transcript
```bash
python scripts/run_summary.py --transcript out.json --output summary.json
```

### 3) Transcribe and summarize in one step
```bash
python scripts/run_summary.py --audio /path/to/meeting.mp3 --output summary.json
```

> This command runs the transcription agent first, then executes the summary agent on the generated transcript.

**Options:**
- `--model` — Whisper model size: `tiny`, `base`, `small`, `medium`, `large` (default: `small`)
- `--output` — Output JSON file path (default: `transcript.json`)
- `--hf-token` — Hugging Face token for Mistral-7B inference (or use `HF_TOKEN` environment variable)

## Output Format

The output JSON contains:
```json
{
  "file": "/path/to/audio.mp3",
  "model": "small",
  "transcript": "Full transcription text...",
  "segments": [
    {
      "start": 0.5,
      "end": 2.3,
      "speaker": "SPEAKER_00",
      "text": "Hello, how are you?"
    }
  ]
}
```

## Troubleshooting

**"Cannot access gated repo for pyannote/speaker-diarization"**
- Ensure `HF_TOKEN` is exported in the same shell where you run the script
- Visit https://huggingface.co/pyannote/speaker-diarization-3.1 to accept the model license
- Confirm your token has read access

**"ffmpeg not found"**
- Install FFmpeg (see System Dependencies above)
- Verify installation: `which ffmpeg`

**Slow transcription**
- GPU acceleration is recommended for large models; check PyTorch installation for your GPU
- Use `--model tiny` or `--model base` for faster CPU-only processing

## Notes

- This is a prototype; adapt `src/orchestrator.py` to integrate with LangGraph if needed
- First run of each model size will download the model (~1-3GB depending on size)
- Speaker diarization requires a Hugging Face token and gated model access

## Quick Helpers

Makefile and scripts are provided to simplify setup and runs:

- Create venv and install deps (you still must install PyTorch as appropriate):

```bash
./scripts/setup.sh
# then follow the printed instructions to install PyTorch, then run:
source .venv/bin/activate
```

- Use Makefile shortcuts:

```bash
make install-venv   # creates .venv and installs requirements.txt
make fetch-sample   # attempt to download an AMI sample into ami_sample.wav
make transcribe     # run transcription on ami_sample.wav
```

- Build a Docker image (CPU-based):

```bash
docker build -t meeting-transcriber .
# Run inside Docker (example):
docker run --rm -v $(pwd):/data meeting-transcriber /data/sample_audio.mp3 --model tiny --output /data/out.json
```


# Meeting Transcription — LangGraph-style Multi-Agent Prototype

This project provides a prototype multi-agent system that converts meeting audio (MP3/WAV) into actionable outputs. It includes:

- **Transcription Agent** — OpenAI Whisper (local) + pyannote.audio speaker diarization
- **Summary Agent** — Meeting-wide summarization using Qwen2.5-7B-Instruct
- **Action Item Agent** — Identifies tasks, assignees, and deadlines from transcripts
 - **Action History Agent** — Persists action items in SQLite, tracks status, and flags overdue/recurring items

## Prerequisites

Before running setup, ensure you have:

- **Python 3.9 or higher** — Verify with: `python3 --version`
- **Hugging Face account** — Free account required for speaker diarization ([sign up](https://huggingface.co/join))
- **Platform-specific requirements:**
  - **macOS:** Homebrew must be installed ([brew.sh](https://brew.sh/))
  - **Linux:** sudo privileges for system package installation
  - **Windows:** FFmpeg must be installed manually (setup.sh does not handle Windows FFmpeg install)

## Quick Start

Get up and running in 2 steps:

```bash
# Step 1: Run automated setup (installs FFmpeg, PyTorch, dependencies, and sets up HF token)
./scripts/setup.sh

# Step 2: Launch the Gradio dashboard
./scripts/run.sh
```

The setup script automatically:
- Detects your OS (macOS/Linux/Windows)
- Installs FFmpeg if missing (requires Homebrew on macOS, sudo on Linux)
- Creates and configures a virtual environment
- Installs PyTorch with optimal settings for your platform (CPU/GPU/MPS)
- Installs all Python dependencies from requirements.txt
- Guides you through Hugging Face token setup

**Note:** First run will download models (~1-3GB depending on size). Subsequent runs use cached models.

## Architecture

This repository implements a LangGraph-style agentic flow via a lightweight `StateGraph`.
The pipeline is defined as a sequence of state nodes that execute in order and pass context between them.

The core flow is:

1. `transcription`
   - Uses `src.agents.transcription_agent` to transcribe audio with Whisper and optionally apply speaker diarization with pyannote.audio
   - Produces a structured transcript JSON with speaker-attributed segments
2. `summary_generation`
   - Uses `src.agents.summary_agent` to generate executive summaries, key decisions, and discussion topics
3. `action_extraction`
   - Uses `src.agents.action_item_agent` to extract tasks, assignees, deadlines, and priorities
4. `history_tracking`
   - Uses `src.agents.action_history_agent` to persist action items in SQLite and flag overdue/recurring items
5. `topic_continuity`
   - Uses `src.agents.topic_continuity_agent` to index meeting topics in ChromaDB for cross-meeting search
6. `escalation_check`
   - Applies deterministic rules to flag recurring topics and overdue items for attention
7. `final_report`
   - Assembles all outputs into a comprehensive final report

The workflow is built and executed in `src/state_graph.py`. Each `StateNode`
directly invokes its relevant agent or tool, returns a partial state update, and
declares its `next_state` transition. There is no additional orchestrator layer.

A simple representation of the flow:

```
Audio Input
      |
      v
Transcription Agent
      |
      v
Summary Agent
      |
      v
Action Item Agent
      |
      v
Action History Agent
      |
      v
Final Report
```

This design makes the pipeline easy to extend with new graph nodes or alternate agent paths.

## Installation

### Requirements

- **Python:** 3.9 or higher
- **Internet connection:** Required for downloading models and dependencies
- **Hugging Face account:** Required for pyannote speaker diarization (free)
- **Platform-specific:** Homebrew (macOS) or sudo privileges (Linux) for system package installation

> **Note:** The `setup.sh` script automatically handles FFmpeg and PyTorch installation. See Manual Installation section below for alternative setup.

### Manual Installation (Alternative)

If you prefer to install manually or the setup script doesn't work for your environment:

#### 1. System Dependencies

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

#### 2. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or on Windows: .venv\Scripts\activate
```

#### 3. Install PyTorch

Install PyTorch using official instructions from https://pytorch.org/get-started/locally/

**Example for macOS (CPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Example for macOS with GPU (Apple Silicon):**
```bash
pip install torch torchaudio
```

#### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Set Up Hugging Face Token (Required for speaker diarization)

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

### 4) Extract action items from transcript
```bash
python scripts/run_action_items.py out.json --output action_items.json
```

> Extracts tasks, assignees, and deadlines with cross-referenced speaker labels.

### 5) Extract action items with different format
```bash
python scripts/run_action_items.py out.json --output actions.md --format markdown
python scripts/run_action_items.py out.json --output actions.txt --format table
```

### 6) Run complete end-to-end pipeline

The `StateGraph` orchestrates the full workflow: Transcription → Summary → Action Items → History Tracking → Topic Continuity → Escalation → Final Report.

```bash
.venv/bin/python3 -c "
from src.state_graph import run_meeting_workflow
state = run_meeting_workflow({
    'audio_path': 'sample_audio.mp3',
    'transcript_output': 'transcript.json',
    'summary_output': 'summary.json',
    'action_items_output': 'action_items.json',
    'history_report_output': 'history_report.json',
})
print('Pipeline complete!')
print(state['final_report'])
"
```

This produces:
- `transcript.json` — speaker-attributed transcript
- `summary.json` — structured meeting summary
- `action_items.json` — extracted action items
- `history_report.json` — action item health report with overdue/recurring flags

### 7) Launch the Gradio dashboard

```bash
.venv/bin/python3 scripts/gradio_app.py
```

or from the repository root with Make:

```bash
make run-gradio
```

The dashboard provides:

- **Audio upload** or transcript paste input
- **Speaker-attributed transcript** viewer with timestamps
- **Structured summary** panel (executive summary, decisions, topics)
- **Action item Kanban board** with editable status updates (open/in-progress/completed)
- **Action item health report** showing overdue and recurring items
- **Escalation report** highlighting topics and actions requiring attention
- **Topic continuity** display with recurring topics and related historical context
- **Natural language queries** ("Ask Agents" tab) — e.g., "Which action items assigned to Mike are overdue?"
- **Downloadable reports** in Markdown and PDF formats
- **Meeting history browser** to load and review past meetings

## Output Format

### Transcript Output
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

### Action Items Output
The action items extraction produces:
```json
{
  "action_items": [
    {
      "action_item": "Prepare budget proposal for steering committee",
      "assignee": "Sarah",
      "deadline": "end of week",
      "priority": "high",
      "context_quote": "I'll prepare the budget proposal by end of week."
    }
  ],
  "total_count": 1
}
```

See [docs/ACTION_ITEM_AGENT.md](docs/ACTION_ITEM_AGENT.md) for detailed documentation on the Action Item Agent.

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

- This is a lightweight LangGraph-style implementation and can be replaced by the LangGraph package if durable execution or distributed checkpoints are needed
- First run of each model size will download the model (~1-3GB depending on size)
- Speaker diarization requires a Hugging Face token and gated model access
- Sample audio files are included in the `amicorpus/` directory for testing
- For production use, consider model hosting vs local inference tradeoffs and hardened extraction heuristics
- The repository includes a Dockerfile for containerized deployment

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
make transcribe     # run transcription on the included sample_audio.mp3
```

- Build a Docker image (CPU-based):

```bash
docker build -t meeting-transcriber .
# Run inside Docker (example):
docker run --rm -v $(pwd):/data meeting-transcriber /data/sample_audio.mp3 --model tiny --output /data/out.json
```

# Meeting Transcription — LangGraph-style Multi-Agent Prototype

This project provides a prototype multi-agent system that converts meeting audio (MP3/WAV) into actionable outputs. It includes:

- **Transcription Agent** — OpenAI Whisper (local) + pyannote.audio speaker diarization
- **Summary Agent** — Meeting-wide summarization using Mistral-7B
- **Action Item Agent** — Identifies tasks, assignees, and deadlines from transcripts
 - **Action History Agent** — Persists action items in SQLite, tracks status, and flags overdue/recurring items

## Architecture

This repository implements a LangGraph-style agentic flow via a lightweight `StateGraph`.
The pipeline is defined as a sequence of state nodes that execute in order and pass context between them.

The core flow is:

1. `transcription`
   - Uses `src.agents.transcription_agent` to transcribe audio and optionally apply speaker diarization.
   - Produces a structured transcript JSON with `transcript` and `segments`.
2. `summary_generation`
   - Uses `src.agents.summary_agent` to summarize the transcript into a structured JSON summary.
3. `action_extraction`
   - Uses `src.agents.action_item_agent` to extract tasks, assignees, deadlines, priorities, and context quotes.
4. `final_report`
   - Assembles the transcript, summary, and action items into a single final report.

The state graph is built in `src/state_graph.py` and executed by the orchestrator in `src/orchestrator.py`.
Each step is implemented as a `StateNode` with an `action` function and a `next_state` transition.

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
The orchestrator now uses a `StateGraph` pipeline internally:
`Transcription -> Summary Generation -> Action Extraction -> Final Report`.

```bash
.venv/bin/python3 -c "
from src.orchestrator import build_default_orchestrator
orch = build_default_orchestrator()
result = orch.run_full_pipeline(
    audio_path='sample_audio.mp3',
    transcript_output='transcript.json',
    summary_output='summary.json',
    action_items_output='action_items.json'
  history_report_output='history_report.json'
)
print('Pipeline complete!')
print(result['final_report'])
"
```

This runs all three agents in a single StateGraph execution and produces:
- `transcript.json`
- `summary.json`
- `action_items.json`
 - `history_report.json` (if requested)

> Runs transcription, summarization, and action item extraction in sequence using the new StateGraph pipeline.

**Options:**
- `--model` — Whisper model size: `tiny`, `base`, `small`, `medium`, `large` (default: `small`)
- `--output` — Output JSON file path (default: `transcript.json`)
- `--hf-token` — Hugging Face token for Mistral-7B inference (or use `HF_TOKEN` environment variable)

**Action Items Extraction Options:**
- `--format` — Output format: `json`, `markdown`, or `table` (default: `json`)
- `--model` — Hugging Face model for extraction (default: `Qwen/Qwen2.5-7B-Instruct`)
- `--hf-token` — Hugging Face API token
 - `--history-db` — SQLite file path for action item history (default: `action_history.db`)
 - `--history-report-output` — Optional health report JSON output path
 - `--reference-date` — Optional date for overdue computation, format `YYYY-MM-DD`

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
make transcribe     # run transcription on the included sample_audio.mp3
```

- Build a Docker image (CPU-based):

```bash
docker build -t meeting-transcriber .
# Run inside Docker (example):
docker run --rm -v $(pwd):/data meeting-transcriber /data/sample_audio.mp3 --model tiny --output /data/out.json
```


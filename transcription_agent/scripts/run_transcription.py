#!/usr/bin/env python3
"""CLI wrapper to run the transcription pipeline."""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running this script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.orchestrator import build_default_orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run meeting transcription with speaker diarization")
    parser.add_argument("audio", help="Path to audio file (wav/mp3)")
    parser.add_argument("--model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--output", default="transcript.json", help="Output JSON file")
    args = parser.parse_args()
    orch = build_default_orchestrator()
    print("Starting transcription pipeline...")
    orch.run_transcription(args.audio, output_json=args.output, model_name=args.model)
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()

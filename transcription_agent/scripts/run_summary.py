"""CLI for meeting summarization using the Summary Agent."""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running this script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.summary_agent import SummaryAgent
from src.agents.transcription_agent import process_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured meeting summary from audio or transcript JSON.")
    parser.add_argument("--audio", help="Path to meeting audio file.", default=None)
    parser.add_argument("--transcript", help="Path to transcript JSON from transcription agent.", default=None)
    parser.add_argument("--output", help="Output summary JSON file.", default="summary.json")
    parser.add_argument("--model", help="Mistral model to use.", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf-token", help="Hugging Face token; if omitted HF_TOKEN is used.", default=None)
    parser.add_argument("--temp-transcript", help="Temporary transcript path when summarizing from audio.", default="temp_transcript.json")
    args = parser.parse_args()

    if not args.audio and not args.transcript:
        raise SystemExit("Provide either --audio or --transcript.")
    if args.audio and args.transcript:
        raise SystemExit("Use only one of --audio or --transcript.")

    transcript_path = args.transcript
    if args.audio:
        print(f"Transcribing audio {args.audio}...")
        transcript_path = args.temp_transcript
        process_audio(args.audio, output_json=transcript_path)

    print("Summarizing meeting...")
    agent = SummaryAgent(model_name=args.model, hf_token=args.hf_token)
    summary = agent.summarize_from_file(transcript_path, output_json=args.output)
    print(f"Wrote summary to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.audio and Path(args.temp_transcript).exists():
        try:
            Path(args.temp_transcript).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()

"""CLI for meeting summarization using the orchestrator."""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running this script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.orchestrator import build_default_orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured meeting summary from audio or transcript JSON.")
    parser.add_argument("--audio", help="Path to meeting audio file.", default=None)
    parser.add_argument("--transcript", help="Path to transcript JSON from transcription agent.", default=None)
    parser.add_argument("--output", help="Output summary JSON file.", default="summary.json")
    parser.add_argument("--action-items-output", help="Optional action items output path.", default=None)
    parser.add_argument("--model", help="Mistral model to use.", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf-token", help="Hugging Face token; if omitted HF_TOKEN is used.", default=None)
    parser.add_argument("--temp-transcript", help="Temporary transcript path when summarizing from audio.", default="temp_transcript.json")
    args = parser.parse_args()

    if not args.audio and not args.transcript:
        raise SystemExit("Provide either --audio or --transcript.")
    if args.audio and args.transcript:
        raise SystemExit("Use only one of --audio or --transcript.")

    orch = build_default_orchestrator()
    transcript_path = args.transcript

    if args.audio:
        transcript_path = args.temp_transcript
        print(f"Transcribing audio {args.audio} to temporary transcript {transcript_path}...")
        orch.run_transcription(args.audio, output_json=transcript_path)

    print("Summarizing meeting...")
    summary = orch.run_summary(
        transcript_data=json.loads(Path(transcript_path).read_text(encoding="utf-8")),
        model_name=args.model,
        hf_token=args.hf_token,
    )

    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.action_items_output:
        print("Extracting action items from transcript...")
        action_items = orch.run_action_items(
            transcript_data=json.loads(Path(transcript_path).read_text(encoding="utf-8")),
            model_name=args.model,
            hf_token=args.hf_token,
        )
        Path(args.action_items_output).write_text(json.dumps(action_items.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote action items to {args.action_items_output}")

    if args.audio and Path(args.temp_transcript).exists():
        try:
            Path(args.temp_transcript).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()

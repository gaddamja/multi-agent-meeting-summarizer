"""CLI for meeting summarization and optional action-history processing."""
import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running this script directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.action_history_agent import ActionHistoryAgent
from src.agents.action_item_agent import ActionItemAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.transcription_agent import process_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured meeting summary from audio or transcript JSON.")
    parser.add_argument("--audio", help="Path to meeting audio file.", default=None)
    parser.add_argument("--transcript", help="Path to transcript JSON from transcription agent.", default=None)
    parser.add_argument("--output", help="Output summary JSON file.", default="summary.json")
    parser.add_argument("--action-items-output", help="Optional action items output path.", default=None)
    parser.add_argument("--history-db", help="SQLite DB for action item history.", default="action_history.db")
    parser.add_argument("--history-report-output", help="Optional action item health report output path.", default=None)
    parser.add_argument("--reference-date", help="Reference date for overdue detection (YYYY-MM-DD).", default=None)
    parser.add_argument("--model", help="Hugging Face model to use.", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf-token", help="Hugging Face token; if omitted HF_TOKEN is used.", default=None)
    parser.add_argument("--temp-transcript", help="Temporary transcript path when summarizing from audio.", default="temp_transcript.json")
    args = parser.parse_args()

    if not args.audio and not args.transcript:
        raise SystemExit("Provide either --audio or --transcript.")
    if args.audio and args.transcript:
        raise SystemExit("Use only one of --audio or --transcript.")

    transcript_path = args.transcript

    if args.audio:
        transcript_path = args.temp_transcript
        print(f"Transcribing audio {args.audio} to temporary transcript {transcript_path}...")
        process_audio(args.audio, output_json=transcript_path)

    print("Summarizing meeting...")
    transcript_data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    summary = SummaryAgent(model_name=args.model, hf_token=args.hf_token).summarize(
        transcript_text=transcript_data.get("transcript"),
        segments=transcript_data.get("segments"),
    )

    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary to {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    action_items = None
    if args.action_items_output or args.history_report_output:
        print("Extracting action items from transcript...")
        action_items = ActionItemAgent(model_name=args.model, hf_token=args.hf_token).extract(
            transcript_text=transcript_data.get("transcript"),
            segments=transcript_data.get("segments"),
        )
        if args.action_items_output:
            Path(args.action_items_output).write_text(json.dumps(action_items.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote action items to {args.action_items_output}")

    if args.history_report_output:
        if action_items is None:
            action_items = ActionItemAgent(model_name=args.model, hf_token=args.hf_token).extract(
                transcript_text=transcript_data.get("transcript"),
                segments=transcript_data.get("segments"),
        )
        print("Tracking action item history and generating health report...")
        history_agent = ActionHistoryAgent(db_path=args.history_db)
        meeting_source = args.audio or str(transcript_path)
        participants = sorted(
            {
                segment.get("speaker")
                for segment in transcript_data.get("segments", [])
                if isinstance(segment, dict) and segment.get("speaker")
            }
        )
        meeting_id = history_agent.save_meeting(
            summary=summary,
            meeting_source=meeting_source,
            transcript_text=transcript_data.get("transcript"),
            participants=participants,
        )
        history_agent.save_action_items(
            action_items,
            meeting_id=meeting_id,
            meeting_source=meeting_source,
        )
        history_report = history_agent.build_health_report(
            action_items,
            meeting_source=meeting_source,
            reference_date=args.reference_date,
        )
        history_report["meeting_id"] = meeting_id
        Path(args.history_report_output).write_text(json.dumps(history_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote action item health report to {args.history_report_output}")

    if args.audio and Path(args.temp_transcript).exists():
        try:
            Path(args.temp_transcript).unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()

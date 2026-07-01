#!/usr/bin/env python3
"""Script to extract action items from a meeting transcript."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to Python path to handle imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.orchestrator import build_default_orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Extract action items from meeting transcript"
    )
    parser.add_argument(
        "transcript_json",
        help="Path to transcript JSON file with segments",
    )
    parser.add_argument(
        "--output",
        default="action_items.json",
        help="Output file for extracted action items (default: action_items.json)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model to use (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face API token (falls back to HF_TOKEN env var)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "table"],
        default="json",
        help="Output format (default: json)",
    )
    
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript_json)
    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)
    if transcript_path.suffix != ".json":
        print(f"Error: Expected JSON file, got: {transcript_path.suffix}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Loading transcript from: {transcript_path}")
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))

        orch = build_default_orchestrator()
        print(f"Extracting action items with model: {args.model}")
        action_items = orch.run_action_items(
            transcript_data,
            model_name=args.model,
            hf_token=args.hf_token,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == "json":
            output_path.write_text(json.dumps(action_items.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.format == "markdown":
            output_path.write_text(_format_as_markdown(action_items), encoding="utf-8")
        elif args.format == "table":
            output_path.write_text(_format_as_table(action_items), encoding="utf-8")

        print(f"✓ Action items saved to: {output_path}")
        print(f"\nExtracted {action_items.total_count} action items:")
        for i, item in enumerate(action_items.action_items, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.priority, "⚪")
            print(f"{i}. {priority_emoji} [{item.assignee}] {item.action_item}")
            print(f"   Deadline: {item.deadline}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _format_as_markdown(action_items) -> str:
    """Format action items as markdown."""
    lines = [
        "# Action Items",
        "",
        f"**Total: {action_items.total_count} item(s)**",
        "",
    ]
    
    for item in action_items.action_items:
        lines.append(f"## {item.action_item}")
        lines.append("")
        lines.append(f"- **Assignee:** {item.assignee}")
        lines.append(f"- **Deadline:** {item.deadline}")
        lines.append(f"- **Priority:** {item.priority.upper()}")
        lines.append("")
        lines.append(f"> {item.context_quote}")
        lines.append("")
    
    return "\n".join(lines)


def _format_as_table(action_items) -> str:
    """Format action items as a table."""
    if not action_items.action_items:
        return "No action items extracted."
    
    # Calculate column widths
    col_widths = {
        "task": max(30, max((len(item.action_item) for item in action_items.action_items), default=30)),
        "assignee": max(15, max((len(item.assignee) for item in action_items.action_items), default=15)),
        "deadline": 15,
        "priority": 8,
    }
    
    lines = []
    
    # Header
    header = f"| {'Task':<{col_widths['task']}} | {'Assignee':<{col_widths['assignee']}} | {'Deadline':<{col_widths['deadline']}} | {'Priority':<{col_widths['priority']}} |"
    lines.append(header)
    lines.append("|" + "|".join(["-" * (col_widths[col] + 2) for col in ["task", "assignee", "deadline", "priority"]]) + "|")
    
    # Rows
    for item in action_items.action_items:
        row = f"| {item.action_item:<{col_widths['task']}} | {item.assignee:<{col_widths['assignee']}} | {item.deadline:<{col_widths['deadline']}} | {item.priority:<{col_widths['priority']}} |"
        lines.append(row)
    
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

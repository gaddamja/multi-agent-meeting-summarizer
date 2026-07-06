import datetime
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.agents.action_history_agent import ActionHistoryAgent, VALID_STATUSES
from src.agents.action_item_agent import ActionItemList
from src.orchestrator import MultiAgentOrchestrator

DEFAULT_HISTORY_DB = "action_history.db"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_SUMMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ACTION_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _safe_path(uploaded_file: Any) -> Optional[str]:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    if hasattr(uploaded_file, "name"):
        return uploaded_file.name
    if isinstance(uploaded_file, dict) and "name" in uploaded_file:
        return uploaded_file["name"]
    return None


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _build_transcript_html(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return "<p>No transcript segments are available.</p>"
    rows = []
    for seg in segments:
        start = _format_timestamp(seg.get("start", 0.0))
        end = _format_timestamp(seg.get("end", 0.0))
        speaker = seg.get("speaker", "unknown")
        text = seg.get("text", "").replace("\n", " ")
        rows.append(f"<tr><td>{start}</td><td>{end}</td><td>{speaker}</td><td>{text}</td></tr>")
    table = (
        "<div style='overflow:auto; max-height:420px;'>"
        "<table style='width:100%; border-collapse: collapse;'>"
        "<thead>"
        "<tr style='background:#f4f4f4;'>"
        "<th style='padding:8px; border:1px solid #ddd;'>Start</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>End</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Speaker</th>"
        "<th style='padding:8px; border:1px solid #ddd;'>Transcript</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return table


def _build_summary_html(summary: Dict[str, Any]) -> str:
    if not summary:
        return "<p>No summary is available.</p>"
    sections = []
    executive = summary.get("executive_summary", "")
    sections.append(f"<h3>Executive Summary</h3><p>{executive}</p>")
    decisions = summary.get("key_decisions") or []
    if decisions:
        sections.append("<h3>Key Decisions</h3><ul>" + "".join(f"<li>{dec}</li>" for dec in decisions) + "</ul>")
    topics = summary.get("discussion_topics") or []
    if topics:
        sections.append("<h3>Discussion Topics</h3><ul>" + "".join(
            f"<li><strong>{item.get('topic', item)}</strong>: {item.get('positions', '') if isinstance(item, dict) else ''}</li>"
            for item in topics
        ) + "</ul>")
    unresolved = summary.get("unresolved_items") or []
    if unresolved:
        sections.append("<h3>Unresolved Items</h3><ul>" + "".join(f"<li>{item}</li>" for item in unresolved) + "</ul>")
    return "".join(sections)


def _build_kanban_html(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>No action items have been extracted yet.</p>"
    columns = {"open": [], "in-progress": [], "completed": []}
    for row in rows:
        status = str(row.get("status", "open")).lower()
        if status not in columns:
            status = "open"
        label = f"<strong>{row.get('assignee','TBD')}</strong><br>{row.get('action_item','')}<br><em>Deadline:</em> {row.get('deadline','TBD')}"
        columns[status].append(label)
    html_columns = []
    for status in ["open", "in-progress", "completed"]:
        cards = "".join(f"<div style='padding:10px; margin-bottom:12px; background:#fff; border:1px solid #ddd; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.08);'>{item}</div>" for item in columns[status])
        html_columns.append(
            f"<div style='flex:1; min-width:250px; margin-right:12px;'>"
            f"<h4 style='margin-bottom:10px; text-transform:uppercase;'>{status.replace('-', ' ').title()}</h4>"
            f"{cards or '<p style="color:#555;">No items</p>'}"
            f"</div>"
        )
    return f"<div style='display:flex; gap:12px; flex-wrap:wrap;'>{''.join(html_columns)}</div>"


def _build_health_report_html(report: Dict[str, Any]) -> str:
    if not report:
        return "<p>No health data is available.</p>"
    status_counts = report.get("status_counts", {})
    overdue = report.get("overdue_items", [])
    recurring = report.get("recurring_items", [])
    parts = [
        f"<h4>Meeting date</h4><p>{report.get('meeting_date') or 'N/A'}</p>",
        f"<h4>Participant count</h4><p>{report.get('participant_count')}</p>",
        "<h4>Current status counts</h4>"
        + "<ul>"
        + "".join(f"<li>{k.title()}: {v}</li>" for k, v in status_counts.items())
        + "</ul>",
    ]
    if overdue:
        parts.append("<h4>Overdue action items</h4><ul>" + "".join(f"<li>{item.get('action_item')} ({item.get('deadline')})</li>" for item in overdue) + "</ul>")
    if recurring:
        parts.append("<h4>Recurring items</h4><ul>" + "".join(f"<li>{item.get('current_action_item')} (previous: {item.get('previous_meeting_source')})</li>" for item in recurring) + "</ul>")
    return "".join(parts)


def _build_topic_graph_image(summary: Dict[str, Any], action_items: List[Dict[str, Any]], recurring_topics: List[Dict[str, Any]]) -> str:
    graph = nx.DiGraph()
    graph.add_node("Meeting Summary", type="summary")
    discussion_topics = summary.get("discussion_topics") or []
    topic_labels = []
    for idx, topic in enumerate(discussion_topics, start=1):
        label = topic if isinstance(topic, str) else topic.get("topic", f"Topic {idx}")
        topic_labels.append(label)
        graph.add_node(label, type="topic")
        graph.add_edge("Meeting Summary", label)
    for idx, item in enumerate(action_items, start=1):
        text = item.get("action_item", "Action item")
        node_label = f"AI {idx}: {text[:48]}{'...' if len(text) > 48 else ''}"
        graph.add_node(node_label, type="action")
        graph.add_edge("Meeting Summary", node_label)
        for topic_label in topic_labels:
            if topic_label.lower() in text.lower() or topic_label.lower() in item.get("context_quote", "").lower():
                graph.add_edge(node_label, topic_label)
    for recurring in recurring_topics or []:
        topic = recurring.get("topic")
        if topic and topic not in graph:
            graph.add_node(topic, type="recurring")
            graph.add_edge("Meeting Summary", topic)
    if graph.number_of_nodes() == 0:
        raise ValueError("Topic continuity graph has no nodes.")
    plt.figure(figsize=(10, 6))
    positions = nx.spring_layout(graph, k=0.8, seed=42)
    node_colors = []
    for node, data in graph.nodes(data=True):
        node_type = data.get("type")
        if node_type == "summary":
            node_colors.append("#5b8cff")
        elif node_type == "topic":
            node_colors.append("#1abc9c")
        elif node_type == "action":
            node_colors.append("#f39c12")
        else:
            node_colors.append("#e74c3c")
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=1200, alpha=0.9)
    nx.draw_networkx_edges(graph, positions, arrowstyle="-|>", arrowsize=14, edge_color="#555")
    nx.draw_networkx_labels(graph, positions, font_size=9, font_family="sans-serif")
    plt.axis("off")
    temp_path = Path(tempfile.mkdtemp()) / "topic_graph.png"
    plt.tight_layout()
    plt.savefig(temp_path, dpi=150)
    plt.close()
    return str(temp_path)


def _build_report_markdown(
    meeting_source: str,
    meeting_date: str,
    transcript_data: Dict[str, Any],
    summary: Dict[str, Any],
    action_items: List[Dict[str, Any]],
    health_report: Dict[str, Any],
    recurring_topics: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# Meeting Report",
        f"**Source:** {meeting_source}",
        f"**Date:** {meeting_date}",
        "",
        "## Transcript",
        transcript_data.get("transcript", ""),
        "",
        "## Summary",
        summary.get("executive_summary", ""),
        "",
    ]
    if summary.get("key_decisions"):
        lines.append("### Key Decisions")
        lines.extend(f"- {item}" for item in summary.get("key_decisions", []))
        lines.append("")
    if summary.get("discussion_topics"):
        lines.append("### Discussion Topics")
        for topic in summary.get("discussion_topics", []):
            if isinstance(topic, dict):
                lines.append(f"- **{topic.get('topic')}**: {topic.get('positions')}")
            else:
                lines.append(f"- {topic}")
        lines.append("")
    if summary.get("unresolved_items"):
        lines.append("### Unresolved Items")
        lines.extend(f"- {item}" for item in summary.get("unresolved_items", []))
        lines.append("")
    lines.append("## Action Items")
    if action_items:
        for idx, item in enumerate(action_items, start=1):
            lines.extend([
                f"### {idx}. {item.get('action_item')}",
                f"- Assignee: {item.get('assignee')}",
                f"- Deadline: {item.get('deadline')}",
                f"- Priority: {item.get('priority')}",
                f"- Context: {item.get('context_quote')}",
                f"- Status: {item.get('status', 'open')}",
                "",
            ])
    else:
        lines.append("No action items were detected.")
    lines.append("")
    lines.append("## Health Report")
    lines.extend([
        f"- Participant count: {health_report.get('participant_count', 0)}",
        f"- Total action items: {health_report.get('total_action_items', 0)}",
        f"- Reference date: {health_report.get('reference_date', '')}",
    ])
    if health_report.get("overdue_items"):
        lines.append("### Overdue Items")
        for item in health_report.get("overdue_items", []):
            lines.append(f"- {item.get('action_item')} ({item.get('deadline')})")
    if recurring_topics:
        lines.append("### Recurring Topics")
        for topic in recurring_topics:
            lines.append(f"- {topic.get('topic')} ({topic.get('count')} meetings)")
    return "\n".join(lines)


def _build_pdf_report(markdown_text: str, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: List[Any] = []
    for paragraph in markdown_text.split("\n\n"):
        clean = paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        style = styles["Heading2"] if paragraph.startswith("## ") else styles["BodyText"]
        if paragraph.startswith("## "):
            story.append(Paragraph(paragraph.replace("## ", ""), styles["Heading2"]))
        elif paragraph.startswith("### "):
            story.append(Paragraph(paragraph.replace("### ", ""), styles["Heading3"]))
        elif paragraph.startswith("- "):
            story.append(Paragraph(paragraph.replace("- ", "• "), styles["BodyText"]))
        else:
            story.append(Paragraph(clean, style))
        story.append(Spacer(1, 0.1 * inch))
    doc.build(story)
    return output_path


def _make_action_items_rows(action_items: ActionItemList, saved_ids: List[int]) -> Tuple[List[List[Any]], List[str], List[Dict[str, Any]]]:
    rows: List[List[Any]] = []
    objects: List[Dict[str, Any]] = []
    for idx, item in enumerate(action_items.action_items):
        item_id = saved_ids[idx] if idx < len(saved_ids) else None
        row = [item_id or 0, item.action_item, item.assignee, item.deadline, item.priority, "open"]
        rows.append(row)
        objects.append({
            "id": item_id,
            "action_item": item.action_item,
            "assignee": item.assignee,
            "deadline": item.deadline,
            "priority": item.priority,
            "status": "open",
            "context_quote": item.context_quote,
        })
    return rows, ["id", "action_item", "assignee", "deadline", "priority", "status"], objects


def _persist_action_items(action_items: ActionItemList, meeting_source: str, meeting_date: str) -> Tuple[List[int], Dict[str, Any]]:
    history_agent = ActionHistoryAgent(db_path=DEFAULT_HISTORY_DB)
    ids = history_agent.save_action_items(action_items, meeting_source=meeting_source, meeting_date=meeting_date)
    report = history_agent.build_health_report(action_items, meeting_source=meeting_source)
    return ids, report


def _update_statuses(rows: List[List[Any]]) -> Tuple[str, Dict[str, Any]]:
    history_agent = ActionHistoryAgent(db_path=DEFAULT_HISTORY_DB)
    items = []
    for row in rows:
        try:
            item_id = int(row[0])
        except Exception:
            continue
        status = str(row[5]).strip().lower()
        if status not in VALID_STATUSES:
            status = "open"
        history_agent.update_status(item_id, status)
        items.append({
            "id": item_id,
            "action_item": row[1],
            "assignee": row[2],
            "deadline": row[3],
            "priority": row[4],
            "status": status,
            "context_quote": "",
        })
    kanban_html = _build_kanban_html(items)
    return kanban_html, {"updated_items": len(items)}


def process_meeting_with_error_handling(
    audio_file: Optional[Any],
    transcript_text: str,
    hf_token: str,
    whisper_model: str,
    summary_model: str,
    action_model: str,
    meeting_name: str,
) -> Tuple[str, str, str, List[List[Any]], str, str, str, str]:
    try:
        audio_path = _safe_path(audio_file)
        meeting_source = meeting_name or (Path(audio_path).name if audio_path else "manual_transcript")
        meeting_date = datetime.date.today().isoformat()
        orch = MultiAgentOrchestrator()
        transcript_data: Dict[str, Any]
        if audio_path:
            transcript_data = orch.run_transcription(audio_path, model_name=whisper_model, hf_token=hf_token)
        elif transcript_text and transcript_text.strip():
            transcript_data = {
                "file": meeting_source,
                "model": None,
                "transcript": transcript_text.strip(),
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": "unknown",
                        "text": transcript_text.strip(),
                    }
                ],
            }
        else:
            raise ValueError("Please upload audio or paste transcript text.")
        summary = orch.run_summary(
            transcript_data,
            model_name=summary_model,
            hf_token=hf_token,
        )
        action_items = orch.run_action_items(
            transcript_data,
            model_name=action_model,
            hf_token=hf_token,
        )
        saved_ids, health_report = _persist_action_items(action_items, meeting_source, meeting_date)
        rows, headers, action_objects = _make_action_items_rows(action_items, saved_ids)
        transcript_html = _build_transcript_html(transcript_data.get("segments", []))
        summary_html = _build_summary_html(summary)
        kanban_html = _build_kanban_html(action_objects)
        graph_path = ""
        recurring_topics = health_report.get("recurring_items", [])
        try:
            graph_path = _build_topic_graph_image(summary, [dict(item, status="open") for item in action_objects], recurring_topics)
        except Exception as e:
            graph_path = ""
        report_md = _build_report_markdown(meeting_source, meeting_date, transcript_data, summary, action_objects, health_report, recurring_topics)
        report_md_path = Path(tempfile.mkdtemp()) / "meeting_report.md"
        report_md_path.write_text(report_md, encoding="utf-8")
        report_pdf_path = Path(tempfile.mkdtemp()) / "meeting_report.pdf"
        try:
            _build_pdf_report(report_md, str(report_pdf_path))
        except Exception:
            report_pdf_path = Path("")
        return (
            transcript_html,
            transcript_data.get("transcript", ""),
            summary_html,
            rows,
            kanban_html,
            graph_path or "",
            str(report_md_path),
            str(report_pdf_path) if report_pdf_path.exists() else "",
        )
    except Exception as e:
        error_msg = f"<p style='color:red;'><strong>Error:</strong> {str(e)}</p>"
        return (error_msg, str(e), error_msg, [], "", "", "", "")


def launch_ui() -> None:
    with gr.Blocks(title="Meeting Summarizer Dashboard") as demo:
        # Hidden State components to persist data across page refreshes
        persisted_transcript_html = gr.State(value="")
        persisted_transcript_text = gr.State(value="")
        persisted_summary_html = gr.State(value="")
        persisted_action_rows = gr.State(value=[])
        persisted_kanban_html = gr.State(value="")
        persisted_graph_path = gr.State(value="")
        persisted_report_md = gr.State(value="")
        persisted_report_pdf = gr.State(value="")
        
        gr.Markdown("# Meeting Summarizer Dashboard")
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(sources="upload", type="filepath", label="Upload meeting audio")
                transcript_input = gr.Textbox(lines=6, label="Or paste transcript text")
                meeting_name = gr.Textbox(label="Meeting name / source", placeholder="Weekly sync, client call, etc.")
                hf_token = gr.Textbox(label="Hugging Face token (optional)", type="password")
                whisper_model = gr.Dropdown(label="Whisper model", value=DEFAULT_WHISPER_MODEL, choices=["tiny", "base", "small", "medium", "large"])
                summary_model = gr.Textbox(label="Summary model", value=DEFAULT_SUMMARY_MODEL)
                action_model = gr.Textbox(label="Action item model", value=DEFAULT_ACTION_MODEL)
                process_button = gr.Button("Run Meeting Pipeline")
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Transcript"):
                        transcript_view = gr.HTML(label="Speaker-attributed transcript")
                        transcript_text_output = gr.Textbox(lines=12, label="Full transcript text")
                    with gr.TabItem("Summary"):
                        summary_output = gr.HTML(label="Structured meeting summary")
                    with gr.TabItem("Action Items"):
                        action_table = gr.Dataframe(headers=["id", "action_item", "assignee", "deadline", "priority", "status"], label="Action item dashboard", interactive=True)
                        kanban_output = gr.HTML(label="Kanban-style action board")
                        save_status_button = gr.Button("Save status updates")
                        status_update_output = gr.Textbox(label="Status update", visible=True)
                    with gr.TabItem("Topic Continuity"):
                        topic_graph = gr.Image(label="Topic continuity graph")
                    with gr.TabItem("Report"):
                        report_md_file = gr.File(label="Download meeting report (Markdown)")
                        report_pdf_file = gr.File(label="Download meeting report (PDF)")
        
        # Store results and update UI display
        def process_and_persist(*args):
            result = process_meeting(*args)
            return result + result  # Return both for display and for persistence
        
        process_button.click(
            fn=process_and_persist,
            inputs=[audio_input, transcript_input, hf_token, whisper_model, summary_model, action_model, meeting_name],
            outputs=[
                transcript_view, transcript_text_output, summary_output, action_table, kanban_output, topic_graph, report_md_file, report_pdf_file,
                persisted_transcript_html, persisted_transcript_text, persisted_summary_html, persisted_action_rows, persisted_kanban_html, persisted_graph_path, persisted_report_md, persisted_report_pdf,
            ],
        )
        
        # On page load, restore persisted values
        demo.load(
            fn=lambda: (
                gr.update(value=gr.State.value if hasattr(gr, 'State') else ""),
            ),
        )
        
        save_status_button.click(
            fn=_update_statuses,
            inputs=[action_table],
            outputs=[kanban_output, status_update_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_ui()

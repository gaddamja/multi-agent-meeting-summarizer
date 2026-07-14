import datetime
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "meeting_summarizer_matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.agents.action_history_agent import ActionHistoryAgent, VALID_STATUSES
from src.agents.action_item_agent import ActionItemList
from src.agents.nlp_query_agent import NLPQueryAgent, TARGETS
from src.state_graph import run_meeting_workflow

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


def _file_output_value(path: Any) -> Optional[str]:
    if not path:
        return None
    file_path = Path(str(path)).expanduser()
    if not file_path.is_file():
        return None
    return str(file_path)


def _default_ui_state() -> Dict[str, Any]:
    return {
        "transcript_html": "",
        "transcript_text": "",
        "summary_html": "",
        "summary": {},
        "action_rows": [],
        "action_items": [],
        "kanban_html": "",
        "graph_path": None,
        "report_md": None,
        "report_pdf": None,
    }


def _normalize_action_rows(rows: Any) -> List[List[Any]]:
    """Convert Gradio Dataframe values (pandas or list) into serializable rows."""
    if rows is None:
        return []
    if isinstance(rows, list):
        return [list(row) if isinstance(row, (list, tuple)) else [row] for row in rows]
    if isinstance(rows, dict):
        data = rows.get("data", [])
        return _normalize_action_rows(data)
    if hasattr(rows, "values") and hasattr(rows.values, "tolist"):
        return rows.values.tolist()
    if hasattr(rows, "to_numpy"):
        return rows.to_numpy().tolist()
    return []


def _normalize_ui_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _default_ui_state()
    if not isinstance(state, dict):
        return base
    normalized = dict(base)
    normalized.update({k: state.get(k) for k in base.keys() if k in state})
    normalized["action_rows"] = _normalize_action_rows(normalized.get("action_rows"))
    normalized["graph_path"] = _file_output_value(normalized.get("graph_path"))
    normalized["report_md"] = _file_output_value(normalized.get("report_md"))
    normalized["report_pdf"] = _file_output_value(normalized.get("report_pdf"))
    return normalized


def _rows_to_action_objects(rows: Any) -> List[Dict[str, Any]]:
    objects = []
    for row in _normalize_action_rows(rows):
        padded = list(row) + [""] * max(0, 6 - len(row))
        objects.append({
            "id": padded[0],
            "action_item": padded[1],
            "assignee": padded[2],
            "deadline": padded[3],
            "priority": padded[4],
            "status": padded[5],
            "context_quote": "",
        })
    return objects


def _load_latest_action_history_state(db_path: str = DEFAULT_HISTORY_DB) -> Dict[str, Any]:
    state = _default_ui_state()
    if not Path(db_path).exists():
        return state

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        has_meetings_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meetings'"
        ).fetchone()
        if has_meetings_table:
            latest_meeting = conn.execute(
                """
                SELECT meeting_id, meeting_source, meeting_date, title, summary_json
                FROM meetings
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
            if latest_meeting is not None:
                rows = conn.execute(
                    """
                    SELECT id, action_item, assignee, deadline, priority, status
                    FROM action_items
                    WHERE meeting_id = ?
                    ORDER BY id ASC
                    """,
                    (latest_meeting["meeting_id"],),
                ).fetchall()
                summary = {}
                if latest_meeting["summary_json"]:
                    try:
                        summary = json.loads(latest_meeting["summary_json"])
                    except json.JSONDecodeError:
                        summary = {}

                action_rows = [
                    [
                        row["id"],
                        row["action_item"],
                        row["assignee"],
                        row["deadline"] or "",
                        row["priority"] or "",
                        row["status"] or "open",
                    ]
                    for row in rows
                ]
                meeting_label = (
                    latest_meeting["title"]
                    or latest_meeting["meeting_source"]
                    or "last saved meeting"
                )
                meeting_date = latest_meeting["meeting_date"] or "unknown date"
                state["summary"] = summary
                state["summary_html"] = _build_summary_html(summary)
                state["action_rows"] = action_rows
                state["action_items"] = _rows_to_action_objects(action_rows)
                state["kanban_html"] = _build_kanban_html(state["action_items"])
                if not state["summary_html"] or state["summary_html"] == "<p>No summary is available.</p>":
                    state["summary_html"] = (
                        "<p><strong>Restored from SQLite meeting history.</strong></p>"
                        f"<p>Showing saved action items for <em>{meeting_label}</em> ({meeting_date}). "
                        "No summary was stored for this meeting.</p>"
                    )
                return state

        latest = conn.execute(
            """
            SELECT meeting_source, meeting_date
            FROM action_items
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        if latest is None:
            return state

        rows = conn.execute(
            """
            SELECT id, action_item, assignee, deadline, priority, status
            FROM action_items
            WHERE COALESCE(meeting_source, '') = COALESCE(?, '')
              AND COALESCE(meeting_date, '') = COALESCE(?, '')
            ORDER BY id ASC
            """,
            (latest["meeting_source"], latest["meeting_date"]),
        ).fetchall()

    action_rows = [
        [
            row["id"],
            row["action_item"],
            row["assignee"],
            row["deadline"] or "",
            row["priority"] or "",
            row["status"] or "open",
        ]
        for row in rows
    ]
    meeting_label = latest["meeting_source"] or "last saved meeting"
    meeting_date = latest["meeting_date"] or "unknown date"
    state["action_rows"] = action_rows
    state["kanban_html"] = _build_kanban_html(_rows_to_action_objects(action_rows))
    state["summary_html"] = (
        "<p><strong>Restored from SQLite action history.</strong></p>"
        f"<p>Showing saved action items for <em>{meeting_label}</em> ({meeting_date}). "
        "Transcript and summary text are not stored in the current SQLite schema.</p>"
    )
    return state


def _restore_ui_from_state(state: Optional[Dict[str, Any]]) -> Tuple[str, str, str, List[List[Any]], str, Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    restored = _normalize_ui_state(state)
    if not restored["action_rows"]:
        restored = _normalize_ui_state(_load_latest_action_history_state())
    return (
        restored["transcript_html"],
        restored["transcript_text"],
        restored["summary_html"],
        restored["action_rows"],
        restored["kanban_html"],
        restored["graph_path"],
        restored["report_md"],
        restored["report_pdf"],
        restored,
    )


def _create_persisted_ui_state() -> Any:
    default_state = _default_ui_state()
    if hasattr(gr, "BrowserState"):
        return gr.BrowserState(default_value=default_state, storage_key="meeting_summarizer_ui_state")
    return gr.State(value=default_state)


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


def _update_statuses(rows: Any) -> Tuple[str, Dict[str, Any]]:
    history_agent = ActionHistoryAgent(db_path=DEFAULT_HISTORY_DB)
    items = []
    for row in _normalize_action_rows(rows):
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


def _update_statuses_and_persist(rows: Any, state: Optional[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    normalized_rows = _normalize_action_rows(rows)
    kanban_html, update_info = _update_statuses(normalized_rows)
    persisted = _normalize_ui_state(state)
    persisted["action_rows"] = normalized_rows
    persisted["kanban_html"] = kanban_html
    return kanban_html, json.dumps(update_info), persisted


def _persist_table_edits(rows: Any, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    persisted = _normalize_ui_state(state)
    persisted["action_rows"] = _normalize_action_rows(rows)
    return persisted


def process_meeting_with_error_handling(
    audio_file: Optional[Any],
    transcript_text: str,
    hf_token: str,
    whisper_model: str,
    summary_model: str,
    action_model: str,
    meeting_name: str,
) -> Tuple[Any, ...]:
    try:
        audio_path = _safe_path(audio_file)
        meeting_source = meeting_name or (Path(audio_path).name if audio_path else "manual_transcript")
        meeting_date = datetime.date.today().isoformat()
        transcript_data: Optional[Dict[str, Any]] = None
        if not audio_path and transcript_text and transcript_text.strip():
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
            if not audio_path:
                raise ValueError("Please upload audio or paste transcript text.")

        workflow_state = run_meeting_workflow(
            {
                "audio_path": audio_path,
                "transcript_data": transcript_data,
                "meeting_source": meeting_source,
                "meeting_date": meeting_date,
                "transcript_model_name": whisper_model,
                "summary_model_name": summary_model,
                "action_item_model_name": action_model,
                "hf_token": hf_token or None,
                "history_db_path": DEFAULT_HISTORY_DB,
                "history_chroma_dir": "./.chromadb",
            }
        )
        transcript_data = workflow_state["transcript_data"]
        summary = workflow_state["summary"]
        action_items = workflow_state["action_items"]
        saved_ids = workflow_state.get("saved_action_item_ids", [])
        health_report = workflow_state.get("action_history_report", {})
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
            _file_output_value(graph_path),
            _file_output_value(report_md_path),
            _file_output_value(report_pdf_path),
            summary,
            action_objects,
        )
    except Exception as e:
        error_msg = f"<p style='color:red;'><strong>Error:</strong> {str(e)}</p>"
        return (error_msg, str(e), error_msg, [], "", None, None, None, {}, [])


def _chat_with_agents(
    message: str,
    history: Optional[List[Dict[str, Any]]],
    target: str,
    state: Optional[Dict[str, Any]],
    model_name: str,
    hf_token: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer one conversational query and append it to Gradio chat history."""
    chat_history: List[Dict[str, Any]] = []
    for entry in history or []:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            chat_history.append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            chat_history.extend([
                {"role": "user", "content": str(entry[0])},
                {"role": "assistant", "content": str(entry[1])},
            ])
    target_key = next((key for key, label in TARGETS.items() if label == target), target or "auto")
    result = NLPQueryAgent(
        history_db_path=DEFAULT_HISTORY_DB,
        chroma_dir="./.chromadb",
    ).ask(
        question=message,
        target=target_key,
        state=_normalize_ui_state(state),
        model_name=model_name or DEFAULT_SUMMARY_MODEL,
        hf_token=hf_token or None,
    )
    answer = f"**{TARGETS.get(result['agent'], result['agent'])} agent**\n\n{result['answer']}"
    chat_history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    return "", chat_history


def launch_ui() -> None:
    with gr.Blocks(title="Meeting Summarizer Dashboard") as demo:
        persisted_ui_state = _create_persisted_ui_state()
        
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
                    with gr.TabItem("Ask Agents"):
                        gr.Markdown("Ask about the current meeting, historical action items, or related topics.")
                        agent_target = gr.Dropdown(
                            label="Agent",
                            choices=list(TARGETS.values()),
                            value=TARGETS["auto"],
                        )
                        agent_chat = gr.Chatbot(label="Agent conversation")
                        agent_question = gr.Textbox(
                            label="Question",
                            placeholder="Which action items assigned to Mike are overdue?",
                        )
                        ask_button = gr.Button("Ask")
        
        # Store results and update UI display
        def process_and_persist(*args):
            result = process_meeting_with_error_handling(*args)
            state_payload = {
                "transcript_html": result[0],
                "transcript_text": result[1],
                "summary_html": result[2],
                "action_rows": result[3],
                "kanban_html": result[4],
                "graph_path": result[5],
                "report_md": result[6],
                "report_pdf": result[7],
                "summary": result[8],
                "action_items": result[9],
            }
            return result[:8] + (state_payload,)
        
        process_button.click(
            fn=process_and_persist,
            inputs=[audio_input, transcript_input, hf_token, whisper_model, summary_model, action_model, meeting_name],
            outputs=[
                transcript_view, transcript_text_output, summary_output, action_table, kanban_output, topic_graph, report_md_file, report_pdf_file, persisted_ui_state
            ],
        )

        demo.load(
            fn=_restore_ui_from_state,
            inputs=[persisted_ui_state],
            outputs=[
                transcript_view,
                transcript_text_output,
                summary_output,
                action_table,
                kanban_output,
                topic_graph,
                report_md_file,
                report_pdf_file,
                persisted_ui_state,
            ],
        )

        action_table.change(
            fn=_persist_table_edits,
            inputs=[action_table, persisted_ui_state],
            outputs=[persisted_ui_state],
        )

        save_status_button.click(
            fn=_update_statuses_and_persist,
            inputs=[action_table, persisted_ui_state],
            outputs=[kanban_output, status_update_output, persisted_ui_state],
        )

        chat_inputs = [agent_question, agent_chat, agent_target, persisted_ui_state, summary_model, hf_token]
        ask_button.click(
            fn=_chat_with_agents,
            inputs=chat_inputs,
            outputs=[agent_question, agent_chat],
        )
        agent_question.submit(
            fn=_chat_with_agents,
            inputs=chat_inputs,
            outputs=[agent_question, agent_chat],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_ui()

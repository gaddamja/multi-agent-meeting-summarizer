"""StateGraph abstraction for a meeting processing pipeline.

Defines a linear StateGraph:
    Transcription -> Summary Generation -> Action Extraction -> Final Report

This module is intentionally lightweight and works with the existing MultiAgentOrchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


StateAction = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class StateNode:
    name: str
    description: str
    action: StateAction
    next_state: Optional[str] = None


class StateGraph:
    def __init__(self, nodes: List[StateNode], start_state: str, final_state: str):
        self.nodes: Dict[str, StateNode] = {node.name: node for node in nodes}
        self.start_state = start_state
        self.final_state = final_state

        if start_state not in self.nodes:
            raise ValueError(f"Start state '{start_state}' is not defined in the graph")
        if final_state not in self.nodes:
            raise ValueError(f"Final state '{final_state}' is not defined in the graph")

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        context = dict(initial_context)
        current_state = self.start_state

        while current_state is not None:
            node = self.nodes[current_state]
            context = node.action(context)
            if current_state == self.final_state:
                break
            current_state = node.next_state

        return context


def build_state_graph(orchestrator: Any) -> StateGraph:
    """Build the default meeting processing state graph."""

    def run_transcription(context: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = context["audio_path"]
        transcript_output = context.get("transcript_output")
        transcript_data = orchestrator.run_transcription(
            audio_path,
            output_json=transcript_output,
            model_name=context.get("transcript_model_name", "small"),
        )
        context["transcript_data"] = transcript_data
        return context

    def run_summary_generation(context: Dict[str, Any]) -> Dict[str, Any]:
        transcript_data = context["transcript_data"]
        summary = orchestrator.run_summary(
            transcript_data,
            model_name=context.get("summary_model_name", "Qwen/Qwen2.5-7B-Instruct"),
            hf_token=context.get("hf_token"),
        )
        context["summary"] = summary
        return context

    def run_action_extraction(context: Dict[str, Any]) -> Dict[str, Any]:
        transcript_data = context["transcript_data"]
        action_items = orchestrator.run_action_items(
            transcript_data,
            model_name=context.get("action_item_model_name", "Qwen/Qwen2.5-7B-Instruct"),
            hf_token=context.get("hf_token"),
        )
        context["action_items"] = action_items
        return context

    def run_history_tracking(context: Dict[str, Any]) -> Dict[str, Any]:
        action_items = context["action_items"]
        history_report = orchestrator.run_history_tracking(
            action_items,
            meeting_source=context.get("audio_path"),
            meeting_date=context.get("meeting_date"),
            history_db_path=context.get("history_db_path", "action_history.db"),
            reference_date=context.get("reference_date"),
        )
        context["action_history_report"] = history_report
        return context

    def run_topic_continuity(context: Dict[str, Any]) -> Dict[str, Any]:
        # Use summary, action_items and transcript to build topic continuity
        meeting_id = context.get("audio_path") or str(context.get("transcript_output")) or f"meeting-{id(context)}"
        summary = context.get("summary") or {}
        # summary could be a dict from SummaryAgent; flatten to text if needed
        summary_text = summary
        if isinstance(summary, dict):
            summary_text = summary.get("executive_summary") or json.dumps(summary)

        action_items = []
        ai_obj = context.get("action_items")
        if ai_obj is not None:
            # ActionItemList or dict-like
            try:
                action_items = ai_obj.action_items
            except Exception:
                action_items = ai_obj.get("action_items", []) if isinstance(ai_obj, dict) else []

        topics = []
        # simplistic topic extraction: use summary keys or empty
        if isinstance(summary, dict):
            topics = [t.get("topic") for t in summary.get("discussion_topics", []) if isinstance(t, dict) and t.get("topic")]

        topic_context = orchestrator.run_topic_continuity(
            meeting_id=meeting_id,
            meeting_source=context.get("audio_path"),
            summary=summary_text,
            action_items=[ {"action_item": ai.action_item, "assignee": ai.assignee} for ai in (action_items or []) ],
            topics=topics,
            chroma_dir=context.get("history_chroma_dir", "./.chromadb"),
        )
        context["topic_continuity"] = topic_context
        return context

    def run_escalation_check(context: Dict[str, Any]) -> Dict[str, Any]:
        topic_context = context.get("topic_continuity") or {}
        history_report = context.get("action_history_report") or {}
        escalations = orchestrator.run_highlight_and_escalate(topic_context, history_report)
        context["escalations"] = escalations
        return context

    def run_final_report(context: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            "transcript": context.get("transcript_data"),
            "summary": context.get("summary"),
            "action_items": context.get("action_items"),
        }
        context["final_report"] = report
        return context

    nodes = [
        StateNode(
            name="transcription",
            description="Run transcription and diarization",
            action=run_transcription,
            next_state="summary_generation",
        ),
        StateNode(
            name="summary_generation",
            description="Generate meeting summary from transcript",
            action=run_summary_generation,
            next_state="action_extraction",
        ),
        StateNode(
            name="action_extraction",
            description="Extract action items from transcript",
            action=run_action_extraction,
            next_state="history_tracking",
        ),
        StateNode(
            name="history_tracking",
            description="Track action items in SQLite and build a health report",
            action=run_history_tracking,
            next_state="topic_continuity",
        ),
        StateNode(
            name="topic_continuity",
            description="Index meeting into ChromaDB and retrieve related historical topics",
            action=run_topic_continuity,
            next_state="escalation_check",
        ),
        StateNode(
            name="escalation_check",
            description="Highlight and escalate recurring/overdue items",
            action=run_escalation_check,
            next_state="final_report",
        ),
        StateNode(
            name="final_report",
            description="Assemble the final report payload",
            action=run_final_report,
            next_state=None,
        ),
    ]

    return StateGraph(nodes=nodes, start_state="transcription", final_state="final_report")

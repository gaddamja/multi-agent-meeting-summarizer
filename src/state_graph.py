"""LangGraph-style workflow for the meeting-processing pipeline.

Each node receives the shared state, invokes one agent or tool directly, and
returns only its state updates. ``StateGraph`` applies those updates before
moving to the next node.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.action_history_agent import ActionHistoryAgent
from src.agents.action_item_agent import ActionItemAgent
from src.agents.summary_agent import SummaryAgent


DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
State = Dict[str, Any]
StateUpdate = Dict[str, Any]
StateAction = Callable[[State], StateUpdate]


@dataclass(frozen=True)
class StateNode:
    name: str
    description: str
    action: StateAction
    next_state: Optional[str] = None


class StateGraph:
    """Small, explicit state-machine runner with LangGraph-style node updates."""

    def __init__(self, nodes: List[StateNode], start_state: str, final_state: str):
        self.nodes = {node.name: node for node in nodes}
        self.start_state = start_state
        self.final_state = final_state
        if len(self.nodes) != len(nodes):
            raise ValueError("State node names must be unique")
        if start_state not in self.nodes:
            raise ValueError(f"Start state '{start_state}' is not defined in the graph")
        if final_state not in self.nodes:
            raise ValueError(f"Final state '{final_state}' is not defined in the graph")
        for node in nodes:
            if node.next_state is not None and node.next_state not in self.nodes:
                raise ValueError(f"Node '{node.name}' references unknown next state '{node.next_state}'")

    def run(self, initial_state: State) -> State:
        state = dict(initial_state)
        current_name: Optional[str] = self.start_state
        visited = 0
        while current_name is not None:
            visited += 1
            if visited > len(self.nodes):
                raise RuntimeError("StateGraph contains a cycle; cyclic execution is not supported")
            node = self.nodes[current_name]
            updates = node.action(dict(state))
            if not isinstance(updates, dict):
                raise TypeError(f"Node '{node.name}' must return a dictionary of state updates")
            state.update(updates)
            state["current_node"] = node.name
            if current_name == self.final_state:
                break
            current_name = node.next_state
        return state


def transcription_node(state: State) -> StateUpdate:
    """Use supplied transcript data or invoke the transcription tool."""
    if state.get("transcript_data"):
        return {"transcript_data": state["transcript_data"]}
    audio_path = state.get("audio_path")
    if not audio_path:
        raise ValueError("The workflow requires audio_path or transcript_data")
    from src.agents.transcription_agent import process_audio

    transcript = process_audio(
        audio_path,
        output_json=state.get("transcript_output"),
        model_name=state.get("transcript_model_name", "small"),
        hf_token=state.get("hf_token"),
    )
    return {"transcript_data": transcript}


def summary_node(state: State) -> StateUpdate:
    transcript = state["transcript_data"]
    summary = SummaryAgent(
        model_name=state.get("summary_model_name", DEFAULT_TEXT_MODEL),
        hf_token=state.get("hf_token"),
    ).summarize(
        transcript_text=transcript.get("transcript"),
        segments=transcript.get("segments"),
    )
    if state.get("summary_output"):
        Path(state["summary_output"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"summary": summary}


def action_extraction_node(state: State) -> StateUpdate:
    transcript = state["transcript_data"]
    action_items = ActionItemAgent(
        model_name=state.get("action_item_model_name", DEFAULT_TEXT_MODEL),
        hf_token=state.get("hf_token"),
    ).extract(
        transcript_text=transcript.get("transcript"),
        segments=transcript.get("segments"),
    )
    if state.get("action_items_output"):
        Path(state["action_items_output"]).write_text(
            json.dumps(action_items.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"action_items": action_items}


def history_tracking_node(state: State) -> StateUpdate:
    action_items = state["action_items"]
    transcript = state.get("transcript_data") or {}
    summary = state.get("summary") or {}
    meeting_source = state.get("meeting_source") or state.get("audio_path") or "manual_transcript"
    meeting_id = state.get("meeting_id") or str(uuid.uuid4())
    participants = sorted(
        {
            segment.get("speaker")
            for segment in transcript.get("segments", [])
            if isinstance(segment, dict) and segment.get("speaker")
        }
    )
    agent = ActionHistoryAgent(db_path=state.get("history_db_path", "action_history.db"))
    meeting_id = agent.save_meeting(
        summary=summary,
        meeting_id=meeting_id,
        meeting_source=meeting_source,
        meeting_date=state.get("meeting_date"),
        title=state.get("meeting_title") or state.get("meeting_name"),
        transcript_text=transcript.get("transcript"),
        participants=participants,
    )
    saved_ids = agent.save_action_items(
        action_items,
        meeting_id=meeting_id,
        meeting_source=meeting_source,
        meeting_date=state.get("meeting_date"),
    )
    report = agent.build_health_report(
        action_items,
        reference_date=state.get("reference_date"),
        meeting_source=meeting_source,
    )
    report.update({
        "meeting_id": meeting_id,
        "meeting_source": meeting_source,
        "meeting_date": state.get("meeting_date"),
    })
    if state.get("history_report_output"):
        Path(state["history_report_output"]).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "meeting_id": meeting_id,
        "saved_action_item_ids": saved_ids,
        "action_history_report": report,
    }


def topic_continuity_node(state: State) -> StateUpdate:
    """Index and retrieve topic context; degrade gracefully if the optional store fails."""
    try:
        from src.agents.topic_continuity_agent import TopicContinuityAgent

        summary = state.get("summary") or {}
        summary_text = (
            summary.get("executive_summary") or json.dumps(summary, ensure_ascii=False)
            if isinstance(summary, dict)
            else str(summary)
        )
        raw_items = getattr(state.get("action_items"), "action_items", [])
        action_items = [
            {"action_item": item.action_item, "assignee": item.assignee} for item in raw_items
        ]
        discussion_topics = summary.get("discussion_topics", []) if isinstance(summary, dict) else []
        topics = [
            item.get("topic")
            for item in discussion_topics
            if isinstance(item, dict) and item.get("topic")
        ]
        meeting_source = state.get("meeting_source") or state.get("audio_path") or "manual_transcript"
        meeting_id = state.get("meeting_id") or meeting_source
        agent = TopicContinuityAgent(
            persist_directory=state.get("history_chroma_dir", "./.chromadb")
        )
        indexed_ids = agent.index_meeting(
            meeting_id, meeting_source, summary_text, action_items, topics
        )
        return {
            "topic_continuity": {
                "indexed_ids": indexed_ids,
                "related": agent.query_related(summary_text, k=10),
                "recurring_topics": agent.find_recurring_topics(threshold=3),
            }
        }
    except Exception as exc:
        return {
            "topic_continuity": {"indexed_ids": [], "related": [], "recurring_topics": []},
            "topic_continuity_error": str(exc),
        }


def escalation_node(state: State) -> StateUpdate:
    """Apply deterministic escalation rules directly as a workflow tool node."""
    topic_context = state.get("topic_continuity") or {}
    history_report = state.get("action_history_report") or {}
    escalations = [
        {"topic": topic.get("topic"), "reason": "recurring"}
        for topic in topic_context.get("recurring_topics", [])
        if topic.get("count", 0) >= 3
    ]
    for item in history_report.get("overdue_items", []):
        matches = [
            result
            for result in topic_context.get("related", [])
            if item.get("action_item", "") in (result.get("document") or "")
        ]
        if len(matches) >= 3:
            escalations.append({"action_item": item, "reason": "overdue_3plus"})
    return {"escalations": {"escalations": escalations}}


def final_report_node(state: State) -> StateUpdate:
    action_items = state.get("action_items")
    report = {
        "transcript": state.get("transcript_data"),
        "summary": state.get("summary"),
        "action_items": action_items.model_dump() if action_items is not None else None,
        "history_report": state.get("action_history_report"),
        "topic_continuity": state.get("topic_continuity"),
        "escalations": state.get("escalations"),
    }
    return {"final_report": report}


def build_state_graph() -> StateGraph:
    """Build the default workflow; nodes directly invoke their agent or tool."""
    nodes = [
        StateNode("transcription", "Transcribe audio or accept transcript input", transcription_node, "summary_generation"),
        StateNode("summary_generation", "Generate a structured meeting summary", summary_node, "action_extraction"),
        StateNode("action_extraction", "Extract structured action items", action_extraction_node, "history_tracking"),
        StateNode("history_tracking", "Persist and evaluate action-item history", history_tracking_node, "topic_continuity"),
        StateNode("topic_continuity", "Index and retrieve historical topic context", topic_continuity_node, "escalation_check"),
        StateNode("escalation_check", "Apply escalation rules", escalation_node, "final_report"),
        StateNode("final_report", "Assemble the workflow result", final_report_node),
    ]
    return StateGraph(nodes, start_state="transcription", final_state="final_report")


def run_meeting_workflow(initial_state: State) -> State:
    """Primary public entry point for meeting processing."""
    return build_state_graph().run(initial_state)

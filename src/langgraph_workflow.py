"""LangGraph-based workflow for the meeting-processing pipeline.

This module implements the meeting processing workflow using LangGraph,
providing built-in checkpointing, state persistence, and support for
conditional routing.

Usage:
    from src.langgraph_workflow import run_meeting_workflow_langgraph
    state = run_meeting_workflow_langgraph(initial_state)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.action_history_agent import ActionHistoryAgent
from src.agents.action_item_agent import ActionItemAgent
from src.agents.summary_agent import SummaryAgent


# Define the state schema using TypedDict
class AgentState(Dict[str, Any]):
    """State dictionary that flows through the workflow."""
    pass


# Node functions (workflow step implementations)
def _log(message: str) -> None:
    print(f"[langgraph] {message}")


def transcription_node(state: AgentState) -> AgentState:
    """Transcribe audio or use supplied transcript."""
    if state.get("transcript_data"):
        _log("transcription_node: using supplied transcript_data")
        state["transcript_data"] = state["transcript_data"]
        return state
    
    audio_path = state.get("audio_path")
    if not audio_path:
        raise ValueError("The workflow requires audio_path or transcript_data")
    
    skip_diarization = state.get("skip_diarization", False)
    _log(f"transcription_node: processing audio via Whisper, skip_diarization={skip_diarization}")
    
    from src.agents.transcription_agent import process_audio
    
    transcript = process_audio(
        audio_path,
        output_json=state.get("transcript_output"),
        model_name=state.get("transcript_model_name", "small"),
        hf_token=state.get("hf_token"),
        skip_diarization=skip_diarization,
    )
    
    _log(f"transcription_node: completed — {len(transcript.get('segments', []))} segments")
    state["transcript_data"] = transcript
    return state


def summary_node(state: AgentState) -> AgentState:
    """Generate structured meeting summary."""
    transcript = state["transcript_data"]
    _log("summary_node: generating summary")
    
    from src.agents.summary_agent import SummaryAgent
    
    summary = SummaryAgent(
        model_name=state.get("summary_model_name", "Qwen/Qwen2.5-7B-Instruct"),
        hf_token=state.get("hf_token"),
    ).summarize(
        transcript_text=transcript.get("transcript"),
        segments=transcript.get("segments"),
    )
    
    _log("summary_node: summary generated")
    state["summary"] = summary
    
    if state.get("summary_output"):
        Path(state["summary_output"]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    
    return state


def action_extraction_node(state: AgentState) -> AgentState:
    """Extract structured action items."""
    transcript = state["transcript_data"]
    _log("action_extraction_node: extracting action items")
    
    from src.agents.action_item_agent import ActionItemAgent
    
    action_items = ActionItemAgent(
        model_name=state.get("action_item_model_name", "Qwen/Qwen2.5-7B-Instruct"),
        hf_token=state.get("hf_token"),
    ).extract(
        transcript_text=transcript.get("transcript"),
        segments=transcript.get("segments"),
    )
    
    _log(f"action_extraction_node: extracted {action_items.total_count} action items")
    state["action_items"] = action_items
    
    if state.get("action_items_output"):
        Path(state["action_items_output"]).write_text(
            json.dumps(action_items.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    
    return state


def history_tracking_node(state: AgentState) -> AgentState:
    """Persist action items to SQLite and generate health report."""
    _log("history_tracking_node: saving to database")
    
    action_items = state["action_items"]
    transcript = state.get("transcript_data") or {}
    summary = state.get("summary") or {}
    meeting_source = state.get("meeting_source") or state.get("audio_path") or "manual_transcript"
    meeting_id = state.get("meeting_id") or str(uuid.uuid4())
    
    participants = sorted({
        segment.get("speaker")
        for segment in transcript.get("segments", [])
        if isinstance(segment, dict) and segment.get("speaker")
    })
    
    agent = ActionHistoryAgent(db_path=state.get("history_db_path", "action_history.db"))
    meeting_id = agent.save_meeting(
        summary=summary,
        meeting_id=meeting_id,
        meeting_source=meeting_source,
        meeting_date=state.get("meeting_date"),
        title=state.get("meeting_title") or state.get("meeting_name"),
        transcript_text=transcript.get("transcript"),
        transcript_segments=transcript.get("segments"),
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
    
    _log(f"history_tracking_node: saved meeting {meeting_id} with {len(saved_ids)} action items")
    state["meeting_id"] = meeting_id
    state["saved_action_item_ids"] = saved_ids
    state["action_history_report"] = report
    
    return state


def topic_continuity_node(state: AgentState) -> AgentState:
    """Index meeting topics and retrieve related historical context."""
    _log("topic_continuity_node: indexing topics")
    
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
        related = agent.query_related(summary_text, k=10)
        recurring = agent.find_recurring_topics(threshold=3)
        
        state["topic_continuity"] = {
            "indexed_ids": indexed_ids,
            "related": related,
            "recurring_topics": recurring,
        }
        
        _log(f"topic_continuity_node: indexed {len(related)} related, {len(recurring)} recurring")
    except Exception as exc:
        _log(f"topic_continuity_node: failed — {exc}")
        state["topic_continuity"] = {
            "indexed_ids": [],
            "related": [],
            "recurring_topics": [],
        }
        state["topic_continuity_error"] = str(exc)
    
    return state


def escalation_node(state: AgentState) -> AgentState:
    """Apply escalation rules."""
    _log("escalation_node: applying rules")
    
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
    
    _log(f"escalation_node: {len(escalations)} escalations")
    state["escalations"] = {"escalations": escalations}
    
    return state


def final_report_node(state: AgentState) -> AgentState:
    """Assemble final report."""
    _log("final_report_node: assembling report")
    
    action_items = state.get("action_items")
    report = {
        "transcript": state.get("transcript_data"),
        "summary": state.get("summary"),
        "action_items": action_items.model_dump() if action_items is not None else None,
        "history_report": state.get("action_history_report"),
        "topic_continuity": state.get("topic_continuity"),
        "escalations": state.get("escalations"),
    }
    
    state["final_report"] = report
    _log("final_report_node: complete")
    
    return state


def build_langgraph_workflow(checkpointing: bool = True) -> StateGraph:
    """Build the workflow using LangGraph.
    
    Args:
        checkpointing: If True, enable state checkpointing for persistence
    
    Returns:
        Compiled LangGraph workflow
    """
    # Define the workflow with AgentState as the state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("transcription", transcription_node)
    workflow.add_node("summary_generation", summary_node)
    workflow.add_node("action_extraction", action_extraction_node)
    workflow.add_node("history_tracking", history_tracking_node)
    workflow.add_node("topic_continuity", topic_continuity_node)
    workflow.add_node("escalation_check", escalation_node)
    workflow.add_node("final_report", final_report_node)
    
    # Add edges (sequential flow)
    workflow.add_edge("transcription", "summary_generation")
    workflow.add_edge("summary_generation", "action_extraction")
    workflow.add_edge("action_extraction", "history_tracking")
    workflow.add_edge("history_tracking", "topic_continuity")
    workflow.add_edge("topic_continuity", "escalation_check")
    workflow.add_edge("escalation_check", "final_report")
    workflow.add_edge("final_report", END)
    
    # Set entry point
    workflow.set_entry_point("transcription")
    
    # Compile with optional checkpointing
    if checkpointing:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()
    
    return app


def run_meeting_workflow_langgraph(initial_state: AgentState, checkpointing: bool = True) -> AgentState:
    """Run the meeting workflow using LangGraph.
    
    Args:
        initial_state: Initial state dictionary
        checkpointing: Enable state checkpointing
    
    Returns:
        Final state with all results
    """
    _log("Building LangGraph workflow...")
    app = build_langgraph_workflow(checkpointing=checkpointing)
    
    _log("Running workflow...")
    config = {"configurable": {"thread_id": initial_state.get("meeting_id", "default")}}
    result = app.invoke(initial_state, config=config)
    
    _log("Workflow completed successfully")
    return result
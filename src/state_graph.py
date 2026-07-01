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

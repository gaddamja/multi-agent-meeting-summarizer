"""Simple orchestrator for multi-agent transcription pipeline.

This is a minimal orchestrator demonstrating how agents can be composed.
"""
import json
from typing import Any, Dict, List, Optional
from src.agents.summary_agent import SummaryAgent
from src.agents.action_item_agent import ActionItemAgent
from src.agents.action_history_agent import ActionHistoryAgent
from src.state_graph import StateGraph, build_state_graph


def _import_transcription_processor():
    from src.agents.transcription_agent import process_audio
    return process_audio


def _import_topic_continuity_agent():
    from src.agents.topic_continuity_agent import TopicContinuityAgent
    return TopicContinuityAgent


class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {}

    def register(self, name: str, agent: Any):
        self.agents[name] = agent

    def run_transcription(self, audio_path: str, **kwargs) -> Dict:
        process_audio = _import_transcription_processor()
        return process_audio(audio_path, **kwargs)

    def run_summary(self, transcript_data: Dict, **kwargs) -> Dict:
        agent = SummaryAgent(**kwargs)
        return agent.summarize(
            transcript_text=transcript_data.get("transcript"),
            segments=transcript_data.get("segments"),
        )

    def run_action_items(self, transcript_data: Dict, **kwargs):
        """Extract action items from transcript data.
        
        Args:
            transcript_data: Dict with 'transcript' and 'segments' keys
            **kwargs: model_name and hf_token for ActionItemAgent
        
        Returns:
            ActionItemList with extracted action items
        """
        agent = ActionItemAgent(**kwargs)
        return agent.extract(
            transcript_text=transcript_data.get("transcript"),
            segments=transcript_data.get("segments"),
        )

    def run_history_tracking(
        self,
        action_items,
        meeting_source: Optional[str] = None,
        meeting_date: Optional[str] = None,
        history_db_path: str = "action_history.db",
        reference_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store action items in SQLite and generate a health report."""
        agent = ActionHistoryAgent(db_path=history_db_path)
        agent.save_action_items(
            action_items,
            meeting_source=meeting_source,
            meeting_date=meeting_date,
        )
        report = agent.build_health_report(
            action_items,
            reference_date=reference_date,
            meeting_source=meeting_source,
        )
        report["meeting_source"] = meeting_source
        report["meeting_date"] = meeting_date
        return report

    def run_topic_continuity(
        self,
        meeting_id: str,
        meeting_source: str,
        summary: str,
        action_items: List[Dict[str, Any]],
        topics: List[str],
        chroma_dir: str = "./.chromadb",
    ) -> Dict[str, Any]:
        TopicContinuityAgent = _import_topic_continuity_agent()
        agent = TopicContinuityAgent(persist_directory=chroma_dir)
        ids = agent.index_meeting(meeting_id, meeting_source, summary, action_items, topics)
        # Query related topics for the summary and return recurring topics
        related = agent.query_related(summary, k=10)
        recurring = agent.find_recurring_topics(threshold=3)
        return {"indexed_ids": ids, "related": related, "recurring_topics": recurring}

    def run_highlight_and_escalate(self, topic_context: Dict[str, Any], history_report: Dict[str, Any]) -> Dict[str, Any]:
        escalations = []
        # If recurring topics present or overdue items appear repeatedly, escalate
        for t in (topic_context.get("recurring_topics") or []):
            if t.get("count", 0) >= 3:
                escalations.append({"topic": t.get("topic"), "reason": "recurring"})

        overdue = history_report.get("overdue_items", []) if history_report else []
        for item in overdue:
            # simple rule: if item appears in >=3 past meetings, escalate
            # history_report may not contain counts; rely on topic_context search
            matches = [r for r in (topic_context.get("related") or []) if item.get("action_item","") in (r.get("document") or "")]
            if len(matches) >= 3:
                escalations.append({"action_item": item, "reason": "overdue_3plus"})

        return {"escalations": escalations}

    def run_transcription_and_summary(
        self,
        audio_path: str,
        transcript_model_name: str = "small",
        summary_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        hf_token: Optional[str] = None,
        transcript_output: Optional[str] = None,
        summary_output: Optional[str] = None,
    ) -> Dict:
        transcript_data = self.run_transcription(
            audio_path,
            model_name=transcript_model_name,
            output_json=transcript_output,
        )
        summary = self.run_summary(
            transcript_data,
            model_name=summary_model_name,
            hf_token=hf_token,
        )
        if summary_output:
            with open(summary_output, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        return {
            "transcript": transcript_data,
            "summary": summary,
        }

    def run_full_pipeline(
        self,
        audio_path: str,
        transcript_model_name: str = "small",
        summary_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        action_item_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        hf_token: Optional[str] = None,
        transcript_output: Optional[str] = None,
        summary_output: Optional[str] = None,
        action_items_output: Optional[str] = None,
        history_db_path: str = "action_history.db",
        history_report_output: Optional[str] = None,
        reference_date: Optional[str] = None,
    ) -> Dict:
        """Run complete pipeline: transcription, summary, and action item extraction.
        
        Args:
            audio_path: Path to audio file
            transcript_model_name: Whisper model name
            summary_model_name: Model for summarization
            action_item_model_name: Model for action item extraction
            hf_token: Hugging Face API token
            transcript_output: Optional path to save transcript
            summary_output: Optional path to save summary
            action_items_output: Optional path to save action items
        
        Returns:
            Dict with all pipeline outputs
        """
        graph = build_state_graph(self)
        context = graph.run(
            {
                "audio_path": audio_path,
                "transcript_model_name": transcript_model_name,
                "summary_model_name": summary_model_name,
                "action_item_model_name": action_item_model_name,
                "hf_token": hf_token,
                "transcript_output": transcript_output,
                "summary_output": summary_output,
                "action_items_output": action_items_output,
                "history_db_path": history_db_path,
                "history_report_output": history_report_output,
                "reference_date": reference_date,
            }
        )

        if summary_output and context.get("summary") is not None:
            with open(summary_output, "w", encoding="utf-8") as f:
                json.dump(context["summary"], f, ensure_ascii=False, indent=2)

        if action_items_output and context.get("action_items") is not None:
            with open(action_items_output, "w", encoding="utf-8") as f:
                json.dump(context["action_items"].model_dump(), f, ensure_ascii=False, indent=2)

        if history_report_output and context.get("action_history_report") is not None:
            with open(history_report_output, "w", encoding="utf-8") as f:
                json.dump(context["action_history_report"], f, ensure_ascii=False, indent=2)

        return {
            "transcript": context.get("transcript_data"),
            "summary": context.get("summary"),
            "action_items": context.get("action_items").model_dump() if context.get("action_items") is not None else None,
            "history_report": context.get("action_history_report"),
            "final_report": context.get("final_report"),
        }


def build_default_orchestrator() -> MultiAgentOrchestrator:
    orch = MultiAgentOrchestrator()
    orch.register("summary", SummaryAgent())
    orch.register("action_items", ActionItemAgent())
    return orch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--output", default="transcript.json")
    parser.add_argument("--model", default="small")
    args = parser.parse_args()
    orch = build_default_orchestrator()
    res = orch.run_transcription(args.audio, output_json=args.output, model_name=args.model)
    print(f"Completed: wrote {args.output}")

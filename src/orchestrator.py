"""Simple orchestrator for multi-agent transcription pipeline.

This is a minimal orchestrator demonstrating how agents can be composed.
"""
import json
from typing import Any, Dict, Optional
from src.agents.summary_agent import SummaryAgent
from src.agents.action_item_agent import ActionItemAgent
from src.agents.transcription_agent import process_audio
from src.state_graph import StateGraph, build_state_graph


class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {}

    def register(self, name: str, agent: Any):
        self.agents[name] = agent

    def run_transcription(self, audio_path: str, **kwargs) -> Dict:
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
            }
        )

        if summary_output and context.get("summary") is not None:
            with open(summary_output, "w", encoding="utf-8") as f:
                json.dump(context["summary"], f, ensure_ascii=False, indent=2)

        if action_items_output and context.get("action_items") is not None:
            with open(action_items_output, "w", encoding="utf-8") as f:
                json.dump(context["action_items"].model_dump(), f, ensure_ascii=False, indent=2)

        return {
            "transcript": context.get("transcript_data"),
            "summary": context.get("summary"),
            "action_items": context.get("action_items").model_dump() if context.get("action_items") is not None else None,
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

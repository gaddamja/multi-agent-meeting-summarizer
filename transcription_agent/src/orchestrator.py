"""Simple orchestrator for multi-agent transcription pipeline.

This is a minimal orchestrator demonstrating how agents can be composed.
"""
import json
from typing import Any, Dict, Optional
from src.agents.summary_agent import SummaryAgent
from src.agents.transcription_agent import process_audio


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


def build_default_orchestrator() -> MultiAgentOrchestrator:
    orch = MultiAgentOrchestrator()
    orch.register("summary", SummaryAgent())
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

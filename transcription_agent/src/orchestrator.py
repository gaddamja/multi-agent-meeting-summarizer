"""Simple orchestrator for multi-agent transcription pipeline.

This is a minimal orchestrator demonstrating how agents can be composed.
"""
from typing import Any, Dict
from src.agents.transcription_agent import process_audio


class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {}

    def register(self, name: str, agent: Any):
        self.agents[name] = agent

    def run_transcription(self, audio_path: str, **kwargs) -> Dict:
        # For now delegate directly to transcription agent.
        return process_audio(audio_path, **kwargs)


def build_default_orchestrator() -> MultiAgentOrchestrator:
    orch = MultiAgentOrchestrator()
    # Could register multiple agents here for extended processing
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

"""Action Item Extraction Agent: identifies tasks, assignees, and deadlines using Qwen2.5-7B-Instruct.

This agent produces structured action items with:
- action_item (task description)
- assignee (responsible person)
- deadline (when it should be completed)
- priority (high, medium, low)
- context_quote (relevant quote from transcript)

Cross-references speaker labels for accurate assignment.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

HF_INFERENCE_TYPE = None
InferenceApi = None
InferenceClient = None
try:
    from huggingface_hub import InferenceApi
    HF_INFERENCE_TYPE = "InferenceApi"
except Exception:
    try:
        from huggingface_hub import InferenceClient
        HF_INFERENCE_TYPE = "InferenceClient"
    except Exception:
        HF_INFERENCE_TYPE = None


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class ActionItem(BaseModel):
    """Structured representation of a single action item."""
    action_item: str = Field(..., description="Clear, concise description of the task/commitment")
    assignee: str = Field(..., description="Person responsible for completing this action")
    deadline: str = Field(..., description="When the action should be completed (date, relative time, or 'TBD')")
    priority: str = Field(..., description="Priority level: 'high', 'medium', or 'low'")
    context_quote: str = Field(..., description="Relevant quote from the transcript providing context")


class ActionItemList(BaseModel):
    """List of extracted action items from a meeting."""
    action_items: List[ActionItem] = Field(default_factory=list, description="List of action items")
    total_count: int = Field(default=0, description="Total number of action items extracted")


def _require_hf_token(hf_token: Optional[str]) -> str:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token is required for action item extraction. "
            "Set HF_TOKEN or pass hf_token explicitly."
        )
    return token


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _segments_to_transcript(segments: List[Dict[str, Any]]) -> str:
    """Convert transcript segments to formatted text with speaker labels."""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        start = _format_timestamp(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"[{speaker}] {start}: {text}")
    return "\n".join(lines)


def _extract_unique_speakers(segments: List[Dict[str, Any]]) -> List[str]:
    """Extract list of unique speakers from segments."""
    speakers = []
    seen = set()
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        if speaker and speaker not in seen:
            speakers.append(speaker)
            seen.add(speaker)
    return speakers


def _build_action_item_prompt(
    transcript: str,
    speakers: List[str],
    context: str = "",
) -> str:
    """Build a structured prompt for action item extraction."""
    header = [
        "You are a meeting action item extraction assistant.",
        "Identify all commitments, deadlines, and responsible parties from the meeting transcript.",
        "",
        "INSTRUCTIONS:",
        "1. Extract ALL explicit action items, commitments, and follow-ups mentioned in the meeting.",
        "2. For each action item, identify:",
        "   - WHAT: Clear description of the task/commitment",
        "   - WHO: The person responsible (use exact speaker names from the transcript)",
        "   - WHEN: The deadline or timeframe (if mentioned, or 'TBD' if unclear)",
        "   - PRIORITY: Assess as 'high', 'medium', or 'low' based on context",
        "   - CONTEXT: Include a quote from the transcript showing where this was mentioned",
        "",
        "3. Cross-reference speaker names ONLY from the list provided.",
        "4. If speaker name is not clear, use the speaker label from the transcript.",
        "5. For deadlines, look for: dates, 'by X date', 'before', 'after', 'next week', etc.",
        "6. If multiple people are responsible, list the primary assignee.",
        "7. Return valid JSON ONLY with no additional text or markdown.",
        "",
        f"SPEAKERS IN THIS MEETING: {', '.join(speakers)}",
        "",
    ]
    
    if context:
        header.append(context)
        header.append("")
    
    header.append("TRANSCRIPT:")
    header.append(transcript)
    header.append("")
    header.append("Return JSON matching this exact structure:")
    header.append("{")
    header.append('  "action_items": [')
    header.append("    {")
    header.append('      "action_item": "task description",')
    header.append('      "assignee": "person name",')
    header.append('      "deadline": "when",')
    header.append('      "priority": "high/medium/low",')
    header.append('      "context_quote": "relevant quote"')
    header.append("    }")
    header.append("  ],")
    header.append('  "total_count": 0')
    header.append("}")
    header.append("")
    
    return "\n".join(header)


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON from model response."""
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from the action item extraction model.")
    
    text = response_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1:
        text = text[start : end + 1]
    
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from model response: {exc}\nResponse was:\n{text}")
    
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object.")
    
    return parsed


def _extract_model_response_content(response: Any) -> str:
    """Extract text content from various response types."""
    if isinstance(response, str):
        return response
    
    if isinstance(response, dict):
        if response.get("error"):
            raise RuntimeError(response["error"])
        if isinstance(response.get("choices"), list) and response["choices"]:
            first_choice = response["choices"][0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return response.get("generated_text", "") or response.get("output", "") or json.dumps(response)
    
    if hasattr(response, "choices") and getattr(response, "choices"):
        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        if hasattr(first_choice, "text"):
            return str(first_choice.text)
    
    return str(response)


def _generate_text(prompt: str, model_name: str, hf_token: Optional[str]) -> str:
    """Generate text using Hugging Face inference."""
    if HF_INFERENCE_TYPE is None:
        raise RuntimeError(
            "huggingface_hub is required for action item extraction. Install it with `pip install huggingface_hub`."
        )
    
    token = _require_hf_token(hf_token)
    full_url = f"https://huggingface.co/{model_name}"
    print(f"Invoking Hugging Face model: {full_url}")
    
    if HF_INFERENCE_TYPE == "InferenceApi":
        try:
            api = InferenceApi(repo_id=model_name, token=token)
            response = api(
                inputs=prompt,
                parameters={
                    "max_new_tokens": 1024,
                    "temperature": 0.3,
                    "return_full_text": False,
                },
            )
        except Exception as e:
            msg = str(e)
            if "Repository Not Found" in msg or "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Model '{model_name}' not found or inaccessible on Hugging Face.\n"
                    "Verify the model name, ensure your HF token has access, and that the model exists at https://huggingface.co/\n"
                    "You can also pass a different model_name to the ActionItemAgent or set a valid HF_TOKEN environment variable."
                )
            raise
    else:
        client = InferenceClient(model=model_name, token=token)
        try:
            response = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise action item extraction assistant. "
                            "Return only valid JSON matching the requested schema with no additional text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                max_tokens=1500,
                temperature=0.3,
                stream=False,
            )
        except Exception as chat_error:
            msg = str(chat_error)
            if "Repository Not Found" in msg or "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Model '{model_name}' not found or inaccessible on Hugging Face.\n"
                    "Verify the model name, ensure your HF token has access, and that the model exists at https://huggingface.co/\n"
                    "You can also pass a different model_name to the ActionItemAgent or set a valid HF_TOKEN environment variable."
                )
            if "provider" in msg.lower() or "StopIteration" in msg or "supported for task" in msg.lower():
                try:
                    response = client.text_generation(
                        prompt,
                        model=model_name,
                        max_new_tokens=1024,
                        temperature=0.3,
                        return_full_text=False,
                    )
                except Exception as text_error:
                    raise RuntimeError(
                        f"Hugging Face inference setup failed: {text_error}\n"
                        "The selected model/provider combination does not support the requested inference path."
                    )
            else:
                raise RuntimeError(f"Text generation failed: {chat_error}")

    return _extract_model_response_content(response)


def _normalize_action_items(raw: Dict[str, Any]) -> ActionItemList:
    """Normalize and validate extracted action items using Pydantic."""
    items_data = raw.get("action_items", [])
    
    if not isinstance(items_data, list):
        items_data = []
    
    action_items = []
    for item_data in items_data:
        if not isinstance(item_data, dict):
            continue
        
        try:
            action = ActionItem(
                action_item=str(item_data.get("action_item", "")).strip(),
                assignee=str(item_data.get("assignee", "")).strip(),
                deadline=str(item_data.get("deadline", "TBD")).strip(),
                priority=str(item_data.get("priority", "medium")).lower().strip(),
                context_quote=str(item_data.get("context_quote", "")).strip(),
            )
            # Validate priority value
            if action.priority not in ["high", "medium", "low"]:
                action.priority = "medium"
            action_items.append(action)
        except Exception:
            continue
    
    return ActionItemList(
        action_items=action_items,
        total_count=len(action_items),
    )


def extract_action_items(
    transcript_text: Optional[str] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    model_name: str = DEFAULT_MODEL,
    hf_token: Optional[str] = None,
) -> ActionItemList:
    """Extract action items from meeting transcript.
    
    Args:
        transcript_text: Raw transcript text (alternative to segments)
        segments: List of transcript segments with speaker, start, end, text
        model_name: Hugging Face model name to use (default: Qwen/Qwen2.5-7B-Instruct)
        hf_token: Hugging Face API token (falls back to HF_TOKEN env var)
    
    Returns:
        ActionItemList containing extracted action items with speaker assignments.
    
    Raises:
        ValueError: If neither transcript_text nor segments provided
        RuntimeError: If Hugging Face token is missing or model access fails
    """
    if segments is None and transcript_text is None:
        raise ValueError("Either transcript_text or segments must be provided.")
    
    # Build transcript from segments if provided
    if segments and len(segments) > 0:
        transcript = _segments_to_transcript(segments)
        speakers = _extract_unique_speakers(segments)
    else:
        transcript = transcript_text or ""
        speakers = []
    
    if not transcript.strip():
        raise ValueError("Transcript is empty after processing.")
    
    # Build and execute extraction prompt
    prompt = _build_action_item_prompt(transcript, speakers)
    raw_text = _generate_text(prompt, model_name, hf_token)
    raw_items = _parse_json_response(raw_text)
    
    # Normalize and validate using Pydantic
    action_items_list = _normalize_action_items(raw_items)
    
    return action_items_list


def extract_action_items_from_transcript_file(
    transcript_json_path: str,
    model_name: str = DEFAULT_MODEL,
    hf_token: Optional[str] = None,
    output_json: Optional[str] = None,
) -> ActionItemList:
    """Extract action items from a transcript JSON file.
    
    Expects JSON with 'segments' key containing list of {speaker, start, end, text}.
    
    Args:
        transcript_json_path: Path to transcript JSON file
        model_name: Hugging Face model name
        hf_token: Hugging Face API token
        output_json: Optional path to save results as JSON
    
    Returns:
        ActionItemList containing extracted action items.
    """
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    
    segments = transcript_data.get("segments", [])
    action_items_list = extract_action_items(
        segments=segments,
        model_name=model_name,
        hf_token=hf_token,
    )
    
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(action_items_list.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"Action items saved to: {output_json}")
    
    return action_items_list


class ActionItemAgent:
    """High-level agent for action item extraction."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
    ):
        """Initialize the action item extraction agent.
        
        Args:
            model_name: Hugging Face model name (default: Qwen/Qwen2.5-7B-Instruct)
            hf_token: Hugging Face API token
        """
        self.model_name = model_name
        self.hf_token = hf_token
    
    def extract(
        self,
        transcript_text: Optional[str] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> ActionItemList:
        """Extract action items from transcript.
        
        Args:
            transcript_text: Raw transcript text
            segments: List of segments with speaker labels
        
        Returns:
            ActionItemList with extracted action items
        """
        return extract_action_items(
            transcript_text=transcript_text,
            segments=segments,
            model_name=self.model_name,
            hf_token=self.hf_token,
        )
    
    def extract_from_file(
        self,
        transcript_json_path: str,
        output_json: Optional[str] = None,
    ) -> ActionItemList:
        """Extract action items from a transcript JSON file.
        
        Args:
            transcript_json_path: Path to transcript JSON
            output_json: Optional path to save results
        
        Returns:
            ActionItemList with extracted action items.
        """
        return extract_action_items_from_transcript_file(
            transcript_json_path,
            model_name=self.model_name,
            hf_token=self.hf_token,
            output_json=output_json,
        )

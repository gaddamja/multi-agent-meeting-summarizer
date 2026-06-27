"""Summary Agent: structured meeting summarization using Mistral-7B.

This agent produces:
- executive_summary (3-5 sentences)
- key_decisions (bulleted list)
- discussion_topics (topic with participant positions)
- unresolved_items

For meetings longer than 60 minutes, it uses a map-reduce strategy by
summarizing chunks and then combining chunk summaries into a final output.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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


DEFAULT_MISTRAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHUNK_DURATION_SECONDS = 12 * 60
LONG_MEETING_THRESHOLD_SECONDS = 60 * 60


def _require_hf_token(hf_token: Optional[str]) -> str:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token is required for summary generation. "
            "Set HF_TOKEN or pass hf_token explicitly."
        )
    return token


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _segments_to_transcript(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        start = _format_timestamp(seg.get("start", 0.0))
        end = _format_timestamp(seg.get("end", 0.0))
        text = seg.get("text", "").strip()
        if not text:
            continue
        lines.append(f"[{speaker}] {start}-{end}: {text}")
    return "\n".join(lines)


def _chunk_segments(segments: List[Dict[str, Any]], max_duration_seconds: int = CHUNK_DURATION_SECONDS) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_duration = 0.0
    for seg in segments:
        segment_duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        if current and current_duration + segment_duration > max_duration_seconds:
            chunks.append(current)
            current = []
            current_duration = 0.0
        current.append(seg)
        current_duration += segment_duration
    if current:
        chunks.append(current)
    return chunks


def _build_prompt(transcript: str, context: str = "", chunk_index: Optional[int] = None, total_chunks: Optional[int] = None) -> str:
    header = [
        "You are a meeting summarization assistant.",
        "Generate only valid JSON with exactly these keys:",
        "- executive_summary",
        "- key_decisions",
        "- discussion_topics",
        "- unresolved_items",
        "",
        "Rules:",
        "1. executive_summary: 3–5 sentences summarizing the meeting.",
        "2. key_decisions: a JSON array of concise decisions made.",
        "3. discussion_topics: an array of objects with 'topic' and 'positions'.",
        "4. unresolved_items: a JSON array of open questions, action items, or follow-ups.",
        "5. Do NOT add any markdown, comments, or explanation outside JSON.",
        "",
    ]
    if chunk_index is not None and total_chunks is not None:
        header.append(f"This is chunk {chunk_index} of {total_chunks}. Summarize the transcript below for this chunk.")
        header.append("")
    if context:
        header.append(context)
        header.append("")
    header.append("Transcript:")
    header.append(transcript)
    header.append("")
    header.append("Output JSON:")
    return "\n".join(header)


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from the summarization model.")
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


def _normalize_summary(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "executive_summary": str(raw.get("executive_summary", "")).strip(),
        "key_decisions": raw.get("key_decisions", []),
        "discussion_topics": raw.get("discussion_topics", []),
        "unresolved_items": raw.get("unresolved_items", []),
    }
    if isinstance(normalized["key_decisions"], str):
        normalized["key_decisions"] = [line.strip("- ") for line in normalized["key_decisions"].splitlines() if line.strip()]
    if isinstance(normalized["discussion_topics"], str):
        normalized["discussion_topics"] = [line.strip() for line in normalized["discussion_topics"].splitlines() if line.strip()]
    if isinstance(normalized["unresolved_items"], str):
        normalized["unresolved_items"] = [line.strip("- ") for line in normalized["unresolved_items"].splitlines() if line.strip()]
    return normalized


def _extract_model_response_content(response: Any) -> str:
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
    if HF_INFERENCE_TYPE is None:
        raise RuntimeError(
            "huggingface_hub is required for summary generation. Install it with `pip install huggingface_hub`."
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
                    "max_new_tokens": 512,
                    "temperature": 0.2,
                    "return_full_text": False,
                },
            )
        except Exception as e:
            msg = str(e)
            if "Repository Not Found" in msg or "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Model '{model_name}' not found or inaccessible on Hugging Face.\n"
                    "Verify the model name, ensure your HF token has access, and that the model exists at https://huggingface.co/\n"
                    "You can also pass a different model_name to the SummaryAgent or set a valid HF_TOKEN environment variable."
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
                            "You are a precise meeting summarization assistant. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                max_tokens=1024,
                temperature=0.2,
                stream=False,
            )
        except Exception as chat_error:
            msg = str(chat_error)
            if "Repository Not Found" in msg or "404" in msg or "not found" in msg.lower():
                raise RuntimeError(
                    f"Model '{model_name}' not found or inaccessible on Hugging Face.\n"
                    "Verify the model name, ensure your HF token has access, and that the model exists at https://huggingface.co/\n"
                    "You can also pass a different model_name to the SummaryAgent or set a valid HF_TOKEN environment variable."
                )
            if "provider" in msg.lower() or "StopIteration" in msg or "supported for task" in msg.lower():
                try:
                    response = client.text_generation(
                        prompt,
                        model=model_name,
                        max_new_tokens=512,
                        temperature=0.2,
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


def summarize_chunk(
    segments: List[Dict[str, Any]],
    model_name: str = DEFAULT_MISTRAL_MODEL,
    hf_token: Optional[str] = None,
    chunk_index: Optional[int] = None,
    total_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    transcript = _segments_to_transcript(segments)
    prompt = _build_prompt(transcript, chunk_index=chunk_index, total_chunks=total_chunks)
    raw_text = _generate_text(prompt, model_name, hf_token)
    raw_summary = _parse_json_response(raw_text)
    return _normalize_summary(raw_summary)


def summarize_transcript(
    transcript_text: Optional[str] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    model_name: str = DEFAULT_MISTRAL_MODEL,
    hf_token: Optional[str] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    if segments is None and transcript_text is None:
        raise ValueError("Either transcript_text or segments must be provided.")

    if segments is not None and len(segments) > 0:
        meeting_duration = max((seg.get("end", 0.0) for seg in segments), default=0.0)
    else:
        meeting_duration = 0.0

    needs_map_reduce = meeting_duration >= LONG_MEETING_THRESHOLD_SECONDS
    if needs_map_reduce and segments is not None:
        chunks = _chunk_segments(segments)
        partial_summaries: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            partial = summarize_chunk(
                chunk,
                model_name=model_name,
                hf_token=hf_token,
                chunk_index=idx,
                total_chunks=len(chunks),
            )
            partial_summaries.append(partial)

        combine_prompt_parts = [
            "You are combining several chunk-level meeting summaries into one final summary.",
            "Each chunk summary is valid JSON with the keys executive_summary, key_decisions, discussion_topics, and unresolved_items.",
            "Merge these summaries into a single JSON object with the same schema. Remove duplicates, prioritize the most important decisions, and keep the final executive summary concise.",
            "",
        ]
        for idx, partial in enumerate(partial_summaries, start=1):
            combine_prompt_parts.append(f"Chunk {idx} summary:")
            combine_prompt_parts.append(json.dumps(partial, ensure_ascii=False, indent=2))
            combine_prompt_parts.append("")
        combine_prompt_parts.append("Output JSON:")
        combine_prompt = "\n".join(combine_prompt_parts)
        raw_text = _generate_text(combine_prompt, model_name, hf_token)
        final_summary = _normalize_summary(_parse_json_response(raw_text))
    else:
        transcript = transcript_text or _segments_to_transcript(segments or [])
        prompt = _build_prompt(transcript)
        raw_text = _generate_text(prompt, model_name, hf_token)
        final_summary = _normalize_summary(_parse_json_response(raw_text))

    if output_json:
        with open(output_json, "w", encoding="utf-8") as output_file:
            json.dump(final_summary, output_file, ensure_ascii=False, indent=2)
    return final_summary


class SummaryAgent:
    def __init__(self, model_name: str = DEFAULT_MISTRAL_MODEL, hf_token: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token

    def summarize(
        self,
        transcript_text: Optional[str] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        output_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        return summarize_transcript(
            transcript_text=transcript_text,
            segments=segments,
            model_name=self.model_name,
            hf_token=self.hf_token,
            output_json=output_json,
        )

    def summarize_from_file(self, transcript_path: str, output_json: Optional[str] = None) -> Dict[str, Any]:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.summarize(transcript_text=data.get("transcript"), segments=data.get("segments"), output_json=output_json)

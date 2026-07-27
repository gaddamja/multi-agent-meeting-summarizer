"""Natural-language, read-only query routing across meeting agents."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.action_history_agent import ActionHistoryAgent
from src.agents.summary_agent import _generate_text


TARGETS = {
    "auto": "Auto",
    "history": "Action history",
    "topics": "Topic continuity",
    "summary": "Summary",
    "transcript": "Transcript",
    "actions": "Current action items",
}


class NLPQueryAgent:
    """Route questions to stored agent outputs without allowing generated SQL."""

    def __init__(self, history_db_path: str = "action_history.db", chroma_dir: str = "./.chromadb"):
        self.history_db_path = history_db_path
        self.chroma_dir = chroma_dir

    @staticmethod
    def _route(question: str, target: str) -> str:
        if target in TARGETS and target != "auto":
            return target
        text = question.lower()
        if any(word in text for word in ("topic", "discussed before", "recurring theme", "related meeting")):
            return "topics"
        if any(word in text for word in ("overdue", "assigned to", "assignee", "completed", "in progress", "history")):
            return "history"
        if any(word in text for word in ("action item", "task", "deadline", "owner")):
            return "actions"
        if any(word in text for word in ("who said", "transcript", "speaker", "mention")):
            return "transcript"
        return "summary"

    @staticmethod
    def _extract_assignee(question: str) -> Optional[str]:
        patterns = (
            r"(?:assigned to|assignee(?: is)?|owned by|owner is|for)\s+([A-Za-z][A-Za-z .'-]{0,60}?)(?=\s+(?:are|is|that|which|with|due|overdue|open|completed|in progress)\b|[?.!,]|$)",
            r"(?:show|list|find)\s+([A-Za-z][A-Za-z .'-]{0,60}?)['’]s\s+(?:action items|tasks)",
        )
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _format_actions(items: List[Dict[str, Any]], heading: str) -> str:
        if not items:
            return f"{heading}: no matching action items found."
        lines = [f"{heading} ({len(items)}):"]
        for item in items:
            deadline = item.get("deadline") or "no deadline"
            status = item.get("status") or "open"
            lines.append(f"- {item.get('action_item', '')} — {item.get('assignee', 'unassigned')}; due {deadline}; {status}")
        return "\n".join(lines)

    def _ask_history(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        text = question.lower()
        overdue = "overdue" in text or "past due" in text
        status = None
        if not overdue:
            if "completed" in text or "done" in text:
                status = "completed"
            elif "in progress" in text or "in-progress" in text:
                status = "in-progress"
            elif "open" in text:
                status = "open"
        assignee = self._extract_assignee(question)
        items = ActionHistoryAgent(self.history_db_path).query_action_items(
            assignee=assignee,
            status=status,
            overdue=overdue,
        )
        filters = [value for value in (f"assigned to {assignee}" if assignee else None, "overdue" if overdue else status) if value]
        heading = "Action items" + (f" ({', '.join(filters)})" if filters else "")
        return self._format_actions(items, heading), items

    def _ask_topics(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            from src.agents.topic_continuity_agent import TopicContinuityAgent

            matches = TopicContinuityAgent(persist_directory=self.chroma_dir).query_related(question, k=5)
        except Exception as exc:
            return f"Topic continuity is unavailable: {exc}", []
        if not matches:
            return "No related historical topics were found.", []
        lines = ["Most relevant historical context:"]
        for match in matches:
            meta = match.get("metadata") or {}
            source = meta.get("meeting_source") or meta.get("meeting_id") or "unknown meeting"
            lines.append(f"- [{meta.get('type', 'context')}] {match.get('document', '')} — {source}")
        return "\n".join(lines), matches

    @staticmethod
    def _context_for(target: str, state: Dict[str, Any]) -> Any:
        if target == "transcript":
            return state.get("transcript_text") or ""
        if target == "actions":
            return state.get("action_items") or state.get("action_rows") or []
        return state.get("summary") or state.get("summary_html") or ""

    def ask(
        self,
        question: str,
        target: str = "auto",
        state: Optional[Dict[str, Any]] = None,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        hf_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"answer": "Please enter a question.", "agent": "none", "sources": []}

        routed = self._route(question, target)
        if routed == "history":
            answer, sources = self._ask_history(question)
            return {"answer": answer, "agent": routed, "sources": sources}
        if routed == "topics":
            answer, sources = self._ask_topics(question)
            return {"answer": answer, "agent": routed, "sources": sources}

        context = self._context_for(routed, state or {})
        if not context:
            return {"answer": f"No {routed} data is loaded. Run a meeting first.", "agent": routed, "sources": []}
        prompt = (
            "Answer the user's question using only the supplied meeting context. "
            "If the answer is absent, say that it is not available. Be concise and do not invent facts.\n\n"
            f"Agent source: {routed}\nContext:\n{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
            f"Question: {question}\nAnswer:"
        )
        try:
            answer = _generate_text(prompt, model_name, hf_token).strip()
        except Exception as exc:
            return {"answer": f"The {routed} context is available, but the language model could not answer: {exc}", "agent": routed, "sources": [context]}
        return {"answer": answer, "agent": routed, "sources": [context]}

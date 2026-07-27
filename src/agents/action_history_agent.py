"""Historical Tracking Agent for meeting action items.

Stores extracted action items in SQLite and generates an action item health report.
The agent flags overdue items, detects recurring items across meetings, and tracks
status transitions for open/in-progress/completed work.
"""
from __future__ import annotations

import datetime
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.action_item_agent import ActionItem, ActionItemList

DEFAULT_HISTORY_DB = "action_history.db"
VALID_STATUSES = {"open", "in-progress", "completed"}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL UNIQUE,
    meeting_source TEXT,
    meeting_date TEXT,
    title TEXT,
    transcript_text TEXT,
    transcript_segments_json TEXT,
    summary_json TEXT,
    executive_summary TEXT,
    participants_json TEXT,
    topics_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT,
    meeting_source TEXT,
    meeting_date TEXT,
    action_item TEXT NOT NULL,
    assignee TEXT NOT NULL,
    deadline TEXT,
    deadline_parsed TEXT,
    priority TEXT,
    context_quote TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meetings_meeting_id ON meetings(meeting_id);
CREATE INDEX IF NOT EXISTS idx_meetings_created_at ON meetings(created_at);
CREATE INDEX IF NOT EXISTS idx_action_items_assignee ON action_items(assignee);
CREATE INDEX IF NOT EXISTS idx_action_items_status ON action_items(status);
CREATE INDEX IF NOT EXISTS idx_action_items_deadline_parsed ON action_items(deadline_parsed);
CREATE INDEX IF NOT EXISTS idx_action_items_action_item ON action_items(action_item);
"""


class ActionHistoryAgent:
    def __init__(self, db_path: str = DEFAULT_HISTORY_DB):
        self.db_path = db_path
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(CREATE_TABLE_SQL)
            # Migrate action_items table
            action_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(action_items)").fetchall()
            }
            if "meeting_id" not in action_columns:
                conn.execute("ALTER TABLE action_items ADD COLUMN meeting_id TEXT")
            # Migrate meetings table
            meeting_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(meetings)").fetchall()
            }
            if "transcript_segments_json" not in meeting_columns:
                conn.execute("ALTER TABLE meetings ADD COLUMN transcript_segments_json TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_action_items_meeting_id ON action_items(meeting_id)"
            )
            conn.commit()

    @staticmethod
    def _normalize_status(status: str) -> str:
        normalized = str(status or "open").strip().lower()
        if normalized not in VALID_STATUSES:
            normalized = "open"
        return normalized

    @staticmethod
    def _parse_deadline(deadline: Optional[str]) -> Optional[str]:
        if not deadline:
            return None
        text = str(deadline).strip()
        if not text:
            return None

        lower = text.lower()
        today = datetime.date.today()

        if "today" in lower:
            return today.isoformat()
        if "tomorrow" in lower:
            return (today + datetime.timedelta(days=1)).isoformat()
        if "yesterday" in lower:
            return (today - datetime.timedelta(days=1)).isoformat()

        iso_match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
        if iso_match:
            try:
                return datetime.date(
                    int(iso_match.group(1)),
                    int(iso_match.group(2)),
                    int(iso_match.group(3)),
                ).isoformat()
            except ValueError:
                return None

        alt_match = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
        if alt_match:
            try:
                return datetime.date(
                    int(alt_match.group(3)),
                    int(alt_match.group(1)),
                    int(alt_match.group(2)),
                ).isoformat()
            except ValueError:
                return None

        return None

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value if value is not None else [], ensure_ascii=False)

    @staticmethod
    def _extract_topics(summary: Optional[Dict[str, Any]]) -> List[str]:
        topics = []
        if not isinstance(summary, dict):
            return topics
        for item in summary.get("discussion_topics") or []:
            if isinstance(item, dict) and item.get("topic"):
                topics.append(str(item["topic"]))
            elif isinstance(item, str):
                topics.append(item)
        return topics

    def save_meeting(
        self,
        summary: Optional[Dict[str, Any]] = None,
        meeting_id: Optional[str] = None,
        meeting_source: Optional[str] = None,
        meeting_date: Optional[str] = None,
        title: Optional[str] = None,
        transcript_text: Optional[str] = None,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        participants: Optional[List[str]] = None,
    ) -> str:
        print(f"[history] Saving meeting to database: {meeting_source or 'unknown_source'}")
        now = datetime.datetime.utcnow().isoformat()
        stable_meeting_id = meeting_id or str(uuid.uuid4())
        summary_data = summary or {}
        executive_summary = (
            summary_data.get("executive_summary") if isinstance(summary_data, dict) else None
        )
        topics = self._extract_topics(summary_data)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meetings (
                    meeting_id,
                    meeting_source,
                    meeting_date,
                    title,
                    transcript_text,
                    transcript_segments_json,
                    summary_json,
                    executive_summary,
                    participants_json,
                    topics_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    meeting_source = excluded.meeting_source,
                    meeting_date = excluded.meeting_date,
                    title = excluded.title,
                    transcript_text = excluded.transcript_text,
                    transcript_segments_json = excluded.transcript_segments_json,
                    summary_json = excluded.summary_json,
                    executive_summary = excluded.executive_summary,
                    participants_json = excluded.participants_json,
                    topics_json = excluded.topics_json,
                    updated_at = excluded.updated_at
                """,
                (
                    stable_meeting_id,
                    meeting_source,
                    meeting_date,
                    title,
                    transcript_text,
                    self._json_dumps(transcript_segments),
                    self._json_dumps(summary_data),
                    executive_summary,
                    self._json_dumps(participants or []),
                    self._json_dumps(topics),
                    now,
                    now,
                ),
            )
            conn.commit()
        print(f"[history] ✓ Meeting saved: {stable_meeting_id}")
        return stable_meeting_id

    def save_action_items(
        self,
        action_items: ActionItemList,
        meeting_id: Optional[str] = None,
        meeting_source: Optional[str] = None,
        meeting_date: Optional[str] = None,
        status: str = "open",
    ) -> List[int]:
        print(f"[history] Saving {action_items.total_count} action items...")
        now = datetime.datetime.utcnow().isoformat()
        normalized_status = self._normalize_status(status)
        inserted_ids: List[int] = []

        with self._connect() as conn:
            cursor = conn.cursor()
            for item in action_items.action_items:
                deadline_parsed = self._parse_deadline(item.deadline)
                cursor.execute(
                    """
                    INSERT INTO action_items (
                        meeting_id,
                        meeting_source,
                        meeting_date,
                        action_item,
                        assignee,
                        deadline,
                        deadline_parsed,
                        priority,
                        context_quote,
                        status,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meeting_id,
                        meeting_source,
                        meeting_date,
                        item.action_item,
                        item.assignee,
                        item.deadline,
                        deadline_parsed,
                        item.priority,
                        item.context_quote,
                        normalized_status,
                        now,
                        now,
                    ),
                )
                inserted_ids.append(cursor.lastrowid)
            conn.commit()

        print(f"[history] ✓ Saved {len(inserted_ids)} action items to database")
        return inserted_ids

    def update_status(self, item_id: int, status: str) -> None:
        normalized_status = self._normalize_status(status)
        now = datetime.datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE action_items SET status = ?, updated_at = ? WHERE id = ?",
                (normalized_status, now, item_id),
            )
            conn.commit()

    def _query_items(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return results

    def get_previous_items_for_assignee(
        self,
        assignee: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        return self._query_items(
            "SELECT * FROM action_items WHERE assignee = ? ORDER BY created_at DESC LIMIT ?",
            (assignee, limit),
        )

    def get_overdue_items(self, reference_date: Optional[str] = None) -> List[Dict[str, Any]]:
        if reference_date:
            try:
                reference = datetime.date.fromisoformat(reference_date)
            except ValueError:
                reference = datetime.date.today()
        else:
            reference = datetime.date.today()

        return self._query_items(
            "SELECT * FROM action_items WHERE status IN ('open', 'in-progress') AND deadline_parsed IS NOT NULL AND deadline_parsed <= ? ORDER BY deadline_parsed ASC",
            (reference.isoformat(),),
        )

    def query_action_items(
        self,
        assignee: Optional[str] = None,
        status: Optional[str] = None,
        overdue: bool = False,
        reference_date: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query action history with a fixed set of safe, composable filters."""
        clauses: List[str] = []
        params: List[Any] = []

        if assignee:
            clauses.append("LOWER(assignee) = LOWER(?)")
            params.append(assignee.strip())
        if status:
            normalized_status = self._normalize_status(status)
            clauses.append("status = ?")
            params.append(normalized_status)
        if overdue:
            try:
                reference = datetime.date.fromisoformat(reference_date) if reference_date else datetime.date.today()
            except ValueError:
                reference = datetime.date.today()
            clauses.extend([
                "status IN ('open', 'in-progress')",
                "deadline_parsed IS NOT NULL",
                "deadline_parsed <= ?",
            ])
            params.append(reference.isoformat())

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        safe_limit = max(1, min(int(limit), 200))
        params.append(safe_limit)
        return self._query_items(
            f"SELECT * FROM action_items{where} ORDER BY COALESCE(deadline_parsed, '9999-12-31'), created_at DESC LIMIT ?",
            tuple(params),
        )

    def find_recurring_items(
        self,
        current_text: str,
        assignee: Optional[str] = None,
        meeting_source: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        text = current_text.strip().lower()
        pattern = f"%{text}%"
        if meeting_source:
            query = (
                "SELECT * FROM action_items WHERE (action_item LIKE ? OR assignee = ?) "
                "AND meeting_source != ? ORDER BY created_at DESC LIMIT ?"
            )
            params = (pattern, assignee or "", meeting_source, limit)
        else:
            query = (
                "SELECT * FROM action_items WHERE action_item LIKE ? OR assignee = ? "
                "ORDER BY created_at DESC LIMIT ?"
            )
            params = (pattern, assignee or "", limit)
        return self._query_items(query, params)

    def _qualifies_as_recurring(self, raw_current: str, raw_previous: str) -> bool:
        current = raw_current.strip().lower()
        previous = raw_previous.strip().lower()
        if not current or not previous:
            return False
        if current == previous:
            return True
        if current in previous or previous in current:
            return True
        return False

    def build_health_report(
        self,
        current_action_items: ActionItemList,
        reference_date: Optional[str] = None,
        meeting_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        print(f"[history] Building action items health report...")
        participants = sorted({item.assignee for item in current_action_items.action_items})
        overdue = self.get_overdue_items(reference_date=reference_date)

        recurring: List[Dict[str, Any]] = []
        for item in current_action_items.action_items:
            previous = self.find_recurring_items(
                item.action_item,
                assignee=item.assignee,
                meeting_source=meeting_source,
                limit=5,
            )
            for candidate in previous:
                if candidate["action_item"] == item.action_item and candidate["meeting_source"] != None:
                    recurring.append(
                        {
                            "current_action_item": item.action_item,
                            "assignee": item.assignee,
                            "previous_meeting_source": candidate.get("meeting_source"),
                            "previous_status": candidate.get("status"),
                            "previous_deadline": candidate.get("deadline"),
                            "previous_created_at": candidate.get("created_at"),
                        }
                    )
                    break

        status_counts = {
            "open": 0,
            "in-progress": 0,
            "completed": 0,
        }
        status_counts["open"] = current_action_items.total_count

        print(f"[history] ✓ Report: {len(overdue)} overdue, {len(recurring)} recurring items")
        report = {
            "meeting_source": None,
            "meeting_date": None,
            "participant_count": len(participants),
            "participants": participants,
            "total_action_items": current_action_items.total_count,
            "status_counts": status_counts,
            "overdue_items": overdue,
            "recurring_items": recurring,
            "database_path": self.db_path,
            "reference_date": reference_date or datetime.date.today().isoformat(),
        }
        return report

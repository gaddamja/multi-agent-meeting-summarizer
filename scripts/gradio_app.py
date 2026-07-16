"""Rich Gradio dashboard for the multi-agent meeting summarizer pipeline."""
from __future__ import annotations

import datetime
import html
import json
import os
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "meeting_summarizer_matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.agents.action_history_agent import ActionHistoryAgent, VALID_STATUSES
from src.agents.action_item_agent import ActionItemList
from src.agents.nlp_query_agent import NLPQueryAgent, TARGETS
from src.state_graph import run_meeting_workflow

DEFAULT_HISTORY_DB = "action_history.db"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_SUMMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ACTION_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── Design Tokens ──────────────────────────────────────────────────────────

SPEAKER_COLORS = [
    ("#4f46e5", "#eef2ff"), ("#0891b2", "#ecfeff"), ("#d97706", "#fffbeb"),
    ("#dc2626", "#fef2f2"), ("#059669", "#ecfdf5"), ("#7c3aed", "#f5f3ff"),
    ("#db2777", "#fdf2f8"), ("#2563eb", "#eff6ff"), ("#ca8a04", "#fefce8"),
    ("#16a34a", "#f0fdf4"), ("#9333ea", "#faf5ff"), ("#ea580c", "#fff7ed"),
]

PRIORITY_STYLES = {
    "high": ("#dc2626", "#fef2f2", "🔴"),
    "medium": ("#d97706", "#fffbeb", "🟡"),
    "low": ("#16a34a", "#f0fdf4", "🟢"),
}

STATUS_STYLES = {
    "open": ("#2563eb", "#eff6ff", "●"),
    "in-progress": ("#d97706", "#fffbeb", "◐"),
    "completed": ("#16a34a", "#f0fdf4", "●"),
}

# ── Production-Grade CSS ────────────────────────────────────────────────────
# Uses a refined tailwind-inspired palette with proper contrast and spacing.

DASHBOARD_CSS = """
/* =====================================================================
   MEETING SUMMARIZER — Design System
   Refined palette • deliberate whitespace • clear hierarchy
   ===================================================================== */

/* ── Root Variables ─────────────────────────────────────────── */
:root {
  --brand-50:  #eef2ff;
  --brand-100: #e0e7ff;
  --brand-200: #c7d2fe;
  --brand-500: #6366f1;
  --brand-600: #4f46e5;
  --brand-700: #4338ca;
  --brand-800: #3730a3;

  --teal-50:   #ecfeff;
  --teal-500:  #14b8a6;
  --teal-700:  #0f766e;

  --slate-50:  #f8fafc;
  --slate-100: #f1f5f9;
  --slate-200: #e2e8f0;
  --slate-300: #cbd5e1;
  --slate-400: #94a3b8;
  --slate-500: #64748b;
  --slate-600: #475569;
  --slate-700: #334155;
  --slate-800: #1e293b;
  --slate-900: #0f172a;

  --red-50:    #fef2f2;
  --red-500:   #ef4444;
  --green-50:  #f0fdf4;
  --green-500: #22c55e;
  --amber-50:  #fffbeb;
  --amber-500: #f59e0b;

  --font-sans:  'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono:  'SF Mono', 'Cascadia Code', 'Consolas', 'Liberation Mono', monospace;

  --radius-sm:  6px;
  --radius-md:  10px;
  --radius-lg:  16px;
  --radius-xl:  20px;

  --shadow-xs:  0 1px 2px rgba(15,23,42,0.05);
  --shadow-sm:  0 1px 3px rgba(15,23,42,0.08), 0 1px 2px rgba(15,23,42,0.04);
  --shadow-md:  0 4px 6px rgba(15,23,42,0.06), 0 2px 4px rgba(15,23,42,0.04);
  --shadow-lg:  0 10px 25px rgba(15,23,42,0.08), 0 4px 10px rgba(15,23,42,0.04);
  --shadow-xl:  0 20px 50px rgba(15,23,42,0.12), 0 8px 20px rgba(15,23,42,0.06);

  --transition: 200ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Container ───────────────────────────────────────── */
.gradio-container {
  max-width: 1480px !important;
  padding: 20px 28px 40px !important;
  font-family: var(--font-sans);
  color: var(--slate-800);
  background:
    radial-gradient(ellipse 70% 40% at 0% 5%, rgba(99,102,241,0.06), transparent 60%),
    radial-gradient(ellipse 50% 30% at 100% 0%, rgba(20,184,166,0.05), transparent 50%),
    var(--slate-50);
  line-height: 1.6;
}

/* ── Headings ───────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
  letter-spacing: -0.025em;
  color: var(--slate-900);
}

/* ── Hero Section ───────────────────────────────────────────── */
.hero-wrap {
  position: relative;
  padding: 36px 40px;
  margin-bottom: 28px;
  border-radius: var(--radius-xl);
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #134e4a 100%);
  color: white;
  box-shadow: var(--shadow-xl);
  overflow: hidden;
  isolation: isolate;
}
.hero-wrap::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 100% 60% at 20% 30%, rgba(255,255,255,0.06), transparent 60%),
    radial-gradient(ellipse 80% 50% at 80% 70%, rgba(129,140,248,0.08), transparent 50%);
  pointer-events: none;
  z-index: -1;
}
.hero-title {
  margin: 0 0 8px 0;
  font-size: 38px;
  font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #fff 50%, #a5b4fc 80%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  margin: 0;
  max-width: 600px;
  color: rgba(255,255,255,0.7);
  font-size: 15px;
  line-height: 1.65;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 14px;
  padding: 6px 18px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.12);
  color: rgba(255,255,255,0.8);
  font-size: 13px;
  font-weight: 500;
  backdrop-filter: blur(6px);
}

/* ── Card System ────────────────────────────────────────────── */
.glass-card {
  border: 1px solid var(--slate-200);
  border-radius: var(--radius-lg);
  background: white;
  box-shadow: var(--shadow-sm);
  transition: box-shadow var(--transition), transform var(--transition);
}
.glass-card:hover {
  box-shadow: var(--shadow-md);
}
.glass-card.padded {
  padding: 24px;
}

/* Gradio overrides for card groups */
.glass-card > .group {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* ── Label styling for form components ──────────────────────── */
label, .label-text {
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--slate-700) !important;
  letter-spacing: 0.01em !important;
  margin-bottom: 4px !important;
}

/* ── Metric Grid ────────────────────────────────────────────── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.metric-card {
  padding: 16px 18px;
  border-radius: var(--radius-md);
  background: var(--brand-50);
  border: 1px solid var(--brand-200);
  transition: transform var(--transition), box-shadow var(--transition);
}
.metric-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.metric-icon {
  font-size: 22px;
  margin-bottom: 6px;
  line-height: 1;
}
.metric-label {
  color: var(--slate-500);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}
.metric-value {
  color: var(--slate-900);
  font-weight: 700;
  font-size: 20px;
  margin-top: 2px;
  overflow-wrap: anywhere;
}

/* ── Buttons ────────────────────────────────────────────────── */
.gradient-btn {
  background: linear-gradient(135deg, var(--brand-600), var(--brand-800)) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  border-radius: var(--radius-sm) !important;
  transition: all var(--transition) !important;
  box-shadow: 0 4px 14px rgba(79,70,229,0.30) !important;
}
.gradient-btn:hover {
  opacity: 0.92 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(79,70,229,0.35) !important;
}
.gradient-btn:active {
  transform: translateY(0) !important;
  box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}

.secondary-btn {
  border: 1px solid var(--slate-200) !important;
  background: white !important;
  color: var(--slate-700) !important;
  font-weight: 500 !important;
  border-radius: var(--radius-sm) !important;
  transition: all var(--transition) !important;
}
.secondary-btn:hover {
  background: var(--slate-100) !important;
  border-color: var(--slate-300) !important;
  color: var(--slate-900) !important;
}

/* ── Summary Sections ───────────────────────────────────────── */
.summary-wrap { padding: 2px; }
.summary-block {
  padding: 18px 20px;
  margin-bottom: 16px;
  border: 1px solid var(--slate-200);
  border-radius: var(--radius-md);
  background: white;
  transition: box-shadow var(--transition);
}
.summary-block:hover {
  box-shadow: var(--shadow-sm);
}
.summary-block h4 {
  margin: 0 0 12px 0;
  color: var(--brand-600);
  font-size: 15px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}
.summary-block p {
  margin: 0;
  color: var(--slate-600);
  line-height: 1.7;
  font-size: 14px;
}
.check-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 8px 0;
  color: var(--slate-600);
  font-size: 14px;
  border-bottom: 1px solid var(--slate-100);
}
.check-item:last-child {
  border-bottom: none;
}
.check-item::before {
  content: "✓";
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--teal-50);
  color: var(--teal-700);
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
  margin-top: 1px;
}
.topic-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 16px;
  margin: 3px;
  border-radius: 999px;
  background: var(--slate-100);
  border: 1px solid var(--slate-200);
  color: var(--slate-600);
  font-size: 13px;
  font-weight: 500;
  transition: all var(--transition);
}
.topic-item:hover {
  background: var(--brand-50);
  border-color: var(--brand-200);
  color: var(--brand-700);
}

/* ── Transcript ─────────────────────────────────────────────── */
.transcript-wrap {
  overflow: auto;
  max-height: 560px;
  border-radius: var(--radius-md);
  border: 1px solid var(--slate-200);
}
.transcript-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.transcript-table thead {
  position: sticky;
  top: 0;
  z-index: 2;
}
.transcript-table th {
  background: var(--slate-100);
  color: var(--slate-600);
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 12px 16px;
  border-bottom: 2px solid var(--slate-200);
  text-align: left;
}
.transcript-table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--slate-100);
  vertical-align: top;
  line-height: 1.55;
}
.transcript-table tbody tr {
  transition: background var(--transition);
}
.transcript-table tbody tr:hover {
  background: var(--slate-50);
}
.speaker-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 14px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 12px;
  white-space: nowrap;
}
.speaker-avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}
.time-chip {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 6px;
  background: var(--slate-100);
  color: var(--slate-500);
  font-size: 12px;
  font-family: var(--font-mono);
  font-weight: 500;
  white-space: nowrap;
}

/* ── Kanban Board ───────────────────────────────────────────── */
.kanban-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 8px;
}
.kanban-col {
  padding: 18px;
  min-height: 240px;
  border: 1px solid var(--slate-200);
  border-radius: var(--radius-md);
  background: var(--slate-50);
}
.kanban-col h4 {
  margin: 0 0 16px 0;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--slate-500);
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
}
.kanban-col h4 .count-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  font-size: 11px;
  font-weight: 700;
  color: white;
}
.kanban-card {
  padding: 16px;
  margin-bottom: 14px;
  background: white;
  border: 1px solid var(--slate-200);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-xs);
  transition: all var(--transition);
  position: relative;
  overflow: hidden;
}
.kanban-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
.kanban-card .priority-ribbon {
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  border-radius: 4px 0 0 4px;
}
.kanban-card .assignee-avatar {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  font-size: 14px;
  font-weight: 700;
  color: white;
  flex-shrink: 0;
}
.kanban-meta {
  color: var(--slate-500);
  font-size: 12px;
  margin-top: 10px;
  display: flex;
  align-items: center;
  gap: 14px;
}

/* ── Chat / Agent ───────────────────────────────────────────── */
.chat-agent-badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  background: var(--brand-50);
  color: var(--brand-600);
  font-size: 11px;
  font-weight: 600;
  margin-bottom: 6px;
}

/* ── Empty States ───────────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 56px 24px;
  text-align: center;
  color: var(--slate-400);
}
.empty-state .empty-icon {
  font-size: 56px;
  margin-bottom: 14px;
  opacity: 0.5;
}
.empty-state .empty-title {
  font-size: 17px;
  font-weight: 600;
  color: var(--slate-600);
  margin-bottom: 6px;
}
.empty-state .empty-desc {
  font-size: 13px;
  max-width: 360px;
  color: var(--slate-400);
  line-height: 1.5;
}

/* ── Toast / Feedback ───────────────────────────────────────── */
.success-toast {
  padding: 10px 18px;
  border-radius: var(--radius-sm);
  background: var(--green-50);
  border: 1px solid #bbf7d0;
  color: #15803d;
  font-size: 13px;
  font-weight: 500;
}
.error-toast {
  padding: 12px 18px;
  border-radius: var(--radius-sm);
  background: var(--red-50);
  border: 1px solid #fecaca;
  color: #991b1b;
  font-size: 13px;
  line-height: 1.5;
}

/* ── Animations ─────────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-in {
  animation: fadeInUp 0.35s ease-out both;
}
.fade-in-d1 { animation-delay: 0.05s; }
.fade-in-d2 { animation-delay: 0.10s; }
.fade-in-d3 { animation-delay: 0.15s; }

/* ── Tabs customisation ─────────────────────────────────────── */
.tabs {
  gap: 0 !important;
}
.tabs button {
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  color: var(--slate-500) !important;
  border-bottom: 2px solid transparent !important;
  transition: all var(--transition) !important;
}
.tabs button.selected {
  color: var(--brand-600) !important;
  border-bottom-color: var(--brand-500) !important;
  font-weight: 600 !important;
}
.tabs button:hover:not(.selected) {
  color: var(--slate-700) !important;
  background: var(--slate-100) !important;
}

/* ── Accordion ──────────────────────────────────────────────── */
.accordion {
  border: 1px solid var(--slate-200) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
}
.accordion > .label-wrap {
  padding: 10px 16px !important;
  background: var(--slate-50) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}
.accordion > .label-wrap:hover {
  background: var(--slate-100) !important;
}

/* ── Dataframe ──────────────────────────────────────────────── */
table.dataframe {
  font-size: 13px !important;
}
table.dataframe th {
  background: var(--slate-100) !important;
  color: var(--slate-600) !important;
  font-weight: 600 !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  padding: 10px 14px !important;
  border-bottom: 2px solid var(--slate-200) !important;
}
table.dataframe td {
  padding: 8px 14px !important;
  border-bottom: 1px solid var(--slate-100) !important;
}

/* ── File / Download buttons ────────────────────────────────── */
.file-preview {
  border: 1px dashed var(--slate-300) !important;
  border-radius: var(--radius-md) !important;
  padding: 8px !important;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: var(--slate-300);
  border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--slate-400);
}

/* ── Responsive ─────────────────────────────────────────────── */
@media (max-width: 820px) {
  .gradio-container { padding: 12px 16px 32px !important; }
  .metric-grid { grid-template-columns: repeat(2, 1fr); }
  .kanban-grid { grid-template-columns: 1fr; }
  .hero-title { font-size: 28px; }
  .hero-wrap { padding: 24px 20px; }
}
"""

# ── Helper functions ──────────────────────────────────────────────────────


def _safe_path(uploaded_file: Any) -> Optional[str]:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    if hasattr(uploaded_file, "name"):
        return uploaded_file.name
    if isinstance(uploaded_file, dict) and "name" in uploaded_file:
        return uploaded_file["name"]
    return None


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _speaker_color(index: int) -> Tuple[str, str]:
    """Return (text_color, bg_color) for a speaker index."""
    pair = SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
    return pair


def _priority_html(priority: str) -> str:
    style = PRIORITY_STYLES.get(priority.lower(), PRIORITY_STYLES["medium"])
    return f'<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:999px;background:{style[1]};color:{style[0]};font-size:12px;font-weight:600;">{style[2]} {priority.title()}</span>'


def _status_html(status: str) -> str:
    style = STATUS_STYLES.get(status.lower(), STATUS_STYLES["open"])
    label = status.replace("-", " ").title()
    return f'<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;background:{style[1]};color:{style[0]};font-size:12px;font-weight:600;">{style[2]} {label}</span>'


def _initials(name: str) -> str:
    parts = name.replace("_", " ").replace("-", " ").split()
    return "".join(p[0].upper() for p in parts if p)[:2] or "?" if name else "?"


# ── HTML Builders ─────────────────────────────────────────────────────────


def _empty_state_html(icon: str, title: str, desc: str) -> str:
    return (
        f'<div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{html.escape(title)}</div>'
        f'<div class="empty-desc">{html.escape(desc)}</div>'
        f'</div>'
    )


def _build_transcript_html(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return _empty_state_html("🎙️", "No transcript available", "Run the meeting pipeline or load a saved meeting to view the speaker-attributed transcript.")

    seen_speakers: Dict[str, int] = {}
    speaker_idx = 0

    rows = []
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        if speaker not in seen_speakers:
            seen_speakers[speaker] = speaker_idx
            speaker_idx += 1
        idx = seen_speakers[speaker]
        txt_color, bg_color = _speaker_color(idx)

        start = _format_timestamp(seg.get("start", 0.0))
        end = _format_timestamp(seg.get("end", 0.0))
        text = seg.get("text", "").replace("\n", " ")

        rows.append(
            f"<tr>"
            f'<td><span class="time-chip">{start}</span></td>'
            f'<td><span class="time-chip">{end}</span></td>'
            f'<td><span class="speaker-badge" style="background:{bg_color};color:{txt_color};"><span class="speaker-avatar" style="background:{txt_color};color:{bg_color};">{_initials(speaker)}</span>{html.escape(speaker)}</span></td>'
            f"<td>{html.escape(text)}</td>"
            f"</tr>"
        )

    return (
        '<div class="transcript-wrap fade-in">'
        '<table class="transcript-table">'
        "<thead><tr>"
        "<th style='width:60px;'>Start</th>"
        "<th style='width:60px;'>End</th>"
        "<th style='width:170px;'>Speaker</th>"
        "<th>Transcript</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>"
    )


def _build_transcript_html_from_text(transcript_text: str, participants: List[str]) -> str:
    """Build a simple transcript view from plain text when segments are not available."""
    if not transcript_text or not transcript_text.strip():
        return _empty_state_html("🎙️", "No transcript available", "This meeting was loaded from history, but the transcript text was not stored.")

    paragraphs = [p.strip() for p in transcript_text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [transcript_text.strip()]

    rows = []
    for idx, para in enumerate(paragraphs):
        speaker = participants[idx % len(participants)] if participants else f"Speaker {idx + 1}"
        txt_color, bg_color = _speaker_color(idx)
        rows.append(
            f"<tr>"
            f'<td><span class="time-chip">--:--</span></td>'
            f'<td><span class="time-chip">--:--</span></td>'
            f'<td><span class="speaker-badge" style="background:{bg_color};color:{txt_color};"><span class="speaker-avatar" style="background:{txt_color};color:{bg_color};">{_initials(speaker)}</span>{html.escape(speaker)}</span></td>'
            f"<td>{html.escape(para)}</td>"
            f"</tr>"
        )

    return (
        '<div class="transcript-wrap fade-in">'
        '<table class="transcript-table">'
        "<thead><tr>"
        "<th style='width:60px;'>Start</th>"
        "<th style='width:60px;'>End</th>"
        "<th style='width:170px;'>Speaker</th>"
        "<th>Transcript</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table></div>"
    )


def _build_summary_html(summary: Dict[str, Any]) -> str:
    if not summary:
        return _empty_state_html("📝", "No summary available", "Run the meeting pipeline or load a saved meeting to see the structured summary.")

    blocks: List[str] = []

    executive = summary.get("executive_summary", "")
    if executive:
        blocks.append(
            '<div class="summary-block fade-in fade-in-d1">'
            "<h4>📋 Executive Summary</h4>"
            f"<p>{html.escape(executive)}</p>"
            "</div>"
        )

    decisions = summary.get("key_decisions") or []
    if decisions:
        items = "".join(f'<div class="check-item">{html.escape(str(d))}</div>' for d in decisions if str(d).strip())
        blocks.append(
            '<div class="summary-block fade-in fade-in-d2">'
            "<h4>✅ Key Decisions</h4>"
            f"{items}"
            "</div>"
        )

    topics = summary.get("discussion_topics") or []
    if topics:
        pills = []
        for t in topics:
            label = t if isinstance(t, str) else t.get("topic", "")
            positions = t.get("positions", "") if isinstance(t, dict) else ""
            text = f"{label}: {positions}" if positions else label
            pills.append(f'<span class="topic-item">{html.escape(text)}</span>')
        blocks.append(
            '<div class="summary-block fade-in fade-in-d3">'
            "<h4>💬 Discussion Topics</h4>"
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;">{" ".join(pills)}</div>'
            "</div>"
        )

    unresolved = summary.get("unresolved_items") or []
    if unresolved:
        items = "".join(f'<div class="check-item" style="color:#dc2626;">{html.escape(str(u))}</div>' for u in unresolved if str(u).strip())
        blocks.append(
            '<div class="summary-block fade-in fade-in-d3">'
            "<h4>❓ Unresolved Items</h4>"
            f"{items}"
            "</div>"
        )

    if not blocks:
        return _empty_state_html("📝", "Summary is empty", "The summary data exists but contains no content.")

    return '<div class="summary-wrap">' + "".join(blocks) + "</div>"


def _build_meeting_details_html(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return _empty_state_html("📊", "No meeting loaded", "Upload audio, paste a transcript, or load a saved meeting to see details here.")

    title = metadata.get("title") or metadata.get("meeting_source") or "Current meeting"
    meeting_date = metadata.get("meeting_date") or "N/A"
    meeting_source = metadata.get("meeting_source") or "N/A"
    meeting_id = metadata.get("meeting_id") or "N/A"
    participant_count = metadata.get("participant_count")
    action_count = metadata.get("action_count")
    participants = metadata.get("participants") or []
    participant_text = ", ".join(str(p) for p in participants) if participants else "N/A"

    metrics = ""
    for label, icon, value in [
        ("Date", "📅", meeting_date),
        ("Source", "📁", meeting_source),
        ("Meeting ID", "🆔", meeting_id),
        ("Participants", "👥", str(participant_count) if participant_count is not None else "N/A"),
        ("Action Items", "📌", str(action_count) if action_count is not None else "N/A"),
    ]:
        metrics += (
            f'<div class="metric-card fade-in">'
            f'<div class="metric-icon">{icon}</div>'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>'
            f"</div>"
        )

    participant_pills = ""
    for i, p in enumerate(participants):
        txt_color, bg_color = _speaker_color(i)
        participant_pills += (
            f'<span class="speaker-badge" style="background:{bg_color};color:{txt_color};margin:2px;">'
            f'<span class="speaker-avatar" style="background:{txt_color};color:{bg_color};">{_initials(p)}</span>'
            f"{html.escape(str(p))}"
            f"</span> "
        )

    return (
        f'<div class="glass-card padded" style="margin-bottom:14px;">'
        f"<h3 style='margin:0 0 6px 0;color:var(--slate-900);font-size:22px;font-weight:700;letter-spacing:-0.03em;'>{html.escape(str(title))}</h3>"
        f'{participant_pills if participant_pills else ""}'
        f'<div class="metric-grid">{metrics}</div>'
        f"</div>"
    )


def _build_kanban_html(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return _empty_state_html("📋", "No action items yet", "Run the meeting pipeline to extract action items from the transcript.")

    columns: Dict[str, List[Dict[str, Any]]] = {"open": [], "in-progress": [], "completed": []}
    for row in rows:
        status = str(row.get("status", "open")).lower()
        if status not in columns:
            status = "open"
        columns[status].append(row)

    col_html = []
    for i, (status, items) in enumerate(columns.items()):
        pstyle = STATUS_STYLES.get(status, STATUS_STYLES["open"])
        label = status.replace("-", " ").title()
        cards_html = ""
        for item in items:
            assignee = str(item.get("assignee", "TBD"))
            deadline = str(item.get("deadline", "TBD"))
            action = str(item.get("action_item", ""))
            priority = str(item.get("priority", "medium")).lower()
            pr = PRIORITY_STYLES.get(priority, PRIORITY_STYLES["medium"])
            is_overdue = deadline.lower() not in ("tbd", "", "none", "no deadline") and deadline != "TBD"

            cards_html += (
                f'<div class="kanban-card fade-in">'
                f'<div class="priority-ribbon" style="background:{pr[0]};"></div>'
                f'<div style="display:flex;align-items:flex-start;gap:12px;margin-left:4px;">'
                f'<span class="assignee-avatar" style="background:{_speaker_color(hash(assignee) % 100)[0]};">{_initials(assignee)}</span>'
                f"<div style='flex:1;min-width:0;'>"
                f"<strong style='color:var(--slate-900);font-size:14px;'>{html.escape(assignee)}</strong>"
                f"<p style='margin:6px 0;color:var(--slate-600);font-size:13px;line-height:1.5;'>{html.escape(action)}</p>"
                f'<div class="kanban-meta">'
                f"<span>{'🔴' if is_overdue else '📅'} {html.escape(deadline)}</span>"
                f"{_priority_html(priority)}"
                f"</div>"
                f"</div></div></div>"
            )

        col_html.append(
            f'<div class="kanban-col">'
            f"<h4><span class='count-badge' style='background:{pstyle[0]};'>{len(items)}</span> {label}</h4>"
            f'{cards_html or "<p style=color:var(--slate-400);font-size:13px;padding:16px 0;text-align:center;>No items</p>"}'
            f"</div>"
        )

    return f'<div class="kanban-grid">{"".join(col_html)}</div>'


def _build_health_report_html(report: Dict[str, Any]) -> str:
    if not report:
        return _empty_state_html("🏥", "No health data", "Run the meeting pipeline to generate an action item health report.")

    status_counts = report.get("status_counts", {})
    overdue = report.get("overdue_items", [])
    recurring = report.get("recurring_items", [])

    parts = [
        f'<div class="glass-card padded" style="margin-bottom:14px;">'
        f"<h4 style='margin:0 0 12px 0;color:var(--brand-600);'>🏥 Action Item Health Report</h4>"
        f'<div class="metric-grid">'
        f'<div class="metric-card"><div class="metric-icon">🏷️</div><div class="metric-label">Status</div><div class="metric-value">{report.get("meeting_date") or "N/A"}</div></div>'
        f'<div class="metric-card"><div class="metric-icon">👥</div><div class="metric-label">Participants</div><div class="metric-value">{report.get("participant_count", 0)}</div></div>'
    ]
    for k, v in status_counts.items():
        icon_map = {"open": "🔵", "in-progress": "🟡", "completed": "🟢"}
        icon = icon_map.get(k, "○")
        parts.append(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-label">{k.title()}</div><div class="metric-value">{v}</div></div>')
    parts.append("</div>")

    if overdue:
        parts.append("<h4 style='margin:20px 0 10px 0;color:#dc2626;font-size:14px;font-weight:600;'>⚠️ Overdue Items</h4><ul style='margin:0;padding:0;list-style:none;'>")
        for item in overdue:
            parts.append(f"<li style='padding:8px 12px;margin-bottom:6px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;font-size:13px;color:#991b1b;'><strong>{html.escape(item.get('action_item',''))}</strong> — due {html.escape(item.get('deadline','') or 'N/A')}</li>")
        parts.append("</ul>")

    if recurring:
        parts.append("<h4 style='margin:20px 0 10px 0;color:#d97706;font-size:14px;font-weight:600;'>🔁 Recurring Items</h4><ul style='margin:0;padding:0;list-style:none;'>")
        for item in recurring:
            parts.append(f"<li style='padding:8px 12px;margin-bottom:6px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;font-size:13px;color:#92400e;'><strong>{html.escape(item.get('current_action_item',''))}</strong> (prev: {html.escape(item.get('previous_meeting_source',''))})</li>")
        parts.append("</ul>")

    parts.append("</div>")
    return "".join(parts)


def _build_topic_continuity_html(topic_continuity: Optional[Dict[str, Any]]) -> str:
    """Build an HTML summary of cross-meeting topic continuity from ChromaDB data."""
    if not topic_continuity:
        return _empty_state_html("🔗", "No topic continuity data", "Process a meeting to see recurring topics and related historical context.")

    recurring_topics = topic_continuity.get("recurring_topics") or []
    related = topic_continuity.get("related") or []

    blocks: List[str] = []

    # ── Recurring Topics ────────────────────────────────────────
    if recurring_topics:
        cards = []
        for topic in recurring_topics:
            count = topic.get("count", 0)
            label = topic.get("topic", "Unknown topic")
            severity = "#dc2626" if count >= 5 else "#d97706"
            bg = "#fef2f2" if count >= 5 else "#fffbeb"
            cards.append(
                f'<div style="display:flex;align-items:center;gap:12px;padding:14px 16px;'
                f'margin-bottom:10px;background:{bg};border:1px solid {severity}33;'
                f'border-radius:10px;border-left:4px solid {severity};">'
                f'<span style="font-size:24px;font-weight:700;color:{severity};min-width:36px;">{count}x</span>'
                f'<div><strong style="color:var(--slate-900);font-size:14px;">{html.escape(str(label))}</strong>'
                f'<p style="margin:2px 0 0;color:var(--slate-500);font-size:12px;">mentioned in {count} meeting(s)</p></div>'
                f'</div>'
            )
        blocks.append(
            '<div class="summary-block fade-in fade-in-d1">'
            '<h4 style="color:#d97706;">🔁 Recurring Topics</h4>'
            '<p style="font-size:13px;color:var(--slate-500);margin:0 0 12px;">Topics appearing across multiple meetings — may indicate unresolved discussions.</p>'
            f'{"".join(cards)}'
            '</div>'
        )
    else:
        blocks.append(
            '<div class="summary-block fade-in">'
            '<h4 style="color:#16a34a;">🔁 Recurring Topics</h4>'
            '<p style="font-size:14px;color:var(--slate-500);margin:0;">No recurring topics detected yet. Topics will appear here as more meetings are processed.</p>'
            '</div>'
        )

    # ── Related Historical Context ──────────────────────────────
    if related:
        items = []
        for entry in related:
            doc = entry.get("document", "") or ""
            meta = entry.get("metadata") or {}
            meeting_source = meta.get("meeting_source", "Unknown meeting")
            doc_type = meta.get("type", "document")
            type_icon = {"summary": "📋", "action_item": "📌", "topic": "💬"}.get(doc_type, "📄")
            distance = entry.get("distance", 0)
            relevance = max(0, min(100, round((1 - distance) * 100)))
            items.append(
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 0;'
                f'border-bottom:1px solid var(--slate-100);">'
                f'<span style="font-size:16px;flex-shrink:0;">{type_icon}</span>'
                f'<div style="flex:1;min-width:0;">'
                f'<strong style="font-size:13px;color:var(--slate-700);">{html.escape(str(meeting_source))}</strong>'
                f'<span style="margin-left:8px;font-size:11px;color:var(--slate-400);">'
                f'{"●" if relevance > 70 else "◐" if relevance > 40 else "○"} {relevance}% match</span>'
                f'<p style="margin:3px 0 0;font-size:13px;color:var(--slate-600);line-height:1.5;">{html.escape(doc[:200])}{"…" if len(doc) > 200 else ""}</p>'
                f'</div></div>'
            )
        blocks.append(
            '<div class="summary-block fade-in fade-in-d2">'
            '<h4 style="color:var(--brand-600);">📚 Related Historical Context</h4>'
            f'{"".join(items)}'
            '</div>'
        )
    else:
        blocks.append(
            '<div class="summary-block fade-in">'
            '<h4 style="color:var(--slate-400);">📚 Related Historical Context</h4>'
            '<p style="font-size:14px;color:var(--slate-500);margin:0;">No related historical content found. Context will appear here from previously indexed meetings.</p>'
            '</div>'
        )

    return '<div class="summary-wrap">' + "".join(blocks) + "</div>"


def _build_report_markdown(
    meeting_source: str,
    meeting_date: str,
    transcript_data: Dict[str, Any],
    summary: Dict[str, Any],
    action_items: List[Dict[str, Any]],
    health_report: Dict[str, Any],
    recurring_topics: List[Dict[str, Any]],
) -> str:
    lines = [
        f"# Meeting Report",
        f"**Source:** {meeting_source}",
        f"**Date:** {meeting_date}",
        "",
        "## Transcript",
        transcript_data.get("transcript", ""),
        "",
        "## Summary",
        summary.get("executive_summary", ""),
        "",
    ]
    if summary.get("key_decisions"):
        lines.append("### Key Decisions")
        lines.extend(f"- {item}" for item in summary.get("key_decisions", []))
        lines.append("")
    if summary.get("discussion_topics"):
        lines.append("### Discussion Topics")
        for topic in summary.get("discussion_topics", []):
            if isinstance(topic, dict):
                lines.append(f"- **{topic.get('topic')}**: {topic.get('positions')}")
            else:
                lines.append(f"- {topic}")
        lines.append("")
    if summary.get("unresolved_items"):
        lines.append("### Unresolved Items")
        lines.extend(f"- {item}" for item in summary.get("unresolved_items", []))
        lines.append("")
    lines.append("## Action Items")
    if action_items:
        for idx, item in enumerate(action_items, start=1):
            lines.extend([
                f"### {idx}. {item.get('action_item')}",
                f"- Assignee: {item.get('assignee')}",
                f"- Deadline: {item.get('deadline')}",
                f"- Priority: {item.get('priority')}",
                f"- Context: {item.get('context_quote')}",
                f"- Status: {item.get('status', 'open')}",
                "",
            ])
    else:
        lines.append("No action items were detected.")
    lines.append("")
    lines.append("## Health Report")
    lines.extend([
        f"- Participant count: {health_report.get('participant_count', 0)}",
        f"- Total action items: {health_report.get('total_action_items', 0)}",
        f"- Reference date: {health_report.get('reference_date', '')}",
    ])
    if health_report.get("overdue_items"):
        lines.append("### Overdue Items")
        for item in health_report.get("overdue_items", []):
            lines.append(f"- {item.get('action_item')} ({item.get('deadline')})")
    if recurring_topics:
        lines.append("### Recurring Topics")
        for topic in recurring_topics:
            lines.append(f"- {topic.get('topic')} ({topic.get('count')} meetings)")
    return "\n".join(lines)


def _build_pdf_report(markdown_text: str, output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: List[Any] = []
    for paragraph in markdown_text.split("\n\n"):
        clean = paragraph.replace("&", "&").replace("<", "<").replace(">", ">")
        style = styles["Heading2"] if paragraph.startswith("## ") else styles["BodyText"]
        if paragraph.startswith("## "):
            story.append(Paragraph(paragraph.replace("## ", ""), styles["Heading2"]))
        elif paragraph.startswith("### "):
            story.append(Paragraph(paragraph.replace("### ", ""), styles["Heading3"]))
        elif paragraph.startswith("- "):
            story.append(Paragraph(paragraph.replace("- ", "• "), styles["BodyText"]))
        else:
            story.append(Paragraph(clean, style))
        story.append(Spacer(1, 0.1 * inch))
    doc.build(story)
    return output_path


def _file_output_value(path: Any) -> Optional[str]:
    if not path:
        return None
    file_path = Path(str(path)).expanduser()
    if not file_path.is_file():
        return None
    return str(file_path)


def _default_ui_state() -> Dict[str, Any]:
    return {
        "meeting_details_html": "",
        "meeting_metadata": {},
        "transcript_html": "",
        "transcript_text": "",
        "summary_html": "",
        "summary": {},
        "action_rows": [],
        "action_items": [],
        "kanban_html": "",
        "topic_continuity_html": "",
        "report_md": None,
        "report_pdf": None,
    }


def _normalize_action_rows(rows: Any) -> List[List[Any]]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [list(row) if isinstance(row, (list, tuple)) else [row] for row in rows]
    if isinstance(rows, dict):
        data = rows.get("data", [])
        return _normalize_action_rows(data)
    if hasattr(rows, "values") and hasattr(rows.values, "tolist"):
        return rows.values.tolist()
    if hasattr(rows, "to_numpy"):
        return rows.to_numpy().tolist()
    return []


def _normalize_ui_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = _default_ui_state()
    if not isinstance(state, dict):
        return base
    normalized = dict(base)
    normalized.update({k: state.get(k) for k in base.keys() if k in state})
    normalized["action_rows"] = _normalize_action_rows(normalized.get("action_rows"))
    normalized["report_md"] = _file_output_value(normalized.get("report_md"))
    normalized["report_pdf"] = _file_output_value(normalized.get("report_pdf"))
    return normalized


def _rows_to_action_objects(rows: Any) -> List[Dict[str, Any]]:
    objects = []
    for row in _normalize_action_rows(rows):
        padded = list(row) + [""] * max(0, 6 - len(row))
        objects.append({
            "id": padded[0],
            "action_item": padded[1],
            "assignee": padded[2],
            "deadline": padded[3],
            "priority": padded[4],
            "status": padded[5],
            "context_quote": "",
        })
    return objects


def _meeting_choice_label(row: sqlite3.Row) -> str:
    title = row["title"] or row["meeting_source"] or "Untitled meeting"
    meeting_date = row["meeting_date"] or "unknown date"
    action_count = row["action_count"] or 0
    return f"{meeting_date} · {title} · {action_count} action item(s) · {row['meeting_id']}"


def _extract_meeting_id(choice: Any) -> Optional[str]:
    if not choice:
        return None
    return str(choice).rsplit(" · ", 1)[-1].strip()


def _list_meeting_choices(db_path: str = DEFAULT_HISTORY_DB) -> List[str]:
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        has_meetings_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meetings'"
        ).fetchone()
        if not has_meetings_table:
            return []
        rows = conn.execute(
            """
            SELECT
                m.meeting_id,
                m.meeting_source,
                m.meeting_date,
                m.title,
                COUNT(a.id) AS action_count
            FROM meetings m
            LEFT JOIN action_items a ON a.meeting_id = m.meeting_id
            GROUP BY m.meeting_id, m.meeting_source, m.meeting_date, m.title, m.updated_at, m.id
            ORDER BY m.updated_at DESC, m.id DESC
            """
        ).fetchall()
    return [_meeting_choice_label(row) for row in rows]


def _load_meeting_state(meeting_id: str, db_path: str = DEFAULT_HISTORY_DB) -> Dict[str, Any]:
    state = _default_ui_state()
    if not meeting_id or not Path(db_path).exists():
        return state

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        meeting = conn.execute(
            """
            SELECT meeting_id, meeting_source, meeting_date, title, summary_json, participants_json, transcript_text, transcript_segments_json
            FROM meetings
            WHERE meeting_id = ?
            """,
            (meeting_id,),
        ).fetchone()
        if meeting is None:
            return state

        rows = conn.execute(
            """
            SELECT id, action_item, assignee, deadline, priority, status
            FROM action_items
            WHERE meeting_id = ?
            ORDER BY id ASC
            """,
            (meeting["meeting_id"],),
        ).fetchall()

    summary = {}
    if meeting["summary_json"]:
        try:
            summary = json.loads(meeting["summary_json"])
        except json.JSONDecodeError:
            summary = {}
    participants = []
    if meeting["participants_json"]:
        try:
            participants = json.loads(meeting["participants_json"])
        except json.JSONDecodeError:
            participants = []
    transcript_segments = []
    if meeting["transcript_segments_json"]:
        try:
            transcript_segments = json.loads(meeting["transcript_segments_json"])
        except json.JSONDecodeError:
            transcript_segments = []
    action_rows = [
        [
            row["id"],
            row["action_item"],
            row["assignee"],
            row["deadline"] or "",
            row["priority"] or "",
            row["status"] or "open",
        ]
        for row in rows
    ]
    meeting_label = meeting["title"] or meeting["meeting_source"] or "selected meeting"
    meeting_date = meeting["meeting_date"] or "unknown date"
    meeting_metadata = {
        "meeting_id": meeting["meeting_id"],
        "meeting_source": meeting["meeting_source"],
        "meeting_date": meeting["meeting_date"],
        "title": meeting["title"],
        "participants": participants,
        "participant_count": len(participants),
        "action_count": len(action_rows),
    }
    transcript_text = meeting["transcript_text"] or ""
    state["meeting_metadata"] = meeting_metadata
    state["meeting_details_html"] = _build_meeting_details_html(meeting_metadata)
    state["summary"] = summary
    state["summary_html"] = _build_summary_html(summary)
    state["transcript_text"] = transcript_text
    # Prefer timestamped segments; fall back to plain-text reconstruction
    if transcript_segments:
        state["transcript_html"] = _build_transcript_html(transcript_segments)
    else:
        state["transcript_html"] = _build_transcript_html_from_text(transcript_text, participants)
    state["action_rows"] = action_rows
    state["action_items"] = _rows_to_action_objects(action_rows)
    state["kanban_html"] = _build_kanban_html(state["action_items"])
    if not state["summary_html"] or "No summary available" in state["summary_html"]:
        state["summary_html"] = _empty_state_html(
            "📝", "Summary restored from history",
            f"Showing saved data for <em>{html.escape(meeting_label)}</em> ({html.escape(meeting_date)}). "
            "Summary text was not stored for this meeting, but action items and transcript are available."
        )
    return state


def _load_latest_action_history_state(db_path: str = DEFAULT_HISTORY_DB) -> Dict[str, Any]:
    choices = _list_meeting_choices(db_path)
    if choices:
        meeting_id = _extract_meeting_id(choices[0])
        if meeting_id:
            return _load_meeting_state(meeting_id, db_path)

    state = _default_ui_state()
    if not Path(db_path).exists():
        return state

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        latest = conn.execute(
            """
            SELECT meeting_source, meeting_date
            FROM action_items
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        if latest is None:
            return state

        rows = conn.execute(
            """
            SELECT id, action_item, assignee, deadline, priority, status
            FROM action_items
            WHERE COALESCE(meeting_source, '') = COALESCE(?, '')
              AND COALESCE(meeting_date, '') = COALESCE(?, '')
            ORDER BY id ASC
            """,
            (latest["meeting_source"], latest["meeting_date"]),
        ).fetchall()

    action_rows = [
        [
            row["id"],
            row["action_item"],
            row["assignee"],
            row["deadline"] or "",
            row["priority"] or "",
            row["status"] or "open",
        ]
        for row in rows
    ]
    meeting_label = latest["meeting_source"] or "last saved meeting"
    meeting_date = latest["meeting_date"] or "unknown date"
    meeting_metadata = {
        "meeting_source": latest["meeting_source"],
        "meeting_date": latest["meeting_date"],
        "title": meeting_label,
        "action_count": len(action_rows),
    }
    state["meeting_metadata"] = meeting_metadata
    state["meeting_details_html"] = _build_meeting_details_html(meeting_metadata)
    state["action_rows"] = action_rows
    state["kanban_html"] = _build_kanban_html(_rows_to_action_objects(action_rows))
    state["summary_html"] = _empty_state_html(
        "📝", "Summary not stored",
        f"Showing saved action items for <em>{html.escape(meeting_label)}</em> ({html.escape(meeting_date)}). "
        "Transcript and summary text are not stored in the current SQLite schema."
    )
    return state


def _restore_ui_from_state(state: Optional[Dict[str, Any]]) -> Tuple[str, str, str, str, List[List[Any]], str, str, Optional[str], Optional[str], Dict[str, Any], str]:
    restored = _normalize_ui_state(state)
    return (
        restored["meeting_details_html"],
        restored["transcript_html"],
        restored["transcript_text"],
        restored["summary_html"],
        restored["action_rows"],
        restored["kanban_html"],
        restored["topic_continuity_html"],
        restored["report_md"],
        restored["report_pdf"],
        restored,
        "",
    )


def _refresh_meeting_choices() -> Tuple[Any, str]:
    choices = _list_meeting_choices()
    value = choices[0] if choices else None
    msg = f"Found {len(choices)} saved meeting(s)." if choices else "No saved meetings found."
    return gr.update(choices=choices, value=value), msg


def _load_selected_meeting(choice: Any) -> Tuple[str, str, str, str, List[List[Any]], str, str, Optional[str], Optional[str], Dict[str, Any], str]:
    meeting_id = _extract_meeting_id(choice)
    state = _load_meeting_state(meeting_id) if meeting_id else _default_ui_state()
    if not meeting_id:
        state["meeting_details_html"] = _empty_state_html("📊", "Select a meeting", "Choose a meeting from the dropdown, then click 'Load selected meeting'.")
    return _restore_ui_from_state(state)


def _create_persisted_ui_state() -> Any:
    default_state = _default_ui_state()
    if hasattr(gr, "BrowserState"):
        return gr.BrowserState(default_value=default_state, storage_key="meeting_summarizer_ui_state")
    return gr.State(value=default_state)


def _make_action_items_rows(
    action_items: ActionItemList,
    saved_ids: List[int],
) -> Tuple[List[List[Any]], List[str], List[Dict[str, Any]]]:
    rows: List[List[Any]] = []
    objects: List[Dict[str, Any]] = []
    for idx, item in enumerate(action_items.action_items):
        item_id = saved_ids[idx] if idx < len(saved_ids) else None
        row = [item_id or 0, item.action_item, item.assignee, item.deadline, item.priority, "open"]
        rows.append(row)
        objects.append({
            "id": item_id,
            "action_item": item.action_item,
            "assignee": item.assignee,
            "deadline": item.deadline,
            "priority": item.priority,
            "status": "open",
            "context_quote": item.context_quote,
        })
    return rows, ["id", "action_item", "assignee", "deadline", "priority", "status"], objects


def _update_statuses(rows: Any) -> Tuple[str, Dict[str, Any]]:
    history_agent = ActionHistoryAgent(db_path=DEFAULT_HISTORY_DB)
    items = []
    for row in _normalize_action_rows(rows):
        try:
            item_id = int(row[0])
        except Exception:
            continue
        status = str(row[5]).strip().lower()
        if status not in VALID_STATUSES:
            status = "open"
        history_agent.update_status(item_id, status)
        items.append({
            "id": item_id,
            "action_item": row[1],
            "assignee": row[2],
            "deadline": row[3],
            "priority": row[4],
            "status": status,
            "context_quote": "",
        })
    kanban_html = _build_kanban_html(items)
    return kanban_html, {"updated_items": len(items)}


def _update_statuses_and_persist(rows: Any, state: Optional[Dict[str, Any]]) -> Tuple[str, str, Dict[str, Any]]:
    normalized_rows = _normalize_action_rows(rows)
    kanban_html, update_info = _update_statuses(normalized_rows)
    persisted = _normalize_ui_state(state)
    persisted["action_rows"] = normalized_rows
    persisted["kanban_html"] = kanban_html
    return kanban_html, json.dumps(update_info), persisted


def _persist_table_edits(rows: Any, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    persisted = _normalize_ui_state(state)
    persisted["action_rows"] = _normalize_action_rows(rows)
    return persisted


def process_meeting_with_error_handling(
    audio_file: Optional[Any],
    transcript_text: str,
    hf_token: str,
    whisper_model: str,
    summary_model: str,
    action_model: str,
    meeting_name: str,
    skip_diarization: bool = False,
) -> Tuple[Any, ...]:
    try:
        audio_path = _safe_path(audio_file)
        meeting_source = meeting_name or (Path(audio_path).name if audio_path else "manual_transcript")
        meeting_date = datetime.date.today().isoformat()
        transcript_data: Optional[Dict[str, Any]] = None
        if not audio_path and transcript_text and transcript_text.strip():
            transcript_data = {
                "file": meeting_source,
                "model": None,
                "transcript": transcript_text.strip(),
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": "unknown",
                        "text": transcript_text.strip(),
                    }
                ],
            }
        else:
            if not audio_path:
                raise ValueError("Please upload audio or paste transcript text.")

        workflow_state = run_meeting_workflow(
            {
                "audio_path": audio_path,
                "transcript_data": transcript_data,
                "meeting_source": meeting_source,
                "meeting_date": meeting_date,
                "transcript_model_name": whisper_model,
                "summary_model_name": summary_model,
                "action_item_model_name": action_model,
                "hf_token": hf_token or None,
                "skip_diarization": skip_diarization,
                "history_db_path": DEFAULT_HISTORY_DB,
                "history_chroma_dir": "./.chromadb",
            }
        )
        transcript_data = workflow_state["transcript_data"]
        summary = workflow_state["summary"]
        action_items = workflow_state["action_items"]
        saved_ids = workflow_state.get("saved_action_item_ids", [])
        health_report = workflow_state.get("action_history_report", {})
        rows, headers, action_objects = _make_action_items_rows(action_items, saved_ids)
        meeting_metadata = {
            "meeting_id": workflow_state.get("meeting_id"),
            "meeting_source": meeting_source,
            "meeting_date": meeting_date,
            "title": meeting_name or meeting_source,
            "participants": health_report.get("participants", []),
            "participant_count": health_report.get("participant_count"),
            "action_count": len(action_objects),
        }
        meeting_details_html = _build_meeting_details_html(meeting_metadata)
        transcript_html = _build_transcript_html(transcript_data.get("segments", []))
        summary_html = _build_summary_html(summary)
        kanban_html = _build_kanban_html(action_objects)
        topic_continuity = workflow_state.get("topic_continuity")
        topic_continuity_html = _build_topic_continuity_html(topic_continuity)
        recurring_topics_from_continuity = (topic_continuity or {}).get("recurring_topics", [])
        report_md = _build_report_markdown(
            meeting_source, meeting_date, transcript_data, summary,
            action_objects, health_report, recurring_topics_from_continuity,
        )
        report_md_path = Path(tempfile.mkdtemp()) / "meeting_report.md"
        report_md_path.write_text(report_md, encoding="utf-8")
        report_pdf_path = Path(tempfile.mkdtemp()) / "meeting_report.pdf"
        try:
            _build_pdf_report(report_md, str(report_pdf_path))
        except Exception:
            report_pdf_path = Path("")
        return (
            meeting_details_html,
            transcript_html,
            transcript_data.get("transcript", ""),
            summary_html,
            rows,
            kanban_html,
            topic_continuity_html,
            _file_output_value(report_md_path),
            _file_output_value(report_pdf_path),
            summary,
            action_objects,
            meeting_metadata,
        )
    except Exception as e:
        error_text = html.escape(str(e))
        error_html = f'<div class="error-toast">❌ <strong>Error:</strong> {error_text}</div>'
        return (error_html, error_html, error_text, error_html, [], "", "", None, None, {}, [], {})


def _chat_with_agents(
    message: str,
    history: Optional[List[Dict[str, Any]]],
    target: str,
    state: Optional[Dict[str, Any]],
    model_name: str,
    hf_token: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    chat_history: List[Dict[str, Any]] = []
    for entry in history or []:
        if isinstance(entry, dict) and "role" in entry and "content" in entry:
            chat_history.append(entry)
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            chat_history.extend([
                {"role": "user", "content": str(entry[0])},
                {"role": "assistant", "content": str(entry[1])},
            ])
    target_key = next((key for key, label in TARGETS.items() if label == target), target or "auto")
    result = NLPQueryAgent(
        history_db_path=DEFAULT_HISTORY_DB,
        chroma_dir="./.chromadb",
    ).ask(
        question=message,
        target=target_key,
        state=_normalize_ui_state(state),
        model_name=model_name or DEFAULT_SUMMARY_MODEL,
        hf_token=hf_token or None,
    )
    agent_label = TARGETS.get(result["agent"], result["agent"])
    answer = f'<span class="chat-agent-badge">🤖 {agent_label}</span>\n\n{result["answer"]}'
    chat_history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    return "", chat_history


# ── Launch UI ──────────────────────────────────────────────────────────────

def launch_ui() -> None:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="teal",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("Space Mono"),
        text_size="md",
        spacing_size="md",
        radius_size="md",
    ).set(
        body_background_fill="transparent",
        block_background_fill="transparent",
        block_border_width="0",
        button_primary_background_fill="linear-gradient(135deg, #4f46e5, #3730a3)",
        button_primary_background_fill_hover="linear-gradient(135deg, #6366f1, #4338ca)",
        button_primary_text_color="white",
        button_primary_shadow="0 4px 14px rgba(79,70,229,0.35)",
        button_primary_shadow_hover="0 6px 20px rgba(79,70,229,0.40)",
        button_large_padding="12px 28px",
        button_small_padding="6px 16px",
        input_border_width="1px",
        input_border_color="#e2e8f0",
        input_background_fill="white",
        input_radius="8px",
        checkbox_label_border_width="1px",
        shadow_drop="0 8px 30px rgba(15,23,42,0.10)",
        shadow_drop_lg="0 20px 60px rgba(15,23,42,0.14)",
    )

    with gr.Blocks(title="Meeting Summarizer Dashboard") as demo:
        persisted_ui_state = _create_persisted_ui_state()

        # ── Hero ────────────────────────────────────────────────
        gr.HTML(
            '<div class="hero-wrap fade-in">'
            '<h1 class="hero-title">Meeting Summarizer</h1>'
            '<p class="hero-sub">Upload audio, paste a transcript, or load a saved meeting — get structured summaries, action items, and topic continuity in seconds.</p>'
            '<span class="hero-badge">⚡ AI-Powered Pipeline</span>'
            '</div>'
        )

        # ── Main Layout ─────────────────────────────────────────
        with gr.Row(equal_height=False):
            # ── Left Panel ──
            with gr.Column(scale=2, min_width=300):
                # ── Input Card ──
                with gr.Group(elem_classes=["glass-card", "padded"]):
                    gr.Markdown("### 🎯 Input")
                    audio_input = gr.Audio(sources="upload", type="filepath", label="Upload meeting audio")
                    transcript_input = gr.Textbox(
                        lines=3,
                        label="Or paste transcript text",
                        placeholder="Paste meeting transcript here...",
                    )
                    meeting_name = gr.Textbox(
                        label="Meeting name / source",
                        placeholder="e.g. Weekly sync, client call",
                    )
                    skip_diarization = gr.Checkbox(
                        label="Skip speaker diarization (faster processing)",
                        value=False,
                        info="Disable if you don't need speaker labels. Significantly speeds up processing.",
                    )
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        hf_token = gr.Textbox(
                            label="Hugging Face token",
                            type="password",
                            placeholder="Optional, falls back to HF_TOKEN env",
                        )
                        whisper_model = gr.Dropdown(
                            label="Whisper model",
                            value=DEFAULT_WHISPER_MODEL,
                            choices=["tiny", "base", "small", "medium", "large"],
                        )
                        summary_model = gr.Textbox(
                            label="Summary model",
                            value=DEFAULT_SUMMARY_MODEL,
                        )
                        action_model = gr.Textbox(
                            label="Action item model",
                            value=DEFAULT_ACTION_MODEL,
                        )
                    with gr.Row():
                        process_button = gr.Button("🚀 Run Pipeline", elem_classes="gradient-btn", scale=3, variant="primary")
                        clear_btn = gr.Button("🗑️ Clear", elem_classes="secondary-btn", scale=1)

                # ── Saved Meetings Card ──
                with gr.Group(elem_classes=["glass-card", "padded"]):
                    gr.Markdown("### 📂 Saved Meetings")
                    with gr.Row():
                        meeting_selector = gr.Dropdown(
                            label="Select a meeting",
                            choices=_list_meeting_choices(),
                            value=None,
                            interactive=True,
                            scale=3,
                        )
                        refresh_meetings_button = gr.Button(
                            "🔄",
                            elem_classes="secondary-btn",
                            scale=1,
                            min_width=44,
                        )
                    load_meeting_button = gr.Button(
                        "📥 Load Selected Meeting",
                        elem_classes="secondary-btn",
                        variant="secondary",
                    )

                # ── Pipeline Status ──
                pipeline_status = gr.HTML(label="")

            # ── Right Panel ──
            with gr.Column(scale=5):
                meeting_details_output = gr.HTML(label="Meeting details")

                with gr.Tabs(elem_classes="tabs"):
                    with gr.TabItem("📝 Transcript"):
                        transcript_view = gr.HTML(label="Speaker-attributed transcript")
                        with gr.Accordion("📄 Raw transcript text", open=False):
                            transcript_text_output = gr.Textbox(
                                lines=8,
                                label="Full transcript text",
                            )

                    with gr.TabItem("📋 Summary"):
                        summary_output = gr.HTML(label="Structured meeting summary")

                    with gr.TabItem("✅ Action Items"):
                        status_update_output = gr.Textbox(label="Status", visible=True)
                        action_table = gr.Dataframe(
                            headers=["id", "action_item", "assignee", "deadline", "priority", "status"],
                            label="Action item dashboard",
                            interactive=True,
                            elem_classes="glass-card",
                        )
                        with gr.Row():
                            save_status_button = gr.Button(
                                "💾 Save Status Updates",
                                elem_classes="gradient-btn",
                                scale=2,
                            )
                        kanban_output = gr.HTML(label="Kanban board")

                    with gr.TabItem("🔗 Topic Continuity"):
                        topic_continuity_output = gr.HTML(label="Cross-meeting topic context")

                    with gr.TabItem("📄 Report"):
                        with gr.Row():
                            report_md_file = gr.File(label="📥 Download Markdown Report")
                            report_pdf_file = gr.File(label="📥 Download PDF Report")

                    with gr.TabItem("💬 Ask Agents"):
                        gr.Markdown("Ask about the current meeting, historical action items, or related topics.")
                        agent_target = gr.Dropdown(
                            label="Agent",
                            choices=list(TARGETS.values()),
                            value=TARGETS["auto"],
                        )
                        agent_chat = gr.Chatbot(
                            label="Agent conversation",
                            height=400,
                        )
                        with gr.Row():
                            agent_question = gr.Textbox(
                                label="Question",
                                placeholder='e.g., "Which action items assigned to Mike are overdue?"',
                                scale=4,
                            )
                            ask_button = gr.Button(
                                "Ask",
                                elem_classes="gradient-btn",
                                scale=1,
                                variant="primary",
                            )

        # ── Callbacks ────────────────────────────────────────────────
        def process_and_persist(*args):
            result = process_meeting_with_error_handling(*args)
            state_payload = {
                "meeting_details_html": result[0],
                "transcript_html": result[1],
                "transcript_text": result[2],
                "summary_html": result[3],
                "action_rows": result[4],
                "kanban_html": result[5],
                "topic_continuity_html": result[6],
                "report_md": result[7],
                "report_pdf": result[8],
                "summary": result[9],
                "action_items": result[10],
                "meeting_metadata": result[11],
            }
            choices = _list_meeting_choices()
            selected = choices[0] if choices else None
            return result[:9] + (state_payload, gr.update(choices=choices, value=selected), '<div class="success-toast">✅ Meeting processed and saved successfully.</div>')

        process_btn_inputs = [audio_input, transcript_input, hf_token, whisper_model, summary_model, action_model, meeting_name, skip_diarization]
        process_btn_outputs = [
            meeting_details_output, transcript_view, transcript_text_output, summary_output,
            action_table, kanban_output, topic_continuity_output, report_md_file, report_pdf_file,
            persisted_ui_state, meeting_selector, pipeline_status,
        ]
        # Enable queue for longer timeouts
        demo.queue(api_open=True, max_size=10)
        
        process_button.click(fn=process_and_persist, inputs=process_btn_inputs, outputs=process_btn_outputs)

        def clear_all():
            empty = _default_ui_state()
            empty_details = _empty_state_html("📊", "No meeting loaded", "Upload audio, paste a transcript, or load a saved meeting.")
            empty_transcript = _empty_state_html("🎙️", "No transcript", "Run the meeting pipeline to see the transcript.")
            empty_summary = _empty_state_html("📝", "No summary", "Run the meeting pipeline to see the summary.")
            empty_kanban = _empty_state_html("📋", "No action items", "Run the meeting pipeline to extract action items.")
            return (
                empty_details, empty_transcript, "", empty_summary, [],
                empty_kanban, "", None, None, empty, "",
            )

        clear_btn_outputs = [
            meeting_details_output, transcript_view, transcript_text_output, summary_output,
            action_table, kanban_output, topic_continuity_output, report_md_file, report_pdf_file,
            persisted_ui_state, pipeline_status,
        ]
        clear_btn.click(fn=clear_all, outputs=clear_btn_outputs)

        demo.load(
            fn=_restore_ui_from_state,
            inputs=[persisted_ui_state],
            outputs=[
                meeting_details_output, transcript_view, transcript_text_output, summary_output,
                action_table, kanban_output, topic_continuity_output, report_md_file, report_pdf_file,
                persisted_ui_state, pipeline_status,
            ],
        )

        refresh_meetings_button.click(
            fn=_refresh_meeting_choices,
            outputs=[meeting_selector, pipeline_status],
        )

        load_meeting_button.click(
            fn=_load_selected_meeting,
            inputs=[meeting_selector],
            outputs=[
                meeting_details_output, transcript_view, transcript_text_output, summary_output,
                action_table, kanban_output, topic_continuity_output, report_md_file, report_pdf_file,
                persisted_ui_state, pipeline_status,
            ],
        )

        action_table.change(
            fn=_persist_table_edits,
            inputs=[action_table, persisted_ui_state],
            outputs=[persisted_ui_state],
        )

        save_status_button.click(
            fn=_update_statuses_and_persist,
            inputs=[action_table, persisted_ui_state],
            outputs=[kanban_output, status_update_output, persisted_ui_state],
        )

        chat_inputs = [agent_question, agent_chat, agent_target, persisted_ui_state, summary_model, hf_token]
        ask_button.click(
            fn=_chat_with_agents,
            inputs=chat_inputs,
            outputs=[agent_question, agent_chat],
        )
        agent_question.submit(
            fn=_chat_with_agents,
            inputs=chat_inputs,
            outputs=[agent_question, agent_chat],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, theme=theme, css=DASHBOARD_CSS)


if __name__ == "__main__":
    launch_ui()
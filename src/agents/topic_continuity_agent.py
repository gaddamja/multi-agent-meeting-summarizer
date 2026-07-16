"""Topic Continuity Agent using ChromaDB + sentence-transformers.

Embeds meeting summaries, action items, and topic texts into ChromaDB to
provide meeting-over-meeting continuity and retrieve related historical
context for a new meeting. Flags topics discussed multiple times without
resolution.
"""
from __future__ import annotations

import os
import sys
import warnings
from io import StringIO
from typing import Any, Dict, List, Optional

# Suppress torchcodec/libtorchcodec loading errors
warnings.filterwarnings("ignore", message=".*Could not load libtorchcodec.*")
warnings.filterwarnings("ignore", message=".*built-in audio decoding will fail.*")
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

class _SuppressStderr:
    """Context manager to suppress stderr output during torch imports."""
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stderr = self.original_stderr

# Import torch-dependent libraries with stderr suppressed to hide torchcodec errors
with _SuppressStderr():
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        chromadb = None
    
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

from uuid import uuid4
from datetime import datetime


DEFAULT_CHROMA_DIR = "./.chromadb"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def _log(prefix: str, message: str) -> None:
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{prefix}] {message}")


class TopicContinuityAgent:
    def __init__(self, persist_directory: str = DEFAULT_CHROMA_DIR, embed_model: str = DEFAULT_EMBED_MODEL):
        if chromadb is None:
            raise RuntimeError("chromadb is required: pip install chromadb sentence-transformers")
        
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required: pip install sentence-transformers")

        # Use new Chroma API (deprecated Settings-based configuration)
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except TypeError:
            # Fallback for older versions
            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
            self.client = chromadb.Client(settings)
        
        self.collection_name = "meetings"
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(self.collection_name)

        _log("topics", f"Loading sentence transformer model: {embed_model}")
        self.model = SentenceTransformer(embed_model)
        _log("topics", "✓ Sentence transformer model loaded")

    def _embed_texts(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def index_meeting(self, meeting_id: str, meeting_source: str, summary: str, action_items: List[Dict[str, Any]], topics: List[str]):
        """Index a meeting: summary, each action item, and topics as separate documents."""
        _log("topics", f"Indexing meeting: {meeting_id}")
        docs = []
        metadatas = []
        ids = []

        # summary
        ids.append(f"{meeting_id}-summary")
        docs.append(summary or "")
        metadatas.append({"meeting_id": meeting_id, "meeting_source": meeting_source, "type": "summary"})

        # action items
        for i, ai in enumerate(action_items or []):
            ids.append(f"{meeting_id}-action-{i}")
            docs.append(ai.get("action_item", ""))
            metadatas.append({"meeting_id": meeting_id, "meeting_source": meeting_source, "type": "action_item", "assignee": ai.get("assignee")})

        # topics
        for i, t in enumerate(topics or []):
            ids.append(f"{meeting_id}-topic-{i}")
            docs.append(t)
            metadatas.append({"meeting_id": meeting_id, "meeting_source": meeting_source, "type": "topic"})

        if not docs:
            _log("topics", f"No documents to index for {meeting_id}")
            return None

        _log("topics", f"Generating embeddings for {len(docs)} documents...")
        embeddings = self._embed_texts(docs)

        try:
            # upsert: delete existing ids then add
            existing = [d for d in ids]
            # Chroma client API: upsert via add with ids
            self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        except Exception:
            # fallback: try simple add
            self.collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

        # persist if supported
        try:
            self.client.persist()
        except Exception:
            pass

        _log("topics", f"✓ Indexed {len(ids)} documents for {meeting_id}")
        return ids

    def query_related(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return k related historical documents for the given text."""
        if not text:
            return []
        _log("topics", f"Querying related topics (k={k})...")
        emb = self._embed_texts([text])[0]
        results = self.collection.query(query_embeddings=[emb], n_results=k, include=["metadatas", "documents", "distances"])
        # results is a dict with keys 'ids','distances','metadatas','documents'
        out = []
        for i in range(len(results.get("ids", [[]])[0])):
            out.append({
                "id": results.get("ids", [[]])[0][i],
                "document": results.get("documents", [[]])[0][i],
                "metadata": results.get("metadatas", [[]])[0][i],
                "distance": results.get("distances", [[]])[0][i],
            })
        _log("topics", f"✓ Found {len(out)} related documents")
        return out

    def find_recurring_topics(self, k: int = 10, threshold: int = 3) -> List[Dict[str, Any]]:
        """Simple heuristic: scan topic documents and count occurrences by normalized text.
        Returns topics with counts >= threshold.
        """
        _log("topics", f"Finding recurring topics (threshold={threshold})...")
        # fetch all topic metadatas/documents
        # Chroma collection has get method 'get' to retrieve all
        snapshot = self.collection.get(include=["ids", "documents", "metadatas"]) or {}
        docs = snapshot.get("documents", [])
        metas = snapshot.get("metadatas", [])

        counts = {}
        examples = {}
        for doc, meta in zip(docs, metas):
            if meta.get("type") != "topic":
                continue
            key = doc.strip().lower()
            counts[key] = counts.get(key, 0) + 1
            examples[key] = (doc, meta)

        recurring = []
        for key, cnt in counts.items():
            if cnt >= threshold:
                doc, meta = examples.get(key, (key, {}))
                recurring.append({"topic": doc, "count": cnt, "example_meta": meta})

        _log("topics", f"✓ Found {len(recurring)} recurring topics")
        return recurring

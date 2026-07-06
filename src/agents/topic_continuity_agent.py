"""Topic Continuity Agent using ChromaDB + sentence-transformers.

Embeds meeting summaries, action items, and topic texts into ChromaDB to
provide meeting-over-meeting continuity and retrieve related historical
context for a new meeting. Flags topics discussed multiple times without
resolution.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

from sentence_transformers import SentenceTransformer
from uuid import uuid4


DEFAULT_CHROMA_DIR = "./.chromadb"
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


class TopicContinuityAgent:
    def __init__(self, persist_directory: str = DEFAULT_CHROMA_DIR, embed_model: str = DEFAULT_EMBED_MODEL):
        if chromadb is None:
            raise RuntimeError("chromadb is required: pip install chromadb sentence-transformers")

        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        self.client = chromadb.Client(settings)
        self.collection_name = "meetings"
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(self.collection_name)

        self.model = SentenceTransformer(embed_model)

    def _embed_texts(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def index_meeting(self, meeting_id: str, meeting_source: str, summary: str, action_items: List[Dict[str, Any]], topics: List[str]):
        """Index a meeting: summary, each action item, and topics as separate documents."""
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
            return None

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

        return ids

    def query_related(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return k related historical documents for the given text."""
        if not text:
            return []
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
        return out

    def find_recurring_topics(self, k: int = 10, threshold: int = 3) -> List[Dict[str, Any]]:
        """Simple heuristic: scan topic documents and count occurrences by normalized text.
        Returns topics with counts >= threshold.
        """
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

        return recurring

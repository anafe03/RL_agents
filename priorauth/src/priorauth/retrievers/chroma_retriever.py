"""ChromaDB retriever — dense-vector retrieval with ChromaDB's default ONNX embeddings.

Uses ChromaDB's in-memory client and its bundled default embedding
function — ONNX-backed `all-MiniLM-L6-v2`. No torch / no sentence-transformers
required (those would add ~600MB+ to the deploy image). The ONNX model
is small (~80 MB), downloads on first use, runs locally.

For the public Streamlit demo, the first request after a cold start
takes a few seconds while the model loads; subsequent requests are
fast. Wrap UI calls in `@st.cache_resource` to avoid re-loading per
session.
"""

from __future__ import annotations

import uuid

import chromadb

from priorauth.models import Case, Guideline
from priorauth.retrievers.base import Retriever, case_to_query


class ChromaRetriever(Retriever):
    name = "chroma_minilm"
    category = "dense_vector"

    def __init__(self) -> None:
        super().__init__()
        self._client = chromadb.Client()
        # Unique collection name per instance so multiple retrievers don't collide.
        self._collection_name = f"priorauth_guidelines_{uuid.uuid4().hex[:8]}"
        self._collection = None
        self._corpus: list[Guideline] = []

    def index(self, guidelines: list[Guideline]) -> None:
        self._corpus = list(guidelines)
        # Recreate the collection cleanly each time index() is called.
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        # Use ChromaDB's default embedding function (ONNX-based all-MiniLM-L6-v2).
        self._collection = self._client.create_collection(name=self._collection_name)
        if not self._corpus:
            return
        self._collection.add(
            ids=[g.id for g in self._corpus],
            documents=[f"{g.topic}. {g.excerpt}" for g in self._corpus],
            metadatas=[{"source": g.source, "topic": g.topic} for g in self._corpus],
        )

    def retrieve(self, case: Case, k: int = 5) -> list[Guideline]:
        if self._collection is None or not self._corpus:
            return []
        query = case_to_query(case)
        result = self._collection.query(query_texts=[query], n_results=min(k, len(self._corpus)))
        ids = (result.get("ids") or [[]])[0]
        by_id = {g.id: g for g in self._corpus}
        return [by_id[i] for i in ids if i in by_id]

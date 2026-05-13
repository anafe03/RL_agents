"""BM25 retriever — classical keyword-based retrieval.

Tokenizes each guideline excerpt + each query, indexes via Okapi BM25
(from `rank-bm25`), returns top-k by BM25 score. No embeddings, no API
calls, no GPU — fully local and instant.

Useful as a *baseline* in the benchmark: how much does a vector retriever
actually beat dumb keyword search on this corpus? The answer is often
"less than you'd think" for small, well-curated corpora.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from priorauth.models import Case, Guideline
from priorauth.retrievers.base import Retriever, case_to_query

_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever(Retriever):
    name = "bm25"
    category = "keyword"

    def __init__(self) -> None:
        super().__init__()
        self._corpus: list[Guideline] = []
        self._bm25: BM25Okapi | None = None

    def index(self, guidelines: list[Guideline]) -> None:
        self._corpus = list(guidelines)
        docs = [
            _tokenize(f"{g.topic} {g.excerpt} {' '.join(g.tags)}")
            for g in self._corpus
        ]
        self._bm25 = BM25Okapi(docs) if docs else None

    def retrieve(self, case: Case, k: int = 5) -> list[Guideline]:
        if self._bm25 is None or not self._corpus:
            return []
        query_tokens = _tokenize(case_to_query(case))
        scores = self._bm25.get_scores(query_tokens)
        # Sort descending; drop zero-score hits to avoid noise
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        return [self._corpus[i] for i, s in ranked[:k] if s > 0]

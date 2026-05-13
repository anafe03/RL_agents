"""Retriever backends + registry."""

from priorauth.retrievers.base import Retriever, case_to_query
from priorauth.retrievers.bm25 import BM25Retriever
from priorauth.retrievers.llm_judged import LLMJudgedRetriever

# Chroma is imported lazily — it pulls in sentence-transformers which is heavy
# and not always desired in CI / minimal environments.


def _chroma_factory():
    from priorauth.retrievers.chroma_retriever import ChromaRetriever
    return ChromaRetriever()


REGISTRY: dict[str, callable] = {
    "bm25": lambda: BM25Retriever(),
    "chroma_minilm": _chroma_factory,
    "llm_judged": lambda: LLMJudgedRetriever(),
}


def get_retriever(name: str) -> Retriever:
    if name not in REGISTRY:
        raise ValueError(f"Unknown retriever: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = [
    "BM25Retriever",
    "LLMJudgedRetriever",
    "REGISTRY",
    "Retriever",
    "case_to_query",
    "get_retriever",
]

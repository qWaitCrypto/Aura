from __future__ import annotations

import importlib.util

import pytest

from aura.runtime.knowledge.agno_adapter import AgnoKnowledgeStore
from aura.runtime.knowledge.config import KnowledgeConfig


def test_rag_search_optional_backend(tmp_path):
    has_chroma = importlib.util.find_spec("chromadb") is not None
    has_lance = importlib.util.find_spec("lancedb") is not None
    if not has_chroma and not has_lance:
        pytest.skip("No supported VectorDB dependency installed (chromadb/lancedb).")

    config = KnowledgeConfig(
        enabled=True,
        retriever="bm25",
        # Prefer chromadb because lancedb.connect can hang in some environments.
        vector_db="chromadb" if has_chroma else "lancedb",
    )
    store = AgnoKnowledgeStore(config=config, project_root=tmp_path)
    store.add_document("test content", source="test.md")
    results = store.search("test", max_results=5)
    assert results
    assert results[0].source == "test.md"

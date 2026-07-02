"""FastAPI integration tests for the retrieval engine."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from retrieval_evaluation_framework.api.app import _PIPELINE_CACHE, app
from retrieval_evaluation_framework.models import Chunk, Document, FileType, ProcessedCorpus

CLIENT = TestClient(app)


def _write_config(tmp_path: Path, index_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "device: cpu",
                "logging:",
                "  level: INFO",
                "embedding:",
                "  model_name: local-hash",
                "  backend: hashing",
                "  device: cpu",
                "  batch_size: 4",
                "  show_progress: false",
                "index:",
                f"  index_directory: {index_path.as_posix()}",
                "retrieval:",
                "  retriever: bm25",
                "  top_k: 2",
                "query_enhancement:",
                "  enabled: false",
                "  method: none",
                "reranking:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_corpus(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "processed_corpus.json"
    document = Document(
        id="doc-1",
        title="handbook",
        text=(
            "Maternity leave policy covers 16 weeks of paid leave. "
            "Remote work policy allows hybrid schedules."
        ),
        source="memory",
        file_type=FileType.TXT,
    )
    chunks = [
        Chunk(
            chunk_id="doc-1:0",
            document_id="doc-1",
            text="Maternity leave policy covers 16 weeks of paid leave.",
            position=0,
            token_count=9,
        ),
        Chunk(
            chunk_id="doc-1:1",
            document_id="doc-1",
            text="Remote work policy allows hybrid schedules.",
            position=1,
            token_count=6,
        ),
    ]
    corpus = ProcessedCorpus(
        device="cpu",
        documents=[document],
        chunks=chunks,
        statistics={
            "document_count": 1,
            "chunk_count": 2,
            "average_chunk_tokens": 7.5,
        },
        config_snapshot={},
    )
    corpus.save_json(corpus_path)
    return corpus_path


def test_openapi_schema_is_available() -> None:
    response = CLIENT.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "Retrieval Evaluation & Optimization Framework"


def test_embed_endpoint_persists_embeddings(tmp_path: Path) -> None:
    _PIPELINE_CACHE.clear()
    index_path = tmp_path / "index"
    config_path = _write_config(tmp_path, index_path)
    corpus_path = _write_corpus(tmp_path)
    embeddings_path = tmp_path / "embeddings"

    response = CLIENT.post(
        "/embed",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "output_path": str(embeddings_path),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_path"] == str(embeddings_path)
    assert body["chunk_count"] == 2
    assert (embeddings_path / "vectors.npy").exists()
    assert (embeddings_path / "metadata.json").exists()


def test_index_and_retrieve_endpoints_work(tmp_path: Path) -> None:
    _PIPELINE_CACHE.clear()
    index_path = tmp_path / "index"
    config_path = _write_config(tmp_path, index_path)
    corpus_path = _write_corpus(tmp_path)

    index_response = CLIENT.post(
        "/index",
        json={
            "config_path": str(config_path),
            "corpus_path": str(corpus_path),
            "index_path": str(index_path),
        },
    )

    assert index_response.status_code == 200
    assert index_response.json()["retriever"] == "bm25"

    _PIPELINE_CACHE.clear()
    retrieve_response = CLIENT.post(
        "/retrieve",
        json={
            "config_path": str(config_path),
            "index_path": str(index_path),
            "query": "What is the maternity leave policy?",
            "top_k": 1,
        },
    )

    assert retrieve_response.status_code == 200
    retrieval_body = retrieve_response.json()
    assert retrieval_body["retriever"] == "bm25"
    assert len(retrieval_body["results"]) == 1
    assert retrieval_body["results"][0]["chunk"]["chunk_id"] == "doc-1:0"

    search_response = CLIENT.post(
        "/search",
        json={
            "config_path": str(config_path),
            "index_path": str(index_path),
            "query": "hybrid schedules",
            "top_k": 1,
        },
    )

    assert search_response.status_code == 200
    search_body = search_response.json()
    assert search_body["retriever"] == "bm25"
    assert len(search_body["results"]) == 1
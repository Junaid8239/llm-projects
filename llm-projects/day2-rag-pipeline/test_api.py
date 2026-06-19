import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    """Health endpoint should return 200 and status ok"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_missing_file_returns_404():
    """Ingesting a file that doesn't exist should return 404, not crash"""
    response = client.post("/ingest", json={"pdf_path": "nonexistent_file.pdf"})
    assert response.status_code == 404


def test_ingest_non_pdf_returns_400():
    """Ingesting a non-PDF file should be rejected with 400"""
    response = client.post("/ingest", json={"pdf_path": "README.md"})
    assert response.status_code == 400


def test_query_missing_collection_returns_404():
    """Querying a collection that was never ingested should return 404"""
    response = client.post("/query", json={
        "collection_name": "nonexistent_collection",
        "question": "What is this about?"
    })
    assert response.status_code == 404


def test_query_missing_question_field_returns_422():
    """Pydantic should reject a request missing the required 'question' field"""
    response = client.post("/query", json={
        "collection_name": "attention_paper"
        # missing required 'question' field
    })
    assert response.status_code == 422


def test_query_real_question_returns_valid_answer():
    """End-to-end: ingest then query should return a grounded answer with sources"""
    # First ensure the collection exists by ingesting
    client.post("/ingest", json={"pdf_path": "attention_paper.pdf"})

    response = client.post("/query", json={
        "collection_name": "attention_paper",
        "question": "What is multi-head attention?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert len(data["answer"]) > 0
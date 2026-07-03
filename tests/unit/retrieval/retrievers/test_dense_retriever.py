"""Tests for DenseRetriever."""

import pytest
from unittest.mock import Mock

from src.retrieval.retrievers.dense_retriever import DenseRetriever
from src.libs.vector_store.base import BaseVectorStore


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock(spec=BaseVectorStore)
    store.search_dense.return_value = [
        {"id": "doc1", "text": "Result 1", "score": 0.95},
        {"id": "doc2", "text": "Result 2", "score": 0.85},
        {"id": "doc3", "text": "Result 3", "score": 0.75}
    ]
    return store


def test_dense_retriever_instantiation(mock_vector_store):
    """DenseRetriever can be instantiated."""
    retriever = DenseRetriever(mock_vector_store)
    assert retriever.vector_store == mock_vector_store


def test_retrieve_basic(mock_vector_store):
    """Retrieve documents successfully."""
    retriever = DenseRetriever(mock_vector_store)
    query_vector = [0.1] * 128

    results = retriever.retrieve(query_vector, top_k=10)

    assert len(results) == 3
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] == 0.95


def test_retrieve_calls_vector_store(mock_vector_store):
    """Retrieve calls vector store search_dense."""
    retriever = DenseRetriever(mock_vector_store)
    query_vector = [0.1] * 128

    retriever.retrieve(query_vector, top_k=5)

    mock_vector_store.search_dense.assert_called_once_with(
        query_vector=query_vector,
        top_k=5,
        filter_dict=None
    )


def test_retrieve_with_top_k(mock_vector_store):
    """Retrieve respects top_k parameter."""
    retriever = DenseRetriever(mock_vector_store)
    query_vector = [0.1] * 128

    retriever.retrieve(query_vector, top_k=20)

    call_kwargs = mock_vector_store.search_dense.call_args[1]
    assert call_kwargs["top_k"] == 20


def test_retrieve_with_filters(mock_vector_store):
    """Retrieve applies metadata filters."""
    retriever = DenseRetriever(mock_vector_store)
    query_vector = [0.1] * 128
    filters = {"category": "technical", "year": 2024}

    retriever.retrieve(query_vector, top_k=10, filter_dict=filters)

    call_kwargs = mock_vector_store.search_dense.call_args[1]
    assert call_kwargs["filter_dict"] == filters


def test_retrieve_returns_results_from_store(mock_vector_store):
    """Retrieve returns results from vector store."""
    retriever = DenseRetriever(mock_vector_store)
    query_vector = [0.1] * 128

    results = retriever.retrieve(query_vector)

    assert results == mock_vector_store.search_dense.return_value


def test_retrieve_with_empty_results(mock_vector_store):
    """Retrieve handles empty results."""
    mock_vector_store.search_dense.return_value = []
    retriever = DenseRetriever(mock_vector_store)
    query_vector = [0.1] * 128

    results = retriever.retrieve(query_vector)

    assert results == []

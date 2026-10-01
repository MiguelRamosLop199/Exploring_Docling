"""Tests de contrato para ``docling_explorer.chunker``."""

from pathlib import Path

import pytest
from docling_core.transforms.chunker import BaseChunk
from transformers import AutoTokenizer

from docling_explorer.chunker import MaxTokenLimitingChunker, chunk_document

SAMPLE_PDF = Path(__file__).parents[1] / "resources" / "JerarquiaDocs.pdf"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def test_chunk_document_returns_base_chunks():
    chunks = chunk_document(SAMPLE_PDF, max_tokens=150)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, BaseChunk)
        assert chunk.text.strip()


def test_chunk_document_respects_max_tokens():
    max_tokens = 150
    chunks = chunk_document(SAMPLE_PDF, max_tokens=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for chunk in chunks:
        token_count = len(
            tokenizer(chunk.text, add_special_tokens=False)["input_ids"]
        )
        assert token_count <= max_tokens, (
            f"Chunk excede el límite de tokens ({token_count} > {max_tokens}): "
            f"{chunk.text[:80]!r}..."
        )


@pytest.mark.parametrize("max_tokens", [50, 100, 250])
def test_chunk_document_with_various_limits(max_tokens: int):
    chunks = chunk_document(SAMPLE_PDF, max_tokens=max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    assert len(chunks) > 0
    for chunk in chunks:
        token_count = len(
            tokenizer(chunk.text, add_special_tokens=False)["input_ids"]
        )
        assert token_count <= max_tokens


def test_max_token_limiting_chunker_class_signature():
    """El chunker expone la interfaz BaseChunker y conserva la firma."""
    chunker = MaxTokenLimitingChunker(max_tokens=150)
    assert chunker.max_tokens == 150

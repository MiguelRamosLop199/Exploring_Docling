"""Tests de contrato para ``docling_explorer.chunker``."""

from pathlib import Path

import pytest
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
from docling_core.transforms.chunker import BaseChunk
from docling_core.types.doc import DoclingDocument as DLDocument
from pydantic import ValidationError
from transformers import AutoTokenizer

from docling_explorer.chunker import MaxTokenLimitingChunker, chunk_document

SAMPLE_PDF = Path(__file__).parents[1] / "resources" / "JerarquiaDocs.pdf"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def sample_document():
    converter = DocumentConverter()
    result = converter.convert(str(SAMPLE_PDF))
    return result.document


def _token_count(text: str, tokenizer: AutoTokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_chunk_document_returns_base_chunks():
    chunks = chunk_document(SAMPLE_PDF, max_tokens=150)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, BaseChunk)
        assert chunk.text.strip()


def test_chunk_document_respects_max_tokens(tokenizer):
    max_tokens = 150
    chunks = chunk_document(SAMPLE_PDF, max_tokens=max_tokens)

    for chunk in chunks:
        token_count = _token_count(chunk.text, tokenizer)
        assert token_count <= max_tokens, (
            f"Chunk excede el límite de tokens ({token_count} > {max_tokens}): "
            f"{chunk.text[:80]!r}..."
        )


@pytest.mark.parametrize("max_tokens", [50, 100, 250])
def test_chunk_document_with_various_limits(max_tokens: int, tokenizer):
    chunks = chunk_document(SAMPLE_PDF, max_tokens=max_tokens)

    assert len(chunks) > 0
    for chunk in chunks:
        token_count = _token_count(chunk.text, tokenizer)
        assert token_count <= max_tokens


def test_chunk_document_preserves_section_headings():
    chunks = chunk_document(SAMPLE_PDF, max_tokens=150)
    chunks_with_headings = [c for c in chunks if c.meta and c.meta.headings]

    assert len(chunks_with_headings) > 0, "Se esperaba al menos un chunk con heading"
    for chunk in chunks_with_headings:
        heading = chunk.meta.headings[0]
        assert chunk.text.startswith(heading), (
            f"Chunk no conserva el título de sección: {chunk.text[:80]!r}"
        )


def test_split_chunks_preserve_section_headings(tokenizer, sample_document):
    max_tokens = 50
    chunker = MaxTokenLimitingChunker(max_tokens=max_tokens)
    chunks = list(chunker.chunk(sample_document))

    chunks_with_headings = [c for c in chunks if c.meta and c.meta.headings]
    assert len(chunks_with_headings) > 0
    for chunk in chunks_with_headings:
        heading = chunk.meta.headings[0]
        assert chunk.text.startswith(heading)
        assert _token_count(chunk.text, tokenizer) <= max_tokens


def test_empty_document_returns_empty_list():
    empty_doc = DLDocument(name="empty")
    chunker = MaxTokenLimitingChunker(max_tokens=512)
    assert list(chunker.chunk(empty_doc)) == []


def test_max_tokens_must_be_positive():
    with pytest.raises(ValidationError):
        MaxTokenLimitingChunker(max_tokens=0)
    with pytest.raises(ValidationError):
        MaxTokenLimitingChunker(max_tokens=-1)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "defecto: con max_tokens=1 se lanza ValueError porque "
        "num_tokens_avail_for_text == 0; el chunker debería manejar "
        "metadatos mayores que max_tokens"
    ),
)
def test_very_small_max_tokens_does_not_crash(tokenizer, sample_document):
    max_tokens = 1
    chunker = MaxTokenLimitingChunker(max_tokens=max_tokens)
    chunks = list(chunker.chunk(sample_document))
    for chunk in chunks:
        assert _token_count(chunk.text, tokenizer) <= max_tokens


def test_chunk_document_rejects_missing_path():
    with pytest.raises((ConversionError, FileNotFoundError)):
        chunk_document("/nonexistent/file.pdf", max_tokens=150)


def test_chunk_document_accepts_string_and_path(tokenizer):
    chunks_str = chunk_document(str(SAMPLE_PDF), max_tokens=150)
    chunks_path = chunk_document(SAMPLE_PDF, max_tokens=150)

    assert len(chunks_str) == len(chunks_path)
    assert len(chunks_str) > 0
    for chunk_str, chunk_path in zip(chunks_str, chunks_path):
        assert chunk_str.text == chunk_path.text
        assert _token_count(chunk_str.text, tokenizer) <= 150


def test_max_token_limiting_chunker_class_signature():
    """El chunker expone la interfaz BaseChunker y conserva la firma."""
    chunker = MaxTokenLimitingChunker(max_tokens=150)
    assert chunker.max_tokens == 150

"""Chunking avanzado de documentos con Docling.

Combina ``HierarchicalChunker`` con un límite rígido de tokens usando un
tokenizer de HuggingFace. Extraído del notebook ``docling_chunker.ipynb``.
"""

from __future__ import annotations

from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Iterable, Iterator, Union

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocMeta,
    HierarchicalChunker,
)
from docling_core.types.doc import DoclingDocument as DLDocument
from pydantic import ConfigDict, PositiveInt
from transformers import AutoTokenizer

PathLikeStr = Union[str, PathLike[str]]


class MaxTokenLimitingChunker(BaseChunker):
    """Chunker que refuerza un límite máximo de tokens sobre otro chunker base."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner_chunker: BaseChunker = HierarchicalChunker()
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    max_tokens: PositiveInt = 512
    delim: str = "\n"

    def _serialize_meta_to_include(self, meta: DocMeta) -> str:
        meta_parts = []
        headings_part = self.delim.join(meta.headings or [])
        if headings_part:
            meta_parts.append(headings_part)
        captions_part = self.delim.join(meta.captions or [])
        if captions_part:
            meta_parts.append(captions_part)
        return self.delim.join(meta_parts)

    def _split_text_to_token_limit(
        self, text: str, max_tokens: int
    ) -> Iterator[str]:
        """Divide ``text`` en trozos de como mucho ``max_tokens`` tokens.

        Cuando es posible, el corte se realiza en el último espacio dentro del
        presupuesto de tokens para no partir palabras entre dos chunks. Si no
        hay espacio disponible, cae al corte por límite de token (puede partir
        una palabra muy larga).
        """
        remaining = text
        while remaining:
            remaining = remaining.lstrip()
            if not remaining:
                return
            tokens = self.tokenizer(
                remaining,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
            offsets = tokens["offset_mapping"]
            if len(offsets) <= max_tokens:
                yield remaining
                return

            limit_char = offsets[max_tokens - 1][1]
            candidate = remaining[:limit_char]
            split_rel = candidate.rfind(" ")
            if split_rel > 0:
                yield candidate[:split_rel]
                remaining = remaining[split_rel + 1 :]
            else:
                # No hay límite de palabra disponible: cortamos justo en el
                # límite de token. Esto solo ocurre con palabras más largas que
                # el presupuesto de tokens.
                yield candidate
                remaining = remaining[limit_char:]

    def _split_above_max_tokens(self, chunk_iter: Iterable[BaseChunk]) -> Iterator[BaseChunk]:
        for chunk in chunk_iter:
            meta = DocMeta.model_validate(chunk.meta)
            meta_text = self._serialize_meta_to_include(meta=meta)
            meta_list = [meta_text] if meta_text else []

            meta_tokens = self.tokenizer(
                meta_text, return_offsets_mapping=True, add_special_tokens=False
            )["offset_mapping"]
            delim_tokens = (
                self.tokenizer(
                    self.delim, return_offsets_mapping=True, add_special_tokens=False
                )["offset_mapping"]
                if meta_text
                else []
            )
            num_tokens_avail_for_text = self.max_tokens - (
                len(meta_tokens) + len(delim_tokens)
            )

            text_tokens = self.tokenizer(
                chunk.text, return_offsets_mapping=True, add_special_tokens=False
            )["offset_mapping"]
            num_text_tokens = len(text_tokens)

            if num_tokens_avail_for_text <= 0:
                # El título no cabe en el presupuesto: se deja fuera del texto del
                # chunk (sigue en ``chunk.meta.headings``) y el cuerpo usa todo
                # ``max_tokens`` para no perder texto.
                meta_list = []
                num_tokens_avail_for_text = self.max_tokens

            full_ser = self.delim.join(meta_list + ([chunk.text] if chunk.text else []))

            if num_text_tokens <= num_tokens_avail_for_text:
                c = deepcopy(chunk)
                c.text = full_ser
                yield c
            else:
                for text in self._split_text_to_token_limit(
                    chunk.text, num_tokens_avail_for_text
                ):
                    c = deepcopy(chunk)
                    c.text = self.delim.join(meta_list + [text])
                    yield c

    def chunk(self, dl_doc: DLDocument, **kwargs) -> Iterator[BaseChunk]:
        chunk_iter = self.inner_chunker.chunk(dl_doc=dl_doc, **kwargs)
        yield from self._split_above_max_tokens(chunk_iter=chunk_iter)


def chunk_document(path: PathLikeStr, max_tokens: int) -> list[BaseChunk]:
    """Convierte un documento y lo divide en chunks respetando ``max_tokens``.

    Args:
        path: ruta al documento (PDF, DOCX, etc.).
        max_tokens: número máximo de tokens por chunk.

    Returns:
        Lista de chunks de Docling que no superan ``max_tokens`` tokens.
    """
    resolved_path = str(Path(path).expanduser().resolve())
    converter = DocumentConverter()
    converted_result = converter.convert(resolved_path)
    chunker = MaxTokenLimitingChunker(max_tokens=max_tokens)
    return list(chunker.chunk(converted_result.document))

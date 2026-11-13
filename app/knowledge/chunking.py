"""Split policy markdown into overlapping chunks while preserving section context."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    section: str | None


def _split_sections(body: str) -> list[tuple[str | None, str]]:
    parts = re.split(r"\n(?=##\s+)", body.strip())
    sections: list[tuple[str | None, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            first_line, _, rest = part.partition("\n")
            title = first_line.removeprefix("## ").strip()
            sections.append((title, rest.strip() or first_line))
        else:
            sections.append((None, part))
    return sections


def _window_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_markdown(
    body: str,
    *,
    max_chars: int = 900,
    overlap: int = 120,
) -> list[TextChunk]:
    """Chunk by section headings; subdivide long sections with overlap."""
    out: list[TextChunk] = []
    idx = 0
    for section, text in _split_sections(body):
        for piece in _window_chunks(text, max_chars=max_chars, overlap=overlap):
            if not piece:
                continue
            out.append(TextChunk(index=idx, content=piece, section=section))
            idx += 1
    return out

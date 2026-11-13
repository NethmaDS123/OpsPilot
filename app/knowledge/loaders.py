"""Load markdown policy files with YAML frontmatter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from typing import Literal, cast

from app.schemas.knowledge import PolicyDocumentMeta

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class LoadedPolicy:
    meta: PolicyDocumentMeta
    body: str


def _parse_scalar(value: str) -> str | None:
    value = value.strip()
    if value.lower() in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _parse_frontmatter(raw: str) -> dict[str, str | None]:
    data: dict[str, str | None] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        data[key.strip()] = _parse_scalar(rest)
    return data


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def load_policy_markdown(path: Path) -> LoadedPolicy:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"Missing YAML frontmatter in {path}")

    fields = _parse_frontmatter(match.group(1))
    required = ("document_id", "title", "company_id", "category", "version", "status")
    missing = [k for k in required if not fields.get(k)]
    if missing:
        raise ValueError(f"Missing frontmatter keys in {path}: {missing}")

    meta = PolicyDocumentMeta(
        document_id=str(fields["document_id"]),
        title=str(fields["title"]),
        company_id=str(fields["company_id"]),
        category=str(fields["category"]),
        version=str(fields["version"]),
        status=cast(Literal["active", "superseded", "draft"], fields["status"]),
        effective_from=_parse_date(fields.get("effective_from")),
        effective_to=_parse_date(fields.get("effective_to")),
        source_path=str(path),
    )
    body = text[match.end() :].strip()
    return LoadedPolicy(meta=meta, body=body)


def iter_policy_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.md"))

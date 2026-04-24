# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Template registry for AIPerf configuration scaffolding.

Metadata is embedded in each template YAML as a ``# @template`` comment block,
parsed into a Pydantic model, and cached. The YAML files are the single source
of truth.
"""

from __future__ import annotations

import functools
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CATEGORY_ORDER: tuple[str, ...] = (
    "Getting Started",
    "Load Testing",
    "Datasets",
    "Sweep & Multi-Run",
    "Advanced",
    "Multimodal",
    "Specialized Endpoints",
)

Category = Literal[
    "Getting Started",
    "Load Testing",
    "Datasets",
    "Sweep & Multi-Run",
    "Advanced",
    "Multimodal",
    "Specialized Endpoints",
]

Difficulty = Literal["beginner", "intermediate", "advanced"]

_SENTINEL = "# @template"


class TemplateInfo(BaseModel):
    """Metadata for a single configuration template."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Filename stem, e.g. 'minimal'.")
    title: str = Field(description="Human-readable title.")
    description: str = Field(description="One-line summary.")
    category: Category = Field(description="Grouping category.")
    tags: tuple[str, ...] = Field(default=(), description="Searchable tags.")
    difficulty: Difficulty = Field(default="beginner", description="Difficulty level.")
    features: tuple[str, ...] = Field(default=(), description="Features demonstrated.")


# ---------------------------------------------------------------------------
# Parsing & registry
# ---------------------------------------------------------------------------


def _templates_dir() -> Path:
    return Path(resources.files("aiperf.config") / "templates")  # type: ignore[arg-type]


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _parse_comment_block(path: Path) -> dict[str, Any]:
    """Extract ``# key: value`` pairs after the ``# @template`` sentinel."""
    lines = path.read_text(encoding="utf-8").splitlines()
    sentinel_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == _SENTINEL),
        None,
    )
    if sentinel_idx is None:
        raise ValueError(f"{path.name}: missing '{_SENTINEL}' comment block")

    meta: dict[str, Any] = {}
    for line in lines[sentinel_idx + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("# ") or ": " not in stripped[2:]:
            break
        key, _, value = stripped[2:].partition(": ")
        if not key.strip() or not value.strip():
            break
        meta[key.strip()] = value.strip()
    return meta


def parse_template_meta(path: Path) -> TemplateInfo:
    """Parse ``# @template`` block and validate via Pydantic.

    Raises:
        ValueError: If sentinel is missing or fields are invalid.
    """
    meta = _parse_comment_block(path)
    # CSV fields → tuples before Pydantic sees them
    for csv_field in ("tags", "features"):
        if csv_field in meta:
            meta[csv_field] = _split_csv(meta[csv_field])
    return TemplateInfo(name=path.stem, **meta)


@functools.lru_cache(maxsize=1)
def _load_all_templates() -> tuple[TemplateInfo, ...]:
    """Scan templates directory, parse metadata, return sorted registry."""
    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    templates = [
        parse_template_meta(p) for p in sorted(_templates_dir().glob("*.yaml"))
    ]
    templates.sort(key=lambda t: (cat_rank.get(t.category, 999), t.name))
    return tuple(templates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_templates(
    *,
    category: str | None = None,
    tag: str | None = None,
) -> list[TemplateInfo]:
    """Return templates, optionally filtered by category or tag."""
    results = list(_load_all_templates())
    if category:
        cat_lower = category.lower()
        results = [t for t in results if cat_lower in t.category.lower()]
    if tag:
        tag_lower = tag.lower()
        results = [t for t in results if any(tag_lower in tg for tg in t.tags)]
    return results


def get_template(name: str) -> TemplateInfo:
    """Look up a template by name. Raises KeyError if not found."""
    for t in _load_all_templates():
        if t.name == name:
            return t
    available = ", ".join(sorted(t.name for t in _load_all_templates()))
    raise KeyError(f"Unknown template '{name}'. Available templates: {available}")


def load_template_content(name: str) -> str:
    """Load the raw YAML content of a template by name."""
    get_template(name)
    path = _templates_dir() / f"{name}.yaml"
    return path.read_text(encoding="utf-8")


def search_templates(query: str) -> list[TemplateInfo]:
    """Search templates by keyword across all metadata fields.

    Name matches appear first, remaining matches follow in registry order.
    """
    q = query.lower()
    name_matches: list[TemplateInfo] = []
    other_matches: list[TemplateInfo] = []
    for t in _load_all_templates():
        searchable = (
            t.name,
            t.title.lower(),
            t.description.lower(),
            t.category.lower(),
            *t.tags,
            *(f.lower() for f in t.features),
        )
        if any(q in field for field in searchable):
            if q in t.name.lower():
                name_matches.append(t)
            else:
                other_matches.append(t)
    return name_matches + other_matches


def apply_overrides(content: str, overrides: dict[str, Any]) -> str:
    """Apply config overrides via ruamel.yaml round-trip (preserves comments).

    Walks the overrides dict and sets matching keys in the parsed YAML
    tree, then dumps back. All comments, formatting, and flow-style
    mappings are retained.
    """
    if not overrides:
        return content

    from io import StringIO

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    data = yaml.load(content)
    if data is None:
        return content

    _deep_set(data, overrides)

    buf = StringIO()
    yaml.dump(data, buf)
    return buf.getvalue()


def _deep_set(target: Any, overrides: dict[str, Any]) -> None:
    """Recursively apply overrides to a ruamel.yaml CommentedMap."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_set(target[key], value)
        else:
            target[key] = value


def strip_spdx_header(content: str) -> str:
    """Remove leading ``# SPDX-*`` lines from template content."""
    lines = content.splitlines(keepends=True)
    start = 0
    for line in lines:
        if line.startswith("# SPDX-"):
            start += 1
        else:
            break
    return "".join(lines[start:])

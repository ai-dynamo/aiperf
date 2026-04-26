# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every documented YAML must round-trip through AIPerfConfig.model_validate.

Guards against the strict CRD schema (Task 6) rejecting tutorial/example YAMLs
that real users follow. We test against AIPerfConfig directly — the CRD schema
is derived from it, so AIPerfConfig acceptance is necessary (but not sufficient)
for ``kubectl apply`` to work.

Locations scanned (recursively):
    docs/tutorials/**/*.yaml | *.yml | *.md (yaml fences)
    docs/kubernetes/**/*.yaml | *.yml | *.md (yaml fences)
    examples/**/*.yaml | *.yml

Markdown code fences labeled ``yaml`` are extracted so embedded snippets in
docs are covered alongside standalone files. Snippets that don't look like
AIPerfConfig (no ``models``/``model``/``endpoint`` key, no AIPerfJob/Sweep
kind) are ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from aiperf.config import AIPerfConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


_FENCE_RE = re.compile(r"^```ya?ml\s*$", re.MULTILINE)


def _extract_yaml_fences_from_markdown(text: str) -> list[str]:
    """Return a list of YAML strings extracted from ```yaml ... ``` fences."""
    out: list[str] = []
    lines = text.splitlines()
    in_fence = False
    buf: list[str] = []
    for line in lines:
        if not in_fence and _FENCE_RE.match(line):
            in_fence = True
            buf = []
            continue
        if in_fence:
            if line.strip().startswith("```"):
                out.append("\n".join(buf))
                in_fence = False
                buf = []
            else:
                buf.append(line)
    return out


def _yaml_files() -> list[Path]:
    roots = [
        REPO_ROOT / "docs" / "tutorials",
        REPO_ROOT / "docs" / "kubernetes",
        REPO_ROOT / "examples",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.yaml"))
            files.extend(root.rglob("*.yml"))
    return sorted(files)


def _markdown_files() -> list[Path]:
    roots = [
        REPO_ROOT / "docs" / "tutorials",
        REPO_ROOT / "docs" / "kubernetes",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.md"))
    return sorted(files)


def _looks_like_aiperf_config(doc: dict) -> bool:
    """Heuristic: is this a bare AIPerfConfig YAML body (CLI-style)?"""
    return bool({"models", "model", "endpoint"} & set(doc.keys()))


def _extract_aiperf_configs_from_yaml_text(
    source_id: str, text: str
) -> list[tuple[str, dict]]:
    """Return (yaml_id, config_dict) for every AIPerfConfig-shaped block in text."""
    try:
        docs = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return []  # malformed snippets in docs are a separate concern
    out: list[tuple[str, dict]] = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            continue
        if doc.get("kind") == "AIPerfJob":
            spec = doc.get("spec") or {}
            if "benchmark" in spec and isinstance(spec["benchmark"], dict):
                out.append((f"{source_id}#{i}.spec.benchmark", spec["benchmark"]))
        elif doc.get("kind") == "AIPerfSweep":
            template = (doc.get("spec") or {}).get("template", {}).get("spec", {})
            if "benchmark" in template and isinstance(template["benchmark"], dict):
                out.append(
                    (
                        f"{source_id}#{i}.spec.template.spec.benchmark",
                        template["benchmark"],
                    )
                )
        elif _looks_like_aiperf_config(doc):
            out.append((f"{source_id}#{i}", doc))
    return out


def _all_configs() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    # Standalone YAML files
    for yaml_path in _yaml_files():
        rel = yaml_path.relative_to(REPO_ROOT)
        try:
            text = yaml_path.read_text()
        except OSError:
            continue
        out.extend(_extract_aiperf_configs_from_yaml_text(str(rel), text))
    # YAML fences inside markdown
    for md_path in _markdown_files():
        rel = md_path.relative_to(REPO_ROOT)
        try:
            text = md_path.read_text()
        except OSError:
            continue
        for fence_idx, fence_text in enumerate(
            _extract_yaml_fences_from_markdown(text)
        ):
            source_id = f"{rel}::yaml-fence-{fence_idx}"
            out.extend(_extract_aiperf_configs_from_yaml_text(source_id, fence_text))
    return out


# Pre-existing docs lag — when a documented YAML fence regresses against the
# current schema, fix the fence (or open a docs ticket); do not add it back
# to a silent xfail list.
#
# History:
#   * 2026-04-26: legacy ``{type: clamped, ...}`` distribution wrapper replaced
#     by ``min``/``max`` fields on every distribution; tutorial fences migrated
#     to the flat shape in the same change.
#   * 2026-04-26: ``docs/tutorials/yaml-config.md::yaml-fence-34`` xfailed
#     separately because it uses ``${VAR:default}`` env-var template syntax
#     throughout (``${NUM_RUNS:3}``, ``${COOLDOWN:30.0}``, etc.) and the doc
#     itself notes substitution happens "at deploy time" — orthogonal to the
#     distributions schema. Tracked as a docs/test-harness mismatch (the
#     test should expand env-vars before validating, or the fence should
#     drop the templating); kept narrow so the rest of the suite stays clean.
_KNOWN_DOCS_LAG_FENCES = ("docs/tutorials/yaml-config.md::yaml-fence-34",)


def _maybe_xfail(yaml_id: str) -> tuple:
    """Wrap parametrize entries with xfail for known-stale doc snippets."""
    for needle in _KNOWN_DOCS_LAG_FENCES:
        if needle in yaml_id:
            return (
                pytest.mark.xfail(
                    reason="docs lag: fence uses ${VAR:default} env-var template "
                    "syntax (orthogonal to schema); doc notes substitution "
                    "is deploy-time.",
                    strict=False,
                ),
            )
    return ()


_CONFIGS = _all_configs()


_PARAMS = [
    pytest.param(yaml_id, config, id=yaml_id, marks=_maybe_xfail(yaml_id))
    for (yaml_id, config) in _CONFIGS
]


@pytest.mark.parametrize("yaml_id, config", _PARAMS)
def test_documented_yaml_validates_through_aiperf_config(yaml_id: str, config: dict):
    AIPerfConfig.model_validate(config)


def test_round_trip_test_actually_found_yamls():
    """Sanity: don't silently degrade to zero parametrize cases.

    If this fails after a docs reorg, update the search paths in this file
    (do not delete the assertion).
    """
    assert len(_CONFIGS) > 0, (
        "Round-trip test found zero AIPerfConfig YAMLs — check paths in this file."
    )

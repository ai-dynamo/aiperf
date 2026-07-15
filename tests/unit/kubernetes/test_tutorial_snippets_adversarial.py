# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial docs checks for tutorial and Kubernetes command snippets.

Focuses on:
- ``docs/tutorials`` and ``docs/kubernetes`` references that should point at
  checked-in files instead of stale branch-era paths;
- ``aiperf kube`` snippets naming subcommands that are registered on the
  current lazy-loaded kube CLI app;
- Kubernetes docs avoiding stale local-install claims for AIPerf itself;
- full ``AIPerfJob`` / ``AIPerfSweep`` YAML snippets validating through the
  same local kind-dispatch path as ``aiperf kube validate``;
- Fern ``docs/index.yml`` navigation paths resolving to real documentation.

Out of scope: generic Kubernetes resources, deliberately incomplete CR fragments,
and anchor validation inside existing Markdown files; broader docs-link rendering
belongs with the Fern end-to-end suite.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import yaml
from pytest import param

from aiperf.cli_commands.kube._app import app as kube_app
from aiperf.kubernetes.validate import KIND_AIPERFJOB, KIND_AIPERFSWEEP, validate_file

# ============================================================================
# Helpers
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS_ROOT = _REPO_ROOT / "docs"
_TARGET_DOC_ROOTS = (_DOCS_ROOT / "tutorials", _DOCS_ROOT / "kubernetes")
_DOCS_INDEX = _DOCS_ROOT / "index.yml"

_MARKDOWN_LINK_RE = re.compile(r"!?(?<!\\)\[[^\]]*\]\((?P<target>[^)]+)\)")
_YAML_FENCE_RE = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
_AIPERF_KUBE_RE = re.compile(r"\baiperf\s+kube\s+(?P<command>[a-z][a-z0-9_-]*)\b")
_AIPERF_INSTALL_RE = re.compile(
    r"\b(?:pip|uv\s+pip)\s+install\s+(?:--[^\s]+\s+)*(?:['\"])?aiperf(?:['\"])?(?:\s|$)"
)
_AIPERF_KINDS = {KIND_AIPERFJOB, KIND_AIPERFSWEEP}


@dataclass(frozen=True, slots=True)
class DocsSnippet:
    """Documentation snippet or reference with source location for failures."""

    path: Path
    line: int
    text: str

    @property
    def source_id(self) -> str:
        """Return a stable pytest id that points at the source line."""
        return f"{self.path.relative_to(_REPO_ROOT)}:{self.line}"


@dataclass(frozen=True, slots=True)
class DocsYamlSnippet(DocsSnippet):
    """Parsed YAML fence with source location for validation failures."""

    doc: dict[str, object]


def _target_markdown_sources() -> list[Path]:
    sources: list[Path] = []
    for root in _TARGET_DOC_ROOTS:
        sources.extend(sorted(root.glob("*.md")))
    return sorted(sources)


def _line_for_offset(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def _strip_link_target(target: str) -> str:
    return target.strip().split("#", 1)[0].split("?", 1)[0]


def _is_external_or_anchor(target: str) -> bool:
    stripped = target.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith(("http://", "https://", "mailto:"))
    )


def _mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def _load_yaml_mappings(body: str, *, source_id: str) -> list[dict[str, object]]:
    try:
        loaded_docs = list(yaml.safe_load_all(body))
    except yaml.YAMLError as exc:  # pragma: no cover - assertion context only
        pytest.fail(f"{source_id}: YAML snippet is not parseable: {exc}")
    return [doc for loaded in loaded_docs if (doc := _mapping(loaded)) is not None]


def _yaml_snippets() -> list[DocsYamlSnippet]:
    snippets: list[DocsYamlSnippet] = []
    for path in _target_markdown_sources():
        text = path.read_text()
        for match in _YAML_FENCE_RE.finditer(text):
            info = match.group("info").strip().lower()
            if "yaml" not in info and "yml" not in info:
                continue
            line = _line_for_offset(text, match.start())
            body = match.group("body")
            if "kind: AIPerf" not in body and "aiperf.nvidia.com" not in body:
                continue
            source_id = f"{path.relative_to(_REPO_ROOT)}:{line}"
            for doc in _load_yaml_mappings(body, source_id=source_id):
                snippets.append(
                    DocsYamlSnippet(path=path, line=line, text=body, doc=doc)
                )
    return snippets


def _full_aiperf_cr_snippets() -> list[DocsYamlSnippet]:
    snippets: list[DocsYamlSnippet] = []
    for snippet in _yaml_snippets():
        if snippet.doc.get("kind") not in _AIPERF_KINDS:
            continue
        if _mapping(snippet.doc.get("metadata")) is None:
            continue
        if _mapping(snippet.doc.get("spec")) is None:
            continue
        snippets.append(snippet)
    return snippets


def _snippet_file(tmp_path: Path, snippet: DocsYamlSnippet) -> Path:
    safe_stem = re.sub(r"[^a-z0-9-]", "-", snippet.path.stem.lower())
    path = tmp_path / f"{safe_stem}-{snippet.line}.yaml"
    path.write_text(snippet.text)
    return path


def _registered_kube_commands() -> set[str]:
    return {command for command in kube_app._commands if not command.startswith("-")}


def _aiperf_kube_command_snippets() -> list[DocsSnippet]:
    snippets: list[DocsSnippet] = []
    for path in _target_markdown_sources():
        text = path.read_text()
        for match in _AIPERF_KUBE_RE.finditer(text):
            snippets.append(
                DocsSnippet(
                    path=path,
                    line=_line_for_offset(text, match.start()),
                    text=match.group(0),
                )
            )
    return snippets


def _without_inline_code(text: str) -> str:
    return re.sub(r"`[^`]*`", lambda match: " " * len(match.group(0)), text)


def _local_markdown_links() -> list[tuple[Path, int, str]]:
    links: list[tuple[Path, int, str]] = []
    for path in _target_markdown_sources():
        text = path.read_text()
        link_text = _without_inline_code(text)
        for match in _MARKDOWN_LINK_RE.finditer(link_text):
            target = match.group("target")
            if _is_external_or_anchor(target):
                continue
            stripped = _strip_link_target(target)
            if not stripped:
                continue
            links.append((path, _line_for_offset(text, match.start()), stripped))
    return links


def _index_paths(value: object) -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        path = value.get("path")
        if isinstance(path, str):
            paths.append(path)
        for child in value.get("contents", []):
            paths.extend(_index_paths(child))
        return paths
    if isinstance(value, list):
        paths = []
        for item in value:
            paths.extend(_index_paths(item))
        return paths
    return []


_AIPERF_CR_SNIPPETS = _full_aiperf_cr_snippets()
_AIPERF_KUBE_COMMAND_SNIPPETS = _aiperf_kube_command_snippets()


# ============================================================================
# Discovery sanity
# ============================================================================


def test_tutorial_snippets_discovery_finds_kube_commands_and_crs() -> None:
    assert _AIPERF_KUBE_COMMAND_SNIPPETS, (
        "expected docs to contain aiperf kube snippets"
    )
    assert _AIPERF_CR_SNIPPETS, (
        "expected docs to contain full AIPerfJob/AIPerfSweep snippets"
    )


# ============================================================================
# Kubernetes command snippets
# ============================================================================


@pytest.mark.parametrize(
    "snippet",
    [param(snippet, id=snippet.source_id) for snippet in _AIPERF_KUBE_COMMAND_SNIPPETS],
)  # fmt: skip
def test_tutorial_snippets_aiperf_kube_commands_are_registered(
    snippet: DocsSnippet,
) -> None:
    command = shlex.split(snippet.text)[2]

    assert command in _registered_kube_commands(), (
        f"{snippet.source_id}: docs mention unregistered kube command "
        f"{snippet.text!r}; registered={sorted(_registered_kube_commands())}"
    )


# ============================================================================
# Kubernetes YAML snippets
# ============================================================================


@pytest.mark.parametrize(
    "snippet",
    [param(snippet, id=snippet.source_id) for snippet in _AIPERF_CR_SNIPPETS],
)  # fmt: skip
def test_tutorial_snippets_full_aiperf_cr_yaml_validates(
    snippet: DocsYamlSnippet, tmp_path: Path
) -> None:
    path = _snippet_file(tmp_path, snippet)

    result = validate_file(path, strict=True)

    assert result.passed, (
        f"{snippet.source_id}: errors={result.errors}; warnings={result.warnings}"
    )


# ============================================================================
# Docs references and stale installation claims
# ============================================================================


def test_tutorial_snippets_local_markdown_links_point_to_existing_files() -> None:
    missing: list[str] = []
    for path, line, target in _local_markdown_links():
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(_REPO_ROOT)
        except ValueError:
            missing.append(
                f"{path.relative_to(_REPO_ROOT)}:{line} -> {target} escapes repo root"
            )
            continue
        if not resolved.exists():
            missing.append(f"{path.relative_to(_REPO_ROOT)}:{line} -> {target}")

    assert not missing, "\n".join(missing)


def test_tutorial_snippets_docs_index_paths_point_to_existing_files() -> None:
    index = yaml.safe_load(_DOCS_INDEX.read_text())
    paths = _index_paths(index.get("navigation", []))

    missing = [path for path in paths if not (_DOCS_ROOT / path).is_file()]

    assert not missing


def test_tutorial_snippets_kubernetes_docs_do_not_claim_stale_aiperf_install_commands() -> (
    None
):
    stale: list[str] = []
    for path in sorted((_DOCS_ROOT / "kubernetes").glob("*.md")):
        text = path.read_text()
        for match in _AIPERF_INSTALL_RE.finditer(text):
            stale.append(
                f"{path.relative_to(_REPO_ROOT)}:{_line_for_offset(text, match.start())}: "
                f"replace {match.group(0).strip()!r} with the current install guidance"
            )

    assert not stale, "\n".join(stale)

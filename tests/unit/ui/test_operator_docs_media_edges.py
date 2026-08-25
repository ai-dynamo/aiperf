# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static checks for operator dashboard documentation media references."""

from __future__ import annotations

import re
from pathlib import Path
from subprocess import run

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS_ROOT = _REPO_ROOT / "docs"
_REMOVED_ARTIFACT_RE = re.compile(
    r"dev/ui-verify/shots-|api-dashboard-v2\.png", re.IGNORECASE
)

_MARKDOWN_REF_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
_STALE_OPERATOR_UI_IMAGE_RE = re.compile(
    r"operator-ui[^\s)\"']*\.(?:png|jpg|jpeg|webp)", re.IGNORECASE
)


def _markdown_sources() -> list[Path]:
    return sorted(_DOCS_ROOT.rglob("*.md"))


def _documentation_sources() -> list[Path]:
    return [*_markdown_sources(), _REPO_ROOT / "README.md"]


def _doc_text(path: Path) -> str:
    return path.read_text()


def _strip_anchor_or_query(target: str) -> str:
    return target.split("#", 1)[0].split("?", 1)[0]


def _media_image_refs(path: Path) -> list[str]:
    text = _doc_text(path)
    refs = [match.group(1).strip() for match in _MARKDOWN_REF_RE.finditer(text)]
    refs.extend(match.group(1).strip() for match in _HTML_IMAGE_RE.finditer(text))
    return [ref for ref in refs if "media/images/" in ref]


def _resolved_doc_ref(path: Path, target: str) -> Path:
    return (path.parent / _strip_anchor_or_query(target)).resolve()


def _tracked_files(pathspec: str) -> list[str]:
    result = run(
        ["git", "ls-files", "--", pathspec],
        capture_output=True,
        check=True,
        cwd=_REPO_ROOT,
        text=True,
    )
    return result.stdout.splitlines()


def test_docs_media_image_references_point_to_existing_files() -> None:
    missing: list[str] = []
    for path in _markdown_sources():
        for target in _media_image_refs(path):
            resolved = _resolved_doc_ref(path, target)
            if not resolved.is_file():
                missing.append(f"{path.relative_to(_REPO_ROOT)} -> {target}")

    assert not missing


def test_dashboard_doc_uses_no_stale_operator_ui_screenshot_paths() -> None:
    dashboard_doc = _DOCS_ROOT / "kubernetes" / "dashboard-ui.md"
    stale_refs = sorted(
        set(_STALE_OPERATOR_UI_IMAGE_RE.findall(_doc_text(dashboard_doc)))
    )

    assert not stale_refs


def test_review_capture_outputs_are_not_tracked() -> None:
    tracked_outputs = _tracked_files("dev/ui-verify/shots-*")
    tracked_outputs.extend(_tracked_files("docs/media/images/api-dashboard-v2.png"))

    assert not tracked_outputs


def test_documentation_does_not_reference_removed_review_artifacts() -> None:
    artifact_references = [
        f"{path.relative_to(_REPO_ROOT)}: {match.group(0)}"
        for path in _documentation_sources()
        for match in _REMOVED_ARTIFACT_RE.finditer(_doc_text(path))
    ]

    assert not artifact_references

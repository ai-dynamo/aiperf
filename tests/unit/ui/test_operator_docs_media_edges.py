# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static checks for operator dashboard documentation media references."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCS_ROOT = _REPO_ROOT / "docs"
_DASHBOARD_DOC = _DOCS_ROOT / "kubernetes" / "dashboard-ui.md"
_MEDIA_IMAGES = _DOCS_ROOT / "media" / "images"
_LATEST_DASHBOARD_IMAGE = _MEDIA_IMAGES / "api-dashboard-v2.png"

_MARKDOWN_REF_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
_STALE_OPERATOR_UI_IMAGE_RE = re.compile(r"operator-ui[^\s)\"']*\.(?:png|jpg|jpeg|webp)", re.IGNORECASE)


def _markdown_sources() -> list[Path]:
    return sorted(_DOCS_ROOT.rglob("*.md"))


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


def test_docs_media_image_references_point_to_existing_files() -> None:
    missing: list[str] = []
    for path in _markdown_sources():
        for target in _media_image_refs(path):
            resolved = _resolved_doc_ref(path, target)
            if not resolved.is_file():
                missing.append(f"{path.relative_to(_REPO_ROOT)} -> {target}")

    assert not missing


def test_dashboard_doc_uses_no_stale_operator_ui_screenshot_paths() -> None:
    stale_refs = sorted(set(_STALE_OPERATOR_UI_IMAGE_RE.findall(_doc_text(_DASHBOARD_DOC))))

    assert not stale_refs


def test_dashboard_doc_screenshot_refs_include_latest_dashboard_image() -> None:
    dashboard_refs = _media_image_refs(_DASHBOARD_DOC)
    if not dashboard_refs:
        return

    resolved_refs = {_resolved_doc_ref(_DASHBOARD_DOC, ref) for ref in dashboard_refs}

    assert _LATEST_DASHBOARD_IMAGE in resolved_refs


def test_latest_dashboard_image_asset_exists() -> None:
    assert _LATEST_DASHBOARD_IMAGE.is_file()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static accessibility checks for operator UI markup."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_INTERACTIVE_EXTENSIONS = {".html", ".js"}
_TAG_BLOCK_RE = re.compile(
    r"<(?P<tag>button|a)\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</(?P=tag)>"
)
_TABLIST_RE = re.compile(
    r"<(?P<tag>[^\s>/]+)\b(?P<attrs>[^>]*)\brole=[\"']tablist[\"'][^>]*>(?P<body>[\s\S]*?)</(?P=tag)>"
)
_PROGRESSBAR_TAG_RE = re.compile(
    r"<(?P<tag>[^\s>/]+)\b(?P<attrs>[^>]*\brole=[\"']progressbar[\"'][^>]*)(?:/>|>[\s\S]*?</(?P=tag)>)"
)
_ONCLICK_NAV_RE = re.compile(
    r"<(?P<tag>div|span|li|tr|td)\b(?P<attrs>[^>]*)\bonclick=\$?\{(?P<handler>[^}]*(?:navigate|window\.location)[^}]*)\}",
    re.IGNORECASE,
)


_KNOWN_NON_INTERACTIVE_NAV_DEBT = {
    # Backdrop clicks dismiss modal overlays; they are not page navigation.
    ("components/artifacts-card.js", "div"),
    ("pages/job-detail.js", "div"),
    # Existing page-level navigation debt outside this targeted log-strip regression.
    ("pages/history.js", "span"),
    ("pages/sweep-detail.js", "tr"),
}


def _ui_files() -> list[Path]:
    return sorted(
        path
        for path in _UI_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in _INTERACTIVE_EXTENSIONS
        and "vendor" not in path.parts
    )


def _source(path: Path) -> str:
    return path.read_text()


def _relative(path: Path) -> str:
    return path.relative_to(_UI_ROOT).as_posix()


def _line_number(src: str, offset: int) -> int:
    return src[:offset].count("\n") + 1


def _literal_or_dynamic_text(body: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", body)
    with_dynamic_text = re.sub(r"\$\{[^}]+\}", " dynamic ", without_tags)
    return re.sub(r"\s+", " ", with_dynamic_text).strip()


def _has_accessible_name(attrs: str, body: str) -> bool:
    return bool(
        re.search(r"\b(?:aria-label|title)\s*=", attrs)
        or _literal_or_dynamic_text(body)
    )


def _has_attr(attrs: str, name: str) -> bool:
    return bool(re.search(rf"\b{name}\s*=", attrs))


def _interactive_elements_missing_names() -> list[str]:
    missing: list[str] = []
    for path in _ui_files():
        src = _source(path)
        for match in _TAG_BLOCK_RE.finditer(src):
            tag = match.group("tag")
            attrs = match.group("attrs")
            body = match.group("body")
            if tag == "a" and not (
                'target="_blank"' in attrs or "target='_blank'" in attrs
            ):
                continue
            if _has_accessible_name(attrs, body):
                continue
            missing.append(
                f"{_relative(path)}:{_line_number(src, match.start())} <{tag}>"
            )
    return missing


def _tablist_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = _source(path)
        for tablist in _TABLIST_RE.finditer(src):
            body = tablist.group("body")
            if 'role="tab"' not in body and "role='tab'" not in body:
                violations.append(
                    f"{_relative(path)}:{_line_number(src, tablist.start())} tablist has no role=tab descendants"
                )
                continue
            for tab in re.finditer(
                r"<button\b(?P<attrs>[^>]*\brole=[\"']tab[\"'][^>]*)>", body
            ):
                if not _has_attr(tab.group("attrs"), "aria-selected"):
                    line = _line_number(src, tablist.start() + tab.start())
                    violations.append(
                        f"{_relative(path)}:{line} role=tab missing aria-selected"
                    )
    return violations


def _progressbar_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = _source(path)
        for progressbar in _PROGRESSBAR_TAG_RE.finditer(src):
            attrs = progressbar.group("attrs")
            missing = [
                name
                for name in ("aria-valuenow", "aria-valuemin", "aria-valuemax")
                if not _has_attr(attrs, name)
            ]
            if missing:
                violations.append(
                    f"{_relative(path)}:{_line_number(src, progressbar.start())} missing {', '.join(missing)}"
                )
    return violations


def _onclick_only_nav_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = _source(path)
        rel = _relative(path)
        for match in _ONCLICK_NAV_RE.finditer(src):
            tag = match.group("tag").lower()
            if (rel, tag) in _KNOWN_NON_INTERACTIVE_NAV_DEBT:
                continue
            violations.append(
                f"{rel}:{_line_number(src, match.start())} <{tag}> uses onclick navigation"
            )
    return violations


def test_interactive_buttons_and_external_links_have_accessible_names() -> None:
    assert _interactive_elements_missing_names() == []


def test_tablists_expose_tabs_with_selected_state() -> None:
    assert _tablist_violations() == []


def test_progressbars_expose_value_bounds() -> None:
    assert _progressbar_violations() == []


def test_critical_navigation_does_not_use_onclick_only_non_interactive_elements() -> (
    None
):
    assert _onclick_only_nav_violations() == []

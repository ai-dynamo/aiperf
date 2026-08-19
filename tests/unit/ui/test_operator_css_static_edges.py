# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static CSS edge checks for operator UI regression-prone classes."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_STYLE_CSS = _UI_ROOT / "style.css"
_COMPONENTS = _UI_ROOT / "components"


_CLASS_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_CSS_SELECTOR_RE = re.compile(r"(^|})\s*([^{}@][^{}]*)\{", re.MULTILINE)
_CSS_CLASS_RE = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)")


def _source(path: Path) -> str:
    return path.read_text()


def _css() -> str:
    return _source(_STYLE_CSS)


def _css_classes() -> set[str]:
    return set(_CSS_CLASS_RE.findall(_css()))


def _literal_class_tokens(src: str) -> set[str]:
    """Return class tokens from literal class attributes and class string fragments."""
    tokens: set[str] = set()
    for quoted in re.findall(r"class(?:Name)?=\$?\{?(['\"])(.*?)\1", src):
        tokens.update(_CLASS_TOKEN_RE.findall(quoted[1]))
    for quoted in re.findall(r"(['\"])([^'\"]*[A-Za-z][A-Za-z0-9_-]+[^'\"]*)\1", src):
        text = quoted[1]
        if (
            "class" in text
            or "log-strip" in text
            or "diagnostics-" in text
            or "job-table" in text
        ):
            tokens.update(_CLASS_TOKEN_RE.findall(text))
    return tokens


def _rule_bodies_by_selector(css: str) -> dict[str, list[str]]:
    bodies: dict[str, list[str]] = defaultdict(list)
    for match in _CSS_SELECTOR_RE.finditer(css):
        selector = " ".join(match.group(2).split())
        if selector.startswith(("from ", "to ", "0%", "50%", "70%", "100%")):
            continue
        start = match.end()
        end = css.find("}", start)
        if end == -1:
            continue
        body = css[start:end]
        normalized = ";".join(line.strip() for line in body.split(";") if line.strip())
        bodies[selector].append(normalized)
    return bodies


def _assert_css_covers(component: Path, expected_classes: set[str]) -> None:
    src = _source(component)
    css_classes = _css_classes()

    missing_from_source = sorted(cls for cls in expected_classes if cls not in src)
    assert not missing_from_source

    missing_from_css = sorted(expected_classes - css_classes)
    assert not missing_from_css


def _declarations(body: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for part in body.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        declarations[key.strip()] = value.strip()
    return declarations


def _has_conflicting_full_blocks(selector_bodies: list[str]) -> bool:
    parsed = [_declarations(body) for body in selector_bodies]
    for idx, left in enumerate(parsed):
        for right in parsed[idx + 1 :]:
            shared = set(left) & set(right)
            if len(left) < 3 or len(right) < 3:
                continue
            if any(left[prop] != right[prop] for prop in shared):
                return True
    return False


def test_log_strip_component_classes_have_css_rules() -> None:
    _assert_css_covers(
        _COMPONENTS / "log-strip.js",
        {
            "log-strip",
            "log-strip--collapsed",
            "log-strip-head",
            "log-strip-title",
            "log-strip-filters",
            "log-strip-filter",
            "log-strip-filter--active",
            "log-strip-filter-count",
            "log-strip-toggle",
            "log-strip-body",
            "log-strip-entry",
            "log-strip-entry--warn",
            "log-strip-entry--error",
            "log-strip-cat",
        },
    )
    src = _source(_COMPONENTS / "log-strip.js")
    css_classes = _css_classes()
    assert "cat: 'phase'" in src
    assert "cat: 'worker'" in src
    assert {"log-strip-cat--phase", "log-strip-cat--worker"} <= css_classes


def test_diagnostics_tabs_classes_have_css_rules() -> None:
    _assert_css_covers(
        _COMPONENTS / "diagnostics-panel.js",
        {
            "diag-tabs",
            "diag-tab",
            "diag-tab--active",
            "diag-tab-count",
        },
    )


def test_table_sticky_classes_keep_scroll_and_sticky_header_rules() -> None:
    css = _css()

    assert ".job-table-wrapper" in css
    assert "overflow-x: auto" in css
    assert "overflow-y: auto" in css
    assert ".job-table thead th" in css
    assert "position: sticky" in css
    assert "top: 0" in css
    assert "z-index: 2" in css
    assert ".metrics-table__group-head" in css
    assert "z-index: 1" in css


def test_critical_selectors_do_not_have_conflicting_duplicate_blocks() -> None:
    bodies = _rule_bodies_by_selector(_css())
    critical_selectors = {
        ".job-table-wrapper",
        ".job-table thead th",
        ".log-strip",
        ".log-strip--collapsed",
        ".log-strip-body",
        ".diagnostics-drawer",
        ".diagnostics-drawer__backdrop",
        ".diagnostics-drawer__body",
        ".metrics-table__group-head",
    }

    conflicting = {
        selector: selector_bodies
        for selector, selector_bodies in bodies.items()
        if selector in critical_selectors
        and _has_conflicting_full_blocks(selector_bodies)
    }

    assert conflicting == {}

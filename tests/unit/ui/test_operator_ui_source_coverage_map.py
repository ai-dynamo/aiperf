# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static source-to-test coverage map for operator UI JavaScript modules."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_TEST_ROOT = _REPO_ROOT / "tests" / "unit" / "ui"
_THIS_FILE = Path(__file__).resolve()

_TRIVIAL_OR_EXTERNAL_MODULES_WITH_REASON: dict[str, str] = {}

_UNCOVERED_MODULES_WITH_REASON = {
    "components/cluster-stats-banner.js": "dashboard presentation component currently covered only through composed page tests",
    "components/diagnostics-conditions-tab.js": "thin diagnostics tab composition around the covered conditions renderer",
    "components/epoch-selector.js": "selector UI wiring lacks direct static-unit coverage after source split",
    "components/spinner.js": "loading indicator helpers lack direct static fixture coverage",
    "components/sweep-live-trial-board.js": "live trial board presentation is exercised through composed sweep-detail page tests; lacks a direct static fixture test",
    "components/sweep-winner-summary.js": "winner summary presentation is exercised through composed sweep-detail page tests; lacks a direct static fixture test",
    "lib/theme-switch.js": "theme switch DOM wiring lacks direct static-unit coverage",
    "lib/theme.js": "theme persistence helpers lack direct static-unit coverage",
    "pages/compare-epochs.js": "compare epoch page orchestration lacks direct static-unit coverage",
}

_GENERIC_STEMS = {
    "app",
    "format",
    "helpers",
    "history",
    "index",
    "jobs",
    "launch",
    "state",
    "time",
}


def _source_modules() -> list[Path]:
    return sorted(_UI_ROOT.rglob("*.js"))


def _source_relpath(path: Path) -> str:
    return path.relative_to(_UI_ROOT).as_posix()


def _test_sources() -> str:
    return "\n".join(
        path.read_text()
        for path in sorted(_TEST_ROOT.glob("test_*.py"))
        if path.resolve() != _THIS_FILE
    )


def _reference_tokens(path: Path) -> set[str]:
    relpath = _source_relpath(path)
    relpath_without_suffix = relpath.removesuffix(path.suffix)
    tokens = {relpath, f"./{relpath}", path.name, relpath_without_suffix}
    if path.stem not in _GENERIC_STEMS:
        tokens.add(path.stem)
    return tokens


def test_operator_ui_source_files_have_static_test_map_entries() -> None:
    """Keep every local UI module either referenced by tests or deliberately allowlisted."""
    test_sources = _test_sources()
    source_relpaths = {_source_relpath(path) for path in _source_modules()}
    ignored_relpaths = set(_TRIVIAL_OR_EXTERNAL_MODULES_WITH_REASON)
    allowlisted_relpaths = set(_UNCOVERED_MODULES_WITH_REASON)

    unknown_allowlist_entries = sorted(
        (ignored_relpaths | allowlisted_relpaths) - source_relpaths
    )
    empty_reasons = sorted(
        relpath
        for relpath, reason in {
            **_TRIVIAL_OR_EXTERNAL_MODULES_WITH_REASON,
            **_UNCOVERED_MODULES_WITH_REASON,
        }.items()
        if not reason.strip()
    )
    uncovered = sorted(
        relpath
        for path in _source_modules()
        if (relpath := _source_relpath(path)) not in ignored_relpaths
        and relpath not in allowlisted_relpaths
        and not any(token in test_sources for token in _reference_tokens(path))
    )

    assert unknown_allowlist_entries == []
    assert empty_reasons == []
    assert uncovered == []

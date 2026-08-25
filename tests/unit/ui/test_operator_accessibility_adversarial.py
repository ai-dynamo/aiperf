# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial static accessibility checks for operator UI markup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_INTERACTIVE_EXTENSIONS = {".html", ".js"}

_OPEN_TAG_RE = re.compile(r"<(?P<tag>[a-zA-Z][\w:-]*)\b(?P<attrs>[^>]*)>")
_EXTERNAL_LINK_RE = re.compile(
    r"<a\b(?P<attrs>[^>]*\btarget=[\"']_blank[\"'][^>]*)>", re.IGNORECASE
)
_ROLE_DIALOG_RE = re.compile(
    r"<(?P<tag>[a-zA-Z][\w:-]*)\b(?P<attrs>[^>]*\brole=[\"']dialog[\"'][^>]*)>",
    re.IGNORECASE,
)
_BUTTON_BLOCK_RE = re.compile(
    r"<button\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</button>", re.IGNORECASE
)
_TABLE_CLICK_TARGET_RE = re.compile(
    r"<(?P<tag>tr|th)\b(?P<attrs>[^>]*\bonclick=\$?\{[^>]*)>", re.IGNORECASE
)
_ACTIVE_CONTROL_RE = re.compile(
    r"<(?P<tag>button|a|span)\b(?P<attrs>[^>]*\bactive\b[^>]*)>", re.IGNORECASE
)


@dataclass(frozen=True)
class TagMatch:
    rel: str
    line: int
    tag: str
    attrs: str


def _ui_files() -> list[Path]:
    return sorted(
        path
        for path in _UI_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in _INTERACTIVE_EXTENSIONS
        and "vendor" not in path.parts
    )


def _relative(path: Path) -> str:
    return path.relative_to(_UI_ROOT).as_posix()


def _line_number(src: str, offset: int) -> int:
    return src[:offset].count("\n") + 1


def _has_attr(attrs: str, name: str) -> bool:
    return bool(re.search(rf"\b{name}\s*=", attrs))


def _attr_value(attrs: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*([\"'])(?P<value>.*?)\1", attrs, re.DOTALL)
    if match:
        return match.group("value")
    return None


def _iter_open_tags(pattern: re.Pattern[str]) -> list[TagMatch]:
    matches: list[TagMatch] = []
    for path in _ui_files():
        src = path.read_text()
        rel = _relative(path)
        for match in pattern.finditer(src):
            matches.append(
                TagMatch(
                    rel=rel,
                    line=_line_number(src, match.start()),
                    tag=match.groupdict().get("tag", "").lower(),
                    attrs=match.group("attrs"),
                )
            )
    return matches


def _keyboard_reachable_table_click_targets() -> list[str]:
    violations: list[str] = []
    for match in _iter_open_tags(_TABLE_CLICK_TARGET_RE):
        attrs = match.attrs
        has_focus_target = _has_attr(attrs, "tabindex") or _has_attr(attrs, "tabIndex")
        has_key_handler = (
            _has_attr(attrs, "onkeydown")
            or _has_attr(attrs, "onKeyDown")
            or _has_attr(attrs, "onkeyup")
            or _has_attr(attrs, "onKeyUp")
        )
        has_interactive_role = bool(
            re.search(r"\brole\s*=\s*[\"'](?:button|link|row|columnheader)[\"']", attrs)
        )
        if not (has_focus_target and has_key_handler and has_interactive_role):
            missing = []
            if not has_focus_target:
                missing.append("tabindex")
            if not has_key_handler:
                missing.append("keyboard handler")
            if not has_interactive_role:
                missing.append("interactive role")
            violations.append(
                f"{match.rel}:{match.line} <{match.tag}> clickable table target missing {', '.join(missing)}"
            )
    return violations


def _active_control_state_violations() -> list[str]:
    violations: list[str] = []
    for match in _iter_open_tags(_ACTIVE_CONTROL_RE):
        attrs = match.attrs
        if any(
            _has_attr(attrs, name)
            for name in ("aria-current", "aria-selected", "aria-pressed")
        ):
            continue
        violations.append(
            f"{match.rel}:{match.line} <{match.tag}> active state has no aria-current/selected/pressed"
        )
    return violations


def _dialog_focus_management_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = path.read_text()
        rel = _relative(path)
        for dialog in _ROLE_DIALOG_RE.finditer(src):
            attrs = dialog.group("attrs")
            line = _line_number(src, dialog.start())
            if not (
                _has_attr(attrs, "aria-label") or _has_attr(attrs, "aria-labelledby")
            ):
                violations.append(f"{rel}:{line} dialog has no accessible name")
            if not _has_attr(attrs, "aria-modal"):
                violations.append(f"{rel}:{line} dialog missing aria-modal declaration")
            local_context = src[max(0, dialog.start() - 1500) : dialog.end() + 1500]
            # The invariant is that Escape dismisses the dialog; which callback
            # does the dismissing is the component's business.
            has_escape_close = "Escape" in local_context and any(
                token in local_context
                for token in ("onClose", "onCancel", "setOpen(false)")
            )
            has_focus_management = bool(
                re.search(
                    r"\.focus\(|focusout|relatedTarget|querySelectorAll\('\[role=",
                    local_context,
                )
            )
            if not has_escape_close:
                violations.append(
                    f"{rel}:{line} dialog has no nearby Escape close handler"
                )
            if not has_focus_management:
                violations.append(f"{rel}:{line} dialog has no nearby focus management")
    return violations


def _button_type_violations() -> list[str]:
    violations: list[str] = []
    for match in _iter_open_tags(_BUTTON_BLOCK_RE):
        if not _has_attr(match.attrs, "type"):
            violations.append(
                f"{match.rel}:{match.line} <button> missing explicit type"
            )
    return violations


def _aria_disabled_click_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = path.read_text()
        rel = _relative(path)
        for button in _BUTTON_BLOCK_RE.finditer(src):
            attrs = button.group("attrs")
            if not _has_attr(attrs, "aria-disabled") or not re.search(
                r"\bon(?:C|c)lick\s*=", attrs
            ):
                continue
            handler_context = attrs + button.group("body")
            guarded = bool(
                re.search(
                    r"if\s*\([^)]*(?:disabled|inert)[^)]*\)\s*return",
                    handler_context,
                    re.IGNORECASE,
                )
            )
            if not guarded:
                violations.append(
                    f"{rel}:{_line_number(src, button.start())} aria-disabled button keeps an unguarded click handler"
                )
    return violations


def _external_link_violations() -> list[str]:
    violations: list[str] = []
    for match in _iter_open_tags(_EXTERNAL_LINK_RE):
        rel_value = _attr_value(match.attrs, "rel")
        rel_tokens = set((rel_value or "").lower().split())
        missing = {"noopener", "noreferrer"} - rel_tokens
        if missing:
            violations.append(
                f"{match.rel}:{match.line} external link rel missing {', '.join(sorted(missing))}"
            )
    return violations


def test_clickable_table_rows_and_headers_are_keyboard_reachable() -> None:
    assert _keyboard_reachable_table_click_targets() == []


def test_visible_active_controls_expose_matching_aria_state() -> None:
    assert _active_control_state_violations() == []


def test_dialog_drawers_declare_modal_state_and_manage_focus() -> None:
    assert _dialog_focus_management_violations() == []


def test_buttons_have_explicit_type_to_avoid_accidental_submit_behavior() -> None:
    assert _button_type_violations() == []


def test_aria_disabled_buttons_guard_click_handlers() -> None:
    assert _aria_disabled_click_violations() == []


def test_external_blank_links_prevent_opener_and_referrer_leaks() -> None:
    assert _external_link_violations() == []

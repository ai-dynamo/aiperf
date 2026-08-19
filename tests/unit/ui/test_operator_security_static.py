# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static adversarial security checks for the operator UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_UI_EXTENSIONS = {".html", ".js"}

_RAW_HTML_SINK_RE = re.compile(
    r"(?:\.(?:innerHTML|outerHTML)\s*=|\binsertAdjacentHTML\s*\(|\bdangerouslySetInnerHTML\s*=)",
)
_DYNAMIC_CODE_RE = re.compile(r"\b(?:eval\s*\(|new\s+Function\b)")
_TARGET_BLANK_TAG_RE = re.compile(
    r"<(?P<tag>a|area|form)\b(?P<attrs>[^>]*\btarget\s*=\s*(?:[\"']_blank[\"']|\$\{[^}]+\})[^>]*)>",
    re.IGNORECASE,
)
_STORAGE_SET_RE = re.compile(
    r"\b(?P<store>localStorage|sessionStorage)\.setItem\s*\((?P<args>[\s\S]*?)\)",
    re.MULTILINE,
)
_URL_TEMPLATE_RE = re.compile(
    r"`(?P<template>[^`]*\$\{[^`]*?(?:/api/v1|\$\{BASE\})[^`]*)`|`(?P<template_after>[^`]*(?:/api/v1|\$\{BASE\})[^`]*\$\{[^`]*)`"
)
_API_INTERPOLATION_RE = re.compile(r"\$\{(?P<expr>[^}]+)\}")
_SENSITIVE_STORAGE_RE = re.compile(
    r"\b(?:api[_-]?key|authorization|bearer|client[_-]?secret|kubeconfig|password|secret|token)\b",
    re.IGNORECASE,
)
_IMPORTMAP_RE = re.compile(
    r"<script\s+type=[\"']importmap[\"']\s*>\s*(?P<json>.*?)\s*</script>",
    re.DOTALL,
)
_REMOTE_SCRIPT_SRC_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*[\"'](?:https?:)?//",
    re.IGNORECASE,
)
_DATA_SCRIPT_SRC_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*[\"']data:",
    re.IGNORECASE,
)
_UNQUOTED_SCRIPT_SRC_RE = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*(?![\"'])",
    re.IGNORECASE,
)
_MODULE_FROM_IMPORT_RE = re.compile(
    r"\b(?:import|export)\b(?:[\s\S]*?)\bfrom\s*([\"'])(?P<specifier>[^\"']+)\1",
    re.MULTILINE,
)
_SIDE_EFFECT_IMPORT_RE = re.compile(
    r"\bimport\s*([\"'])(?P<specifier>[^\"']+)\1",
)
_DYNAMIC_IMPORT_RE = re.compile(
    r"\bimport\s*\(\s*([\"'])(?P<specifier>[^\"']+)\1\s*\)",
)


@dataclass(frozen=True)
class SourceMatch:
    rel: str
    line: int
    snippet: str


def _ui_files() -> list[Path]:
    return sorted(
        path
        for path in _UI_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in _UI_EXTENSIONS
        and "vendor" not in path.parts
    )


def _relative(path: Path) -> str:
    return path.relative_to(_UI_ROOT).as_posix()


def _line_number(src: str, offset: int) -> int:
    return src[:offset].count("\n") + 1


def _line_at(src: str, offset: int) -> str:
    start = src.rfind("\n", 0, offset) + 1
    end = src.find("\n", offset)
    if end == -1:
        end = len(src)
    return src[start:end].strip()


def _has_attr(attrs: str, name: str) -> bool:
    return bool(re.search(rf"\b{name}\s*=", attrs))


def _attr_value(attrs: str, name: str) -> str | None:
    match = re.search(rf"\b{name}\s*=\s*([\"'])(?P<value>.*?)\1", attrs, re.DOTALL)
    if match:
        return match.group("value")
    return None


def _source_matches(pattern: re.Pattern[str]) -> list[SourceMatch]:
    matches: list[SourceMatch] = []
    for path in _ui_files():
        src = path.read_text()
        rel = _relative(path)
        for match in pattern.finditer(src):
            matches.append(
                SourceMatch(
                    rel=rel,
                    line=_line_number(src, match.start()),
                    snippet=_line_at(src, match.start()),
                )
            )
    return matches


def _raw_html_sink_violations() -> list[str]:
    return [f"{m.rel}:{m.line} {m.snippet}" for m in _source_matches(_RAW_HTML_SINK_RE)]


def _dynamic_code_violations() -> list[str]:
    return [f"{m.rel}:{m.line} {m.snippet}" for m in _source_matches(_DYNAMIC_CODE_RE)]


def _target_blank_rel_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = path.read_text()
        rel_path = _relative(path)
        for match in _TARGET_BLANK_TAG_RE.finditer(src):
            attrs = match.group("attrs")
            rel_value = _attr_value(attrs, "rel")
            rel_tokens = set((rel_value or "").lower().split())
            missing = {"noopener", "noreferrer"} - rel_tokens
            if missing:
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel_path}:{line} <{match.group('tag')}> target=_blank rel missing {', '.join(sorted(missing))}"
                )
    return violations


# The one secret the console is allowed to hold: the operator bearer token the
# user types into the token modal. It is reviewed rather than redacted, so it is
# pinned by exact key and covered by its own assertions below -- tab-scoped
# storage only, and a clear path off it.
_REVIEWED_SESSION_SECRET_KEY = "aiperf.mutating.token"


def _storage_sensitive_payload_violations() -> list[str]:
    violations: list[str] = []
    for path in _ui_files():
        src = path.read_text()
        rel_path = _relative(path)
        for match in _STORAGE_SET_RE.finditer(src):
            context = src[max(0, match.start() - 500) : match.end() + 500]
            if not _SENSITIVE_STORAGE_RE.search(context):
                continue
            if "redactConfigForYaml" in context or "SENSITIVE_CONFIG_KEYS" in context:
                continue
            if (
                match.group("store") == "sessionStorage"
                and _REVIEWED_SESSION_SECRET_KEY in context
            ):
                continue
            line = _line_number(src, match.start())
            violations.append(
                f"{rel_path}:{line} {match.group('store')}.setItem stores sensitive-looking payload without nearby redaction"
            )
    return violations


def _reviewed_token_storage_violations() -> list[str]:
    """The bearer token may only live in tab-scoped storage, and must be
    erasable: `localStorage` survives the browser restart that the token's
    whole threat model assumes it does not."""
    violations: list[str] = []
    for path in _ui_files():
        src = path.read_text()
        rel_path = _relative(path)
        if _REVIEWED_SESSION_SECRET_KEY not in src:
            continue
        for match in _STORAGE_SET_RE.finditer(src):
            context = src[max(0, match.start() - 500) : match.end() + 500]
            if (
                _REVIEWED_SESSION_SECRET_KEY in context
                and match.group("store") != "sessionStorage"
            ):
                violations.append(
                    f"{rel_path}:{_line_number(src, match.start())} bearer token written to {match.group('store')}"
                )
        if "sessionStorage.removeItem" not in src:
            violations.append(f"{rel_path} stores the bearer token with no clear path")
    return violations


def _raw_api_url_interpolation_violations() -> list[str]:
    raw_part_re = re.compile(
        r"\b(?:container|epoch|filename|format|jobId|name|namespace|ns|pod|sweepName)\b"
    )
    violations: list[str] = []
    for path in _ui_files():
        src = path.read_text()
        rel_path = _relative(path)
        for match in _URL_TEMPLATE_RE.finditer(src):
            template = match.group("template") or match.group("template_after") or ""
            for interpolation in _API_INTERPOLATION_RE.finditer(template):
                expr = interpolation.group("expr").strip()
                if not raw_part_re.search(expr):
                    continue
                if "encodeURIComponent" in expr or expr.endswith("Seg"):
                    continue
                line = _line_number(src, match.start())
                violations.append(
                    f"{rel_path}:{line} API URL interpolates raw expression `${{{expr}}}`"
                )
    return violations


def test_ui_does_not_use_raw_html_or_dangerous_html_sinks() -> None:
    assert _raw_html_sink_violations() == []


def test_ui_does_not_evaluate_dynamic_code() -> None:
    assert _dynamic_code_violations() == []


def test_blank_target_links_prevent_opener_and_referrer_leaks() -> None:
    assert _target_blank_rel_violations() == []


def test_web_storage_does_not_persist_sensitive_payloads_without_redaction() -> None:
    assert _storage_sensitive_payload_violations() == []


def test_the_bearer_token_stays_tab_scoped_and_clearable() -> None:
    assert _reviewed_token_storage_violations() == []


def test_token_bearing_ui_loads_only_self_hosted_javascript() -> None:
    """Third-party JavaScript would execute with access to sessionStorage."""
    index_html = (_UI_ROOT / "index.html").read_text()
    assert _REMOTE_SCRIPT_SRC_RE.search(index_html) is None
    assert _DATA_SCRIPT_SRC_RE.search(index_html) is None
    assert _UNQUOTED_SCRIPT_SRC_RE.search(index_html) is None

    import_map_match = _IMPORTMAP_RE.search(index_html)
    assert import_map_match is not None
    import_map = json.loads(import_map_match.group("json"))
    imports = import_map["imports"]
    assert all(
        isinstance(path, str) and path.startswith("./vendor/")
        for path in imports.values()
    )
    assert all((_UI_ROOT / path).is_file() for path in imports.values())

    chart_asset = _UI_ROOT / "vendor" / "chart.umd.min.js"
    assert chart_asset.is_file()
    assert 'src="./vendor/chart.umd.min.js"' in index_html

    for module_path in (_UI_ROOT / "vendor").glob("*.mjs"):
        module_source = module_path.read_text()
        for import_re in (
            _MODULE_FROM_IMPORT_RE,
            _SIDE_EFFECT_IMPORT_RE,
            _DYNAMIC_IMPORT_RE,
        ):
            for match in import_re.finditer(module_source):
                specifier = match.group("specifier")
                assert specifier.startswith("./"), (module_path.name, specifier)
                assert (module_path.parent / specifier).is_file(), (
                    module_path.name,
                    specifier,
                )


def test_executable_asset_guards_reject_bypass_url_forms() -> None:
    """A future guard change must not miss alternate executable URL syntax."""
    for script_src in (
        "https://cdn.example.invalid/runtime.js",
        "//cdn.example.invalid/runtime.js",
    ):
        assert _REMOTE_SCRIPT_SRC_RE.search(f'<script src="{script_src}">')
    assert _DATA_SCRIPT_SRC_RE.search('<script src="data:text/javascript,alert(1)">')

    for import_re in (
        _MODULE_FROM_IMPORT_RE,
        _SIDE_EFFECT_IMPORT_RE,
        _DYNAMIC_IMPORT_RE,
    ):
        if import_re is _MODULE_FROM_IMPORT_RE:
            source = (
                'import { runtime } from "https://cdn.example.invalid/runtime.mjs";'
            )
        elif import_re is _SIDE_EFFECT_IMPORT_RE:
            source = 'import "https://cdn.example.invalid/runtime.mjs";'
        else:
            source = 'import("https://cdn.example.invalid/runtime.mjs");'
        match = import_re.search(source)
        assert match is not None
        assert not match.group("specifier").startswith("./")

    for source in (
        'import{runtime}from"https://cdn.example.invalid/runtime.mjs";',
        'export{runtime}from"https://cdn.example.invalid/runtime.mjs";',
    ):
        match = _MODULE_FROM_IMPORT_RE.search(source)
        assert match is not None
        assert not match.group("specifier").startswith("./")

    assert _UNQUOTED_SCRIPT_SRC_RE.search("<script src=//cdn.example.invalid/x.js>")


def test_chart_attribution_matches_vendored_notice() -> None:
    """The attribution must retain the vendored Chart.js copyright year."""
    chart_source = (_UI_ROOT / "vendor" / "chart.umd.min.js").read_text()
    attributions = (_REPO_ROOT / "ATTRIBUTIONS.md").read_text()
    assert "(c) 2025 Chart.js Contributors" in chart_source
    assert "Copyright (c) 2014-2025 Chart.js Contributors" in attributions


def test_api_urls_encode_user_controlled_path_parts() -> None:
    assert _raw_api_url_interpolation_violations() == []

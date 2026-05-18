# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static adversarial security checks for the operator UI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tests.unit.ui.node_utils import run_node

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_COMPONENT_RAIL_PATH = _UI_ROOT / "components" / "job-detail-rail.js"
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
            line = _line_number(src, match.start())
            violations.append(
                f"{rel_path}:{line} {match.group('store')}.setItem stores sensitive-looking payload without nearby redaction"
            )
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


def test_api_urls_encode_user_controlled_path_parts() -> None:
    assert _raw_api_url_interpolation_violations() == []


def test_script_like_rail_labels_remain_interpolated_values_not_template_markup() -> (
    None
):
    script_like_label = '<script>alert("rail")</script><img src=x onerror=alert(1)>'
    script = f"""
        import fs from 'node:fs';
        const source = fs.readFileSync({str(_COMPONENT_RAIL_PATH)!r}, 'utf8')
          .replace(new RegExp('^import .*;\\n', 'gm'), '')
          .replace(/export function /g, 'function ');

        function html(strings, ...values) {{
          return {{ __html: true, strings: Array.from(strings), values }};
        }}

        eval(source + '\\nglobalThis.RailAction = RailAction; globalThis.RailCard = RailCard; globalThis.RailKv = RailKv;');

        function templateStrings(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{
            for (const item of node) templateStrings(item, out);
            return out;
          }}
          if (typeof node === 'object' && node.__html) {{
            out.push(...node.strings);
            for (const value of node.values) templateStrings(value, out);
          }}
          return out;
        }}

        function collectValues(node, out = []) {{
          if (node == null || node === false) return out;
          if (Array.isArray(node)) {{
            for (const item of node) collectValues(item, out);
            return out;
          }}
          if (typeof node === 'object' && node.__html) {{
            for (const value of node.values) collectValues(value, out);
            return out;
          }}
          out.push(node);
          return out;
        }}

        const rendered = [
          RailAction({{ icon: '!', label: {json.dumps(script_like_label)}, onClick: () => {{}} }}),
          RailCard({{ title: {json.dumps(script_like_label)}, testId: 'evil-card', children: [] }}),
          RailKv({{ k: {json.dumps(script_like_label)}, v: {json.dumps(script_like_label)} }}),
        ];
        const templates = templateStrings(rendered).join('');
        const values = collectValues(rendered).map(String);
        console.log(JSON.stringify({{
          valuesContainLabel: values.includes({json.dumps(script_like_label)}),
          templatesContainLabel: templates.includes({json.dumps(script_like_label)}),
        }}));
    """

    out = json.loads(run_node(script))
    assert out == {"valuesContainLabel": True, "templatesContainLabel": False}

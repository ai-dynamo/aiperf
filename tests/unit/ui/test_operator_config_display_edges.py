# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static edge tests for operator UI config display and relaunch behavior."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_JOB_DETAIL_JS = _UI_ROOT / "pages" / "job-detail.js"
_RELAUNCH_BUTTON_JS = _UI_ROOT / "components" / "relaunch-button.js"


SENSITIVE_KEYS = (
    "api_key",
    "apiKey",
    "authorization",
    "password",
    "secret",
    "token",
)


def _source(path: Path) -> str:
    return path.read_text()


def _function_body(source: str, function_name: str) -> str:
    signature = re.search(rf"function {re.escape(function_name)}\([^)]*\) \{{", source)
    assert signature is not None, f"{function_name} must remain statically testable"
    start = signature.end() - 1
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError(f"{function_name} body was not balanced")


def test_job_config_yaml_viewer_redacts_secret_like_fields_before_serializing() -> None:
    """The config modal must not expose endpoint credentials copied from CR specs."""
    src = _source(_JOB_DETAIL_JS)
    body = _function_body(src, "JobConfigSection")

    assert "redact" in src.lower(), "config display needs an explicit redaction path"
    assert all(key in src for key in SENSITIVE_KEYS)
    assert "content=${serializeYaml" not in body, (
        "raw spec should not be serialized directly"
    )


def test_relaunch_prefill_redacts_secret_like_fields_before_session_storage() -> None:
    """Re-launch prefill lives in sessionStorage, so it must not copy raw secrets."""
    src = _source(_RELAUNCH_BUTTON_JS)
    body = _function_body(src, "RelaunchButton")

    assert "redact" in src.lower(), (
        "relaunch should scrub credentials before writing sessionStorage"
    )
    assert all(key in src for key in SENSITIVE_KEYS)
    assert "spec," not in body, (
        "raw spec should not be embedded in the prefilled manifest"
    )


def test_missing_config_fetch_does_not_leave_job_detail_in_permanent_loading_state() -> (
    None
):
    """A missing/404 config response should render an unavailable state, not spin forever."""
    src = _source(_JOB_DETAIL_JS)

    assert "const [jobConfigMissing" in src or "const [jobConfigLoaded" in src
    assert ".then(r => r.ok ? r.json() : null)" not in src
    assert "Loading job configuration" in src
    assert (
        "configuration unavailable" in src.lower()
        or "config unavailable" in src.lower()
    )


def test_nested_benchmark_and_sweep_config_are_addressed_by_summary_logic() -> None:
    """Config summaries should handle current nested benchmark shape and sweep metadata."""
    src = _source(_JOB_DETAIL_JS)
    body = _function_body(src, "JobConfigSection")

    assert "const benchmark = spec.benchmark ?? spec;" in body
    assert "benchmark.endpoint" in body
    assert "benchmark.models" in body
    assert "benchmark.phases" in body
    assert "benchmark.datasets" in body
    assert "spec.sweep" in body or "benchmark.sweep" in body


def test_yaml_serializers_quote_urls_so_launch_parser_does_not_split_on_colons() -> (
    None
):
    """The display and relaunch YAML emitters should share URL-safe quoting rules."""
    job_src = _source(_JOB_DETAIL_JS)
    relaunch_src = _source(_RELAUNCH_BUTTON_JS)

    assert "/^[\\w./@\\-+]+$/" in relaunch_src
    assert "/^[\\w./@\\-+]+$/" in job_src
    assert "/^[\\w./:@\\-+]+$/" not in job_src


def test_config_modal_uses_yaml_download_and_preserves_long_nested_values() -> None:
    """Long JSON-ish/YAML values should remain readable and downloadable as YAML."""
    src = _source(_JOB_DETAIL_JS)
    section = _function_body(src, "JobConfigSection")
    modal = _function_body(src, "SpecViewerModal")

    assert "filename=${(name ?? 'aiperfjob') + '.yaml'}" in section
    assert "type: 'application/yaml'" in modal
    assert "white-space: pre" in modal
    assert "overflow: auto" in src
    # A long nested value is now ellipsised to one line instead of wrapping.
    # That is only acceptable while the full text stays recoverable, so the
    # tooltip is the load-bearing half of the pair.
    assert 'class="job-config-item-value"' in section
    assert "title=${item.value}" in section
    css = (_JOB_DETAIL_JS.parents[1] / "style.css").read_text(encoding="utf-8")
    rule = css.split(".job-config-item-value {", 1)[1].split("}", 1)[0]
    assert "text-overflow: ellipsis" in rule


def test_relaunch_button_only_renders_for_non_empty_specs_and_keeps_source_identity() -> (
    None
):
    """Relaunch assumes a CR-shaped config and should preserve source metadata in the handoff."""
    src = _source(_RELAUNCH_BUTTON_JS)
    body = _function_body(src, "RelaunchButton")

    assert "const spec = config?.spec;" in body
    assert (
        "if (!spec || Object.keys(spec).length === 0 || !namespace || !name) return null;"
        in body
    )
    assert "sourceNs: namespace" in body
    assert "sourceName: name" in body
    assert "kind: config.kind ?? 'AIPerfJob'" in body
    assert "apiVersion: config.apiVersion ?? 'aiperf.nvidia.com/v1alpha1'" in body

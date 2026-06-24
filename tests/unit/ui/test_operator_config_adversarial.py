# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial static tests for operator UI config display and relaunch safety."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UI_ROOT = _REPO_ROOT / "src" / "aiperf" / "operator" / "ui"
_JOB_DETAIL_JS = _UI_ROOT / "pages" / "job-detail.js"
_RELAUNCH_BUTTON_JS = _UI_ROOT / "components" / "relaunch-button.js"
_LAUNCH_JS = _UI_ROOT / "pages" / "launch.js"

_SECRET_KEY_EXAMPLES = (
    "api_key",
    "apiKey",
    "authorization",
    "bearerToken",
    "client_secret",
    "password",
    "secretRef",
    "token",
)

_K8S_METADATA_EXAMPLES = (
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
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


def test_config_display_and_relaunch_use_shared_recursive_secret_redaction() -> None:
    """Secret-looking keys must be scrubbed at arbitrary nesting depth before YAML exists."""
    job_src = _source(_JOB_DETAIL_JS)
    relaunch_src = _source(_RELAUNCH_BUTTON_JS)
    combined = job_src + "\n" + relaunch_src

    assert re.search(r"function\s+redact[A-Za-z0-9_]*\(", combined), (
        "config display and relaunch need an explicit shared redaction helper"
    )
    assert re.search(
        r"Object\.entries\(|for \(const \[[^\]]+\] of Object\.entries", combined
    ), "redaction must walk object entries instead of only known top-level paths"
    assert "Array.isArray" in combined, (
        "redaction must recurse into arrays of secret-bearing objects"
    )
    assert all(key in combined for key in _SECRET_KEY_EXAMPLES), (
        "redaction key matcher should cover snake/camel/key-ref variants seen in configs"
    )
    assert "[REDACTED]" in combined, "redacted YAML should show a stable placeholder"


def test_config_modal_serializes_redacted_sweep_specs_not_raw_spec() -> None:
    """AIPerfSweep configs can hide credentials under sweep variations or child templates."""
    src = _source(_JOB_DETAIL_JS)
    body = _function_body(src, "JobConfigSection")

    assert "redact" in body.lower(), (
        "JobConfigSection must redact before passing content to the modal"
    )
    assert "content=${serializeYaml({" not in body, (
        "modal content must not serialize raw spec inline"
    )
    assert "spec.sweep" in body or "benchmark.sweep" in body, (
        "summary/display logic must account for sweep specs"
    )
    assert "AIPerfSweep" in body or "config.kind" in body, (
        "modal should preserve sweep kind when showing sweep YAML"
    )


def test_yaml_serializers_quote_urls_with_colons_in_display_and_relaunch() -> None:
    """URL strings like http://svc:8000 must not be emitted as ambiguous bare scalars."""
    job_src = _source(_JOB_DETAIL_JS)
    relaunch_src = _source(_RELAUNCH_BUTTON_JS)

    assert "/^[\\w./@\\-+]+$/" in job_src, (
        "job-detail YAML serializer must exclude ':' from bare scalars"
    )
    assert "/^[\\w./@\\-+]+$/" in relaunch_src, (
        "relaunch YAML serializer must exclude ':' from bare scalars"
    )
    assert "/^[\\w./:@\\-+]+$/" not in job_src + relaunch_src


def test_yaml_serializers_handle_multiline_strings_explicitly() -> None:
    """Multiline prompts, headers, and env values should round-trip as valid readable YAML."""
    combined = _source(_JOB_DETAIL_JS) + "\n" + _source(_RELAUNCH_BUTTON_JS)

    assert "includes('\\n')" in combined or 'includes("\\n")' in combined, (
        "YAML serializers must branch explicitly for multiline strings"
    )
    assert "|" in combined or "\\n" in combined, (
        "multiline strings should be block-emitted or escaped deliberately"
    )


def test_malformed_config_fetch_reaches_terminal_unavailable_state() -> None:
    """404s, invalid JSON, and network errors must not leave the config card spinning forever."""
    src = _source(_JOB_DETAIL_JS)

    assert (
        "jobConfigLoaded" in src
        or "jobConfigError" in src
        or "configUnavailable" in src
    )
    assert ".then(r => r.ok ? r.json() : null)" not in src, (
        "non-OK config responses need explicit loaded/error state"
    )
    assert re.search(r"catch\([^)]*\)\s*=>\s*\{[^}]*setJobConfig", src, re.DOTALL), (
        "config fetch catch path should update config state, not silently swallow"
    )
    assert (
        "configuration unavailable" in src.lower()
        or "config unavailable" in src.lower()
    )


def test_relaunch_manifest_strips_server_owned_metadata_and_status() -> None:
    """Relaunch handoff must not copy immutable Kubernetes metadata from the source CR."""
    src = _source(_RELAUNCH_BUTTON_JS)
    body = _function_body(src, "RelaunchButton")

    assert "status" not in body, (
        "relaunch manifest must not carry status back to the launch form"
    )
    assert "metadata: config.metadata" not in body
    assert "labels: config" not in body and "annotations: config" not in body
    assert all(field in src for field in _K8S_METADATA_EXAMPLES), (
        "metadata stripping should name the server-owned fields it intentionally drops"
    )
    assert re.search(
        r"metadata:\s*\{\s*name: suggestRetryName\(name\),\s*namespace,\s*\}",
        body,
        re.DOTALL,
    )


def test_relaunch_does_not_navigate_when_session_storage_set_fails() -> None:
    """QuotaExceededError/SecurityError during prefill should keep the user on the detail page."""
    src = _source(_RELAUNCH_BUTTON_JS)
    body = _function_body(src, "RelaunchButton")

    set_pos = body.index("sessionStorage.setItem")
    catch_pos = body.index("catch (err)")
    navigate_pos = body.index("navigate('/launch')")
    return_pos = body.index("return;", catch_pos)
    assert set_pos < catch_pos < return_pos < navigate_pos
    assert "console.warn('Unable to prepare launch prefill', err);" in body


def test_launch_consumes_prefill_defensively_when_session_storage_access_fails() -> (
    None
):
    """SecurityError from getItem/removeItem and malformed JSON should not break Launch mount."""
    src = _source(_LAUNCH_JS)
    body = _function_body(src, "Launch")

    get_pos = body.index("sessionStorage.getItem('aiperf.launch.prefill')")
    first_catch_pos = body.index("catch (_e) { return; }", get_pos)
    remove_pos = body.index("sessionStorage.removeItem('aiperf.launch.prefill')")
    json_parse_pos = body.index("JSON.parse(raw)")
    stale_check_pos = body.index("Date.now() - payload.at > 60000")
    assert get_pos < first_catch_pos < remove_pos < json_parse_pos < stale_check_pos
    assert body.count("catch (_e)") >= 3, (
        "getItem, removeItem, and JSON.parse must each be guarded"
    )

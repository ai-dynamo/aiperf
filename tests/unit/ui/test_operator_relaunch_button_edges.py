# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge tests for operator UI relaunch config cloning behavior.

The relaunch component currently keeps manifest construction inside the Preact
click handler, so these tests use static checks for the handler contract and a
small extracted-function runtime check for retry-name behavior.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELAUNCH_BUTTON_JS = (
    _REPO_ROOT
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "relaunch-button.js"
)


def _source() -> str:
    return _RELAUNCH_BUTTON_JS.read_text()


def _exported_function_source(source: str, function_name: str) -> str:
    match = re.search(
        rf"export function {re.escape(function_name)}\([^)]*\) \{{(?P<body>.*?)\n\}}",
        source,
        re.DOTALL,
    )
    assert match is not None, f"{function_name} must remain statically testable"
    signature = re.search(
        rf"export function {re.escape(function_name)}\([^)]*\)", source
    )
    assert signature is not None
    return (
        signature.group(0).replace("export ", "") + " {" + match.group("body") + "\n}"
    )


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _strip_js_comments(source: str) -> str:
    """Drop // line and /* */ block comments so substring scans only see code.

    The relaunch component documents the server-owned fields it drops
    (``managedFields``, ``resourceVersion``, ``uid``, ...) in a leading comment;
    those words must not be matched as if they were live manifest keys.
    """
    no_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", no_blocks)


def test_relaunch_manifest_strips_existing_resource_metadata_and_status() -> None:
    """Cloned manifests must not carry server-owned CR metadata into Launch."""
    src = _source()
    code = _strip_js_comments(src)

    assert (
        "metadata: {\n            name: suggestRetryName(name),\n            namespace,"
        in src
    )
    assert "metadata: config.metadata" not in code
    assert "status:" not in code
    assert "managedFields" not in code
    assert "resourceVersion" not in code
    assert "uid" not in code


def test_relaunch_manifest_preserves_source_namespace_in_manifest_and_prefill() -> None:
    """Relaunch should clone into the original namespace unless the editor changes it."""
    src = _source()

    assert "namespace," in src
    assert "sourceNs: namespace" in src
    assert (
        "metadata: {\n            name: suggestRetryName(name),\n            namespace,"
        in src
    )


def test_relaunch_preserves_job_vs_sweep_kind_and_copies_spec_verbatim() -> None:
    """AIPerfSweep relaunches must stay sweeps; jobs must stay jobs."""
    src = _source()

    assert "kind: config.kind ?? 'AIPerfJob'" in src
    assert "const spec = config?.spec;" in src
    # Spec is copied wholesale (job stays a job, sweep stays a sweep) but routed
    # through redactConfigForYaml so sensitive keys never land in the editor.
    assert "spec: redactConfigForYaml(spec)," in src
    assert "delete spec.sweep" not in src
    assert "kind: 'AIPerfJob'" not in src


def test_retry_name_does_not_collide_for_two_relaunches_in_same_minute() -> None:
    """Two launches in the same minute need distinct target names to avoid CR conflict."""
    fn_src = _exported_function_source(_source(), "suggestRetryName")
    output = _run_node(
        f"""
        const RealDate = Date;
        globalThis.Date = class extends RealDate {{
          constructor(...args) {{
            return args.length ? new RealDate(...args) : new RealDate(2026, 4, 18, 12, 34, 0);
          }}
        }};
        {fn_src}
        console.log(JSON.stringify([suggestRetryName('bench'), suggestRetryName('bench')]));
        """
    )
    first, second = json.loads(output)

    assert first != second


def test_retry_name_strips_existing_retry_suffix_before_suggesting_new_name() -> None:
    """Repeated relaunches should replace, not stack, the retry timestamp suffix."""
    fn_src = _exported_function_source(_source(), "suggestRetryName")
    output = _run_node(
        f"""
        const RealDate = Date;
        globalThis.Date = class extends RealDate {{
          constructor(...args) {{
            return args.length ? new RealDate(...args) : new RealDate(2026, 4, 18, 12, 34, 0);
          }}
        }};
        {fn_src}
        console.log(suggestRetryName('bench-retry-260518-1111'));
        """
    )

    assert output == "bench-retry-260518-1234"


def test_relaunch_button_is_disabled_or_hidden_when_identity_is_incomplete() -> None:
    """Avoid serializing undefined name/namespace into the cloned manifest."""
    src = _source()

    has_identity_guard = "!namespace" in src or "!name" in src
    has_disabled_state = "disabled=" in src or "aria-disabled" in src
    assert has_identity_guard or has_disabled_state


def test_session_storage_errors_are_not_silently_treated_as_success() -> None:
    """Private-mode/quota failures should not navigate to an empty launch editor."""
    src = _source()

    assert "fall through to navigate" not in src
    assert "catch (_e)" not in src

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the local operator-UI screenshot harness."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_CAPTURE = _REPO_ROOT / "dev" / "ui-verify" / "capture_fixtures.py"
_README = _REPO_ROOT / "dev" / "ui-verify" / "README.md"
_SERVER = _REPO_ROOT / "dev" / "ui-verify" / "serve.mjs"
_SHOOTER = _REPO_ROOT / "dev" / "ui-verify" / "shoot.mjs"


def test_ui_verify_server_serves_mjs_modules_as_javascript() -> None:
    """Browsers reject module scripts sent with the octet-stream fallback."""
    source = _SERVER.read_text()
    assert "'.mjs': 'text/javascript; charset=utf-8'" in source


def test_ui_verify_harness_uses_ignored_output_and_no_personal_namespace() -> None:
    """Review captures must not recreate tracked files or embed a personal namespace."""
    shooter = _SHOOTER.read_text()
    fixture_capture = _FIXTURE_CAPTURE.read_text()
    readme = _README.read_text()

    assert "artifacts/ui-verify/shots" in shooter
    assert "process.env.NAMESPACE" in shooter
    assert "acasagrande-aiperf-bench" not in shooter
    assert "acasagrande-aiperf-bench" not in fixture_capture
    assert "acasagrande-aiperf-bench" not in readme

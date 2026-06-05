# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tests/harness/utils.py — specifically the platform-branching
shlex behavior introduced for Windows path support (Bug 6).

POSIX-mode shlex.split treats backslash as an escape character. Windows
paths like C:\\Users\\... would be silently mangled into C:Users... on
non-POSIX-shlex parsing. The harness now selects POSIX vs non-POSIX mode
based on sys.platform so test commands that interpolate Windows paths
preserve their backslashes.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.harness.utils import AIPerfCLI


class TestParseCommandShlexMode:
    """Verify _parse_command picks POSIX vs non-POSIX shlex mode by platform."""

    def test_unix_uses_posix_mode_normal_path(self) -> None:
        """On non-Windows the parser runs in POSIX mode (default shlex behavior)."""
        with patch("tests.harness.utils.sys.platform", "linux"):
            args = AIPerfCLI._parse_command(
                "aiperf profile --file /tmp/data.jsonl --request-count 5"
            )
        assert args == ["profile", "--file", "/tmp/data.jsonl", "--request-count", "5"]

    def test_windows_uses_non_posix_mode_preserves_backslashes(self) -> None:
        """On Windows the parser runs in non-POSIX mode so backslashes in
        interpolated paths (C:\\Users\\...) are preserved as literal chars
        rather than treated as escape introducers."""
        cmd = r"aiperf profile --file C:\Users\test\data.jsonl --request-count 5"
        with patch("tests.harness.utils.sys.platform", "win32"):
            args = AIPerfCLI._parse_command(cmd)
        assert args == [
            "profile",
            "--file",
            r"C:\Users\test\data.jsonl",
            "--request-count",
            "5",
        ]

    def test_unix_posix_mode_strips_backslashes_from_windows_style_paths(self) -> None:
        """Confirms the bug: with POSIX shlex (the pre-fix Linux/macOS code
        path), backslashes are eaten as escape characters. This is why
        Windows-runtime tests need the platform branch — the fix is not
        just cosmetic."""
        cmd = r"aiperf profile --file C:\Users\test\data.jsonl"
        with patch("tests.harness.utils.sys.platform", "linux"):
            args = AIPerfCLI._parse_command(cmd)
        # POSIX shlex consumed every backslash as an escape introducer
        assert "C:Userstestdata.jsonl" in args
        assert r"C:\Users\test\data.jsonl" not in args

    def test_drops_leading_aiperf_token_on_both_platforms(self) -> None:
        """Sanity: the post-shlex slicing logic (drop leading 'aiperf') runs
        identically regardless of shlex mode."""
        for plat in ("linux", "win32"):
            with patch("tests.harness.utils.sys.platform", plat):
                args = AIPerfCLI._parse_command("aiperf profile --request-count 1")
            assert args[0] == "profile", f"first arg wrong on {plat}: {args}"

    def test_handles_continuation_backslashes_on_both_platforms(self) -> None:
        """Backslash-newline continuations are normalized BEFORE shlex runs
        (cmd.replace("\\\\\\n", " ")), so they work regardless of platform."""
        cmd = "aiperf profile \\\n  --request-count 1"
        for plat in ("linux", "win32"):
            with patch("tests.harness.utils.sys.platform", plat):
                args = AIPerfCLI._parse_command(cmd)
            assert args == ["profile", "--request-count", "1"], (
                f"continuation handling wrong on {plat}: {args}"
            )

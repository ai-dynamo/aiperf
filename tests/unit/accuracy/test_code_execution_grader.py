# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted unit tests for ``code_execution`` grader internals.

Covers the in-process helpers that don't need a real sandbox
(``_decode_private_test_cases``, which translates LCB's upstream-encoded
``private_test_cases`` blob: base64 → zlib → pickle → json) plus the
daemon-flag handling in ``grade`` (with ``codegen_metrics`` mocked).

The real daemon-fork path — spawning a ``ProcessPoolExecutor`` from a
daemon process, which is what actually broke LCB grading — is guarded in
``tests/component_integration/test_code_execution_daemon_grading.py``.
"""

from __future__ import annotations

import base64
import multiprocessing as mp
import pickle
import zlib
from unittest.mock import MagicMock

import orjson
import pytest

import aiperf.accuracy.graders.code_execution as code_execution
from aiperf.accuracy.graders.code_execution import (
    CodeExecutionGrader,
    _decode_private_test_cases,
)


def _encode_lcb_private_test_cases(cases: list[dict[str, str]]) -> str:
    """Mirror upstream LCB's encoding so tests can build realistic
    ``private_test_cases`` payloads without hand-rolling base64.

    Matches the inverse of
    ``lighteval.tasks.tasks.lcb.codegen_metrics.translate_private_test_cases``:
    ``json.dumps(cases) -> pickle.dumps -> zlib.compress -> base64.b64encode``.
    """
    json_bytes = orjson.dumps(cases)
    return base64.b64encode(zlib.compress(pickle.dumps(json_bytes.decode()))).decode()


class TestDecodePrivateTestCases:
    """``_decode_private_test_cases`` is the only consumer that needs to
    handle LCB's encoded blob; pin both the encoded path (production
    data) and the legacy plain-JSON fallback (test fixtures and older
    in-process callers)."""

    def test_decodes_lcb_encoded_blob(self) -> None:
        """Production LCB data is base64/zlib/pickle/json — the
        encoded path must round-trip through ``translate_private_test_cases``
        to recover the list of cases.

        Skip cleanly when ``lighteval`` isn't installed: without
        ``translate_private_test_cases`` the decoder falls through to
        plain-JSON parsing, which would fail confusingly on base64
        bytes instead of testing what the docstring claims.
        """
        pytest.importorskip(
            "lighteval.tasks.tasks.lcb.codegen_metrics",
            reason="encoded-blob decode requires lighteval's translate_private_test_cases",
        )
        cases = [
            {"input": "[1, 2]", "output": "[2, 1]"},
            {"input": "[3]", "output": "[3]"},
        ]
        encoded = _encode_lcb_private_test_cases(cases)
        assert _decode_private_test_cases(encoded) == cases

    def test_falls_back_to_plain_json_string(self) -> None:
        """Test fixtures and pre-encoded-era callers pass
        ``private_test_cases`` as a plain JSON string. The fallback
        path must still accept that so existing callers don't break."""
        cases = [{"input": "x", "output": "y"}]
        raw = orjson.dumps(cases).decode()
        assert _decode_private_test_cases(raw) == cases

    def test_passes_through_already_deserialized_list(self) -> None:
        """An in-process caller may hand the grader a pre-parsed
        list of dicts. Pass it through verbatim — no encode/decode
        round-trip."""
        cases = [{"input": "x", "output": "y"}]
        assert _decode_private_test_cases(cases) is cases

    def test_empty_or_missing_returns_empty(self) -> None:
        """A payload with no private cases (None / empty string /
        empty list) returns ``[]`` so the caller can concatenate
        with public cases without a special-case."""
        assert _decode_private_test_cases(None) == []
        assert _decode_private_test_cases("") == []
        assert _decode_private_test_cases([]) == []


@pytest.mark.asyncio
class TestGradeClearsDaemonFlag:
    """Regression: AIPerf runs the record processor as a daemon (every service
    is spawned ``daemon=True``). ``codegen_metrics`` fans out to a
    ProcessPoolExecutor, which Python forbids from a daemon process. Before the
    fix, grading died with "daemonic processes are not allowed to have children"
    and was silently mislabeled ``unparsed``. ``grade`` must clear the daemon
    flag around the ``codegen_metrics`` call.
    """

    def _set_daemon(self, value: bool) -> None:
        try:
            mp.current_process().daemon = value
        except AssertionError:
            mp.current_process()._config["daemon"] = value

    async def test_codegen_metrics_runs_with_daemon_cleared(self, monkeypatch) -> None:
        # Record the daemon flag as seen from inside codegen_metrics.
        seen: dict[str, bool] = {}

        def fake_codegen_metrics(*_args, **_kwargs):
            seen["daemon"] = mp.current_process().daemon
            return {"pass@1": 1.0}, {}

        monkeypatch.setattr(code_execution, "_HAS_LIGHTEVAL_LCB", True)
        monkeypatch.setattr(code_execution, "codegen_metrics", fake_codegen_metrics)
        monkeypatch.setattr(code_execution, "extract_code", lambda _text: "print(1)")

        grader = CodeExecutionGrader(run=MagicMock())
        payload = orjson.dumps(
            {"public_test_cases": [{"input": "1", "output": "1"}], "metadata": ""}
        ).decode()

        original = mp.current_process().daemon
        try:
            self._set_daemon(True)
            result = await grader.grade("```python\nprint(1)\n```", payload)
        finally:
            self._set_daemon(original)

        # codegen_metrics saw a non-daemon process (the fix), and the daemon
        # flag was restored afterward.
        assert seen["daemon"] is False
        assert mp.current_process().daemon == original
        # Grading ran to completion instead of being mislabeled unparsed.
        assert result.unparsed is False
        assert result.correct is True

    async def test_codegen_metrics_exception_becomes_grading_failure(
        self, monkeypatch
    ) -> None:
        """If sandboxed execution raises (e.g. the daemon-fork error), grade()
        must return a clean failure result — not propagate and crash the
        record processor."""

        def _boom(*_args, **_kwargs):
            raise RuntimeError("daemonic processes are not allowed to have children")

        monkeypatch.setattr(code_execution, "_HAS_LIGHTEVAL_LCB", True)
        monkeypatch.setattr(code_execution, "codegen_metrics", _boom)
        monkeypatch.setattr(code_execution, "extract_code", lambda _text: "print(1)")

        grader = CodeExecutionGrader(run=MagicMock())
        payload = orjson.dumps(
            {"public_test_cases": [{"input": "1", "output": "1"}], "metadata": ""}
        ).decode()

        result = await grader.grade("```python\nprint(1)\n```", payload)
        assert result.correct is False
        assert result.unparsed is True
        assert "sandboxed exec failed" in result.reasoning

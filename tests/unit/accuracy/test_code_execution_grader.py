# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted unit tests for ``code_execution`` grader internals.

Covers the in-process helpers that don't need a real sandbox
(``_decode_private_test_cases``, which translates LCB's upstream-encoded
``private_test_cases`` blob: base64 -> zlib -> pickle -> json; and
``_derive_grade_timeout``) plus ``grade``'s delegation to the out-of-process
codegen worker (with ``CodegenGradingWorker.grade_codegen`` mocked at the
client boundary).

The real fork-based sandbox now lives in the worker subprocess and is
exercised by ``tests/component_integration/test_code_execution_daemon_grading.py``.
"""

from __future__ import annotations

import base64
import pickle
import zlib
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

import aiperf.accuracy.graders.code_execution as code_execution
from aiperf.accuracy.graders._codegen_worker_client import CodegenWorkerError
from aiperf.accuracy.graders.code_execution import (
    CodeExecutionGrader,
    _decode_private_test_cases,
    _derive_grade_timeout,
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


class TestDeriveGradeTimeout:
    """``_derive_grade_timeout`` sizes the client-side wall-clock ceiling to the
    test-case count and hard-caps it so a single wedged worker can't stall the
    run without firing on merely slow grades."""

    def test_scales_with_cases_and_caps_at_5_min(self) -> None:
        assert _derive_grade_timeout(1) < _derive_grade_timeout(10)
        assert _derive_grade_timeout(100000) == 300.0

    def test_cap_is_configurable_via_env(self, monkeypatch) -> None:
        """The hard cap reads AIPERF_ACCURACY_LCB_GRADE_TIMEOUT_MAX_S so a slow
        large-problem grade need not be prematurely failed by the default 300s."""
        from aiperf.common.environment import Environment

        monkeypatch.setattr(Environment.ACCURACY, "LCB_GRADE_TIMEOUT_MAX_S", 60.0)
        assert _derive_grade_timeout(100000) == 60.0
        # Small problems stay below the cap regardless.
        assert _derive_grade_timeout(1) == 7.0 + 5.0 + 30.0


@pytest.mark.asyncio
class TestGradeDelegatesToWorker:
    """``grade`` delegates sandboxed execution to the out-of-process worker via
    ``CodegenGradingWorker`` and maps its result / errors onto ``GradingResult``."""

    async def test_grade_uses_worker_and_maps_pass_at_1(self, monkeypatch) -> None:
        monkeypatch.setattr(code_execution, "_HAS_LIGHTEVAL_LCB", True)
        monkeypatch.setattr(code_execution, "extract_code", lambda _t: "print(1)")
        grader = CodeExecutionGrader(run=MagicMock())
        grader._worker.grade_codegen = AsyncMock(return_value={"pass@1": 1.0})

        payload = orjson.dumps(
            {"public_test_cases": [{"input": "1", "output": "1"}], "metadata": ""}
        ).decode()
        result = await grader.grade("```python\nprint(1)\n```", payload)
        assert result.correct is True
        assert result.unparsed is False
        grader._worker.grade_codegen.assert_awaited_once()

    async def test_worker_error_becomes_grading_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(code_execution, "_HAS_LIGHTEVAL_LCB", True)
        monkeypatch.setattr(code_execution, "extract_code", lambda _t: "print(1)")
        grader = CodeExecutionGrader(run=MagicMock())
        grader._worker.grade_codegen = AsyncMock(side_effect=CodegenWorkerError("boom"))

        payload = orjson.dumps(
            {"public_test_cases": [{"input": "1", "output": "1"}], "metadata": ""}
        ).decode()
        result = await grader.grade("```python\nprint(1)\n```", payload)
        assert result.correct is False
        assert result.unparsed is True
        assert "sandboxed exec failed" in result.reasoning

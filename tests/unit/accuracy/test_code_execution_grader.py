# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted unit tests for ``code_execution`` grader internals.

The full sandboxed grading path (``CodeExecutionGrader.grade``) needs
``lighteval``'s ``codegen_metrics`` plus a ``ProcessPoolExecutor``, so
end-to-end grading is exercised in the component_integration suite.
These tests cover the in-process helpers that don't need a sandbox:
``_decode_private_test_cases``, which translates LCB's upstream-encoded
``private_test_cases`` blob (base64 → zlib → pickle → json) into the
list of ``{"input", "output"}`` dicts ``codegen_metrics`` expects.
"""

from __future__ import annotations

import base64
import pickle
import zlib

import orjson
import pytest

from aiperf.accuracy.graders.code_execution import (
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

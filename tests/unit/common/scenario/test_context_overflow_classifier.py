# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``is_context_overflow_response`` classifies overflow responses from raw body text and JSON error shapes."""

import pytest

from aiperf.common.scenario import is_context_overflow_response


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # Plain-text body, exact substring.
        ("Error: context length exceeded for this prompt", True),
        # Case-insensitive against the body.
        ("ERROR: CONTEXT LENGTH EXCEEDED", True),
        ("Maximum Context tokens reached", True),
        # OpenAI-style nested error.message.
        (
            b'{"error": {"message": "This model\'s maximum context length is 4096 tokens.", "type": "invalid_request_error", "code": "context_length_exceeded"}}',
            True,
        ),
        # OpenAI shape but the substring lives only in the .code, not .message:
        # we still match because the raw body contains it.
        (
            '{"error": {"message": "bad", "code": "context_length_exceeded"}}',
            True,
        ),
        # vLLM-style flat detail body. No nested error.message but raw text matches.
        ('{"detail": "Prompt is too long: 12345 > 4096"}', True),
        # JSON with a string-shaped error field.
        ('{"error": "context length too big"}', True),
        # No match -- unrelated server error.
        ("Internal server error", False),
        ("502 Bad Gateway", False),
        ('{"error": {"message": "rate limit"}}', False),
        # Empty / None / zero-length.
        (None, False),
        ("", False),
        (b"", False),
    ],
)
def test_is_context_overflow_response_default_substrings(
    body: str | bytes | None, expected: bool
) -> None:
    assert is_context_overflow_response(body=body) is expected


def test_is_context_overflow_response_custom_substrings() -> None:
    """Caller-provided substring list overrides the env default."""
    body = "ServerError: kv-cache full while decoding"
    # Default allowlist doesn't catch this.
    assert is_context_overflow_response(body=body) is False
    # Caller can extend.
    assert is_context_overflow_response(body=body, substrings=["kv-cache full"]) is True


def test_is_context_overflow_response_empty_substring_list_disables_detection() -> None:
    """An empty allowlist short-circuits to False even on otherwise-matching bodies."""
    body = "Error: context length exceeded"
    assert is_context_overflow_response(body=body, substrings=[]) is False


def test_is_context_overflow_response_classifies_purely_from_body() -> None:
    """The classifier's verdict comes entirely from the body, not any status code."""
    assert is_context_overflow_response(body="context length too big") is True
    assert is_context_overflow_response(body="other error") is False


def test_is_context_overflow_response_handles_invalid_utf8_bytes() -> None:
    """Bytes that fail strict UTF-8 decode still go through replace mode."""
    body = b"\xff\xfe context length \xff\xff"
    assert is_context_overflow_response(body=body) is True


def test_is_context_overflow_response_non_dict_json_falls_back_to_raw_match() -> None:
    """A JSON array body shouldn't crash the OpenAI parse step."""
    body = '["context length exceeded", "details"]'
    assert is_context_overflow_response(body=body) is True


def test_is_context_overflow_response_invalid_json_uses_raw_match_only() -> None:
    """Non-JSON raw body still works via the substring scan."""
    body = "<html><body>The prompt is too long for this model</body></html>"
    assert is_context_overflow_response(body=body) is True


def test_is_context_overflow_response_signature_excludes_status_code() -> None:
    import inspect

    params = inspect.signature(is_context_overflow_response).parameters
    assert set(params) == {"body", "substrings"}


def test_is_context_overflow_response_unknown_kwargs_raise_typeerror() -> None:
    """Passing an unsupported kwarg fails loud rather than being silently accepted."""
    with pytest.raises(TypeError):
        is_context_overflow_response(  # type: ignore[call-arg]
            body="context length too big",
            status_code=400,
        )

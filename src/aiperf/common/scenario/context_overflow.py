# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime context-overflow detection for the InferenceX AgentX scenario.

Substring allowlist lives on ``Environment.AGENTX`` so users can extend
without code changes.
"""

from typing import Any

import orjson

from aiperf.common.environment import Environment


def is_context_overflow_response(
    *,
    body: str | bytes | None,
    substrings: list[str] | None = None,
) -> bool:
    """Classify whether an error response indicates a context-overflow.

    Performs a case-insensitive substring match against:
    1. The raw response body text.
    2. The OpenAI-style nested ``error.message`` field, when the body
       parses as JSON. Falls through silently on non-JSON bodies (e.g.
       vLLM's ``{"detail": "..."}`` shape — which is still caught by the
       raw-body match in step 1).

    Callers are expected to pre-filter to error responses (the
    ``InferenceResultParser`` only invokes this on records with
    ``has_error=True``); status-code gating lives at the call site so this
    function stays a pure body-based classifier.

    Args:
        body: The raw response body. ``str`` or ``bytes``; ``None`` returns
            False. Empty body returns False.
        substrings: Override the allowlist for tests. ``None`` reads
            ``Environment.AGENTX.CONTEXT_OVERFLOW_SUBSTRINGS`` at call time
            so test settings overrides take effect.

    Returns:
        True iff at least one substring (case-insensitive) appears in
        either the raw body text or the parsed ``error.message`` field.
    """
    if body is None:
        return False

    # Resolve substring list lazily so tests can override the env setting.
    candidates: list[str] = (
        substrings
        if substrings is not None
        else list(Environment.AGENTX.CONTEXT_OVERFLOW_SUBSTRINGS)
    )
    if not candidates:
        return False

    text: str = (
        body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    )

    if not text:
        return False

    lowered = text.lower()
    needles = [s.lower() for s in candidates if s]

    # 1. Raw body match.
    for needle in needles:
        if needle in lowered:
            return True

    # 2. OpenAI-style {"error": {"message": "..."}} match.
    nested_message = _extract_openai_error_message(text)
    if nested_message:
        nested_lower = nested_message.lower()
        for needle in needles:
            if needle in nested_lower:
                return True

    return False


def _extract_openai_error_message(text: str) -> str | None:
    """Return the OpenAI-style ``error.message`` field from a JSON body.

    Returns ``None`` when the body doesn't parse as JSON, or when the
    expected ``{"error": {"message": ...}}`` shape isn't present. Tolerates
    a string-shaped ``error`` field (some servers return ``"error":
    "..."``) by using it as the message.
    """
    try:
        parsed: Any = orjson.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    err = parsed.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str):
            return msg
    elif isinstance(err, str):
        return err
    return None

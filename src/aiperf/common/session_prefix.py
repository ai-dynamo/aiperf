# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session-instance unique prompt prefix (prefix-cache de-duplication control)."""

from __future__ import annotations

import random


def make_unique_session_prefix(correlation_id: str, num_tokens: int) -> str:
    """Build a deterministic, per-session-instance unique text prefix.

    Seeded by ``correlation_id`` so the result is byte-identical across every
    turn of one session (preserving realistic within-session prefix-cache reuse)
    while differing across resampled instances of the same conversation.
    Prepending it to a session's first turn defeats the unrealistic
    cross-resample prefix-cache hits that would otherwise inflate throughput when
    a finite dataset is sampled with replacement under server prefix caching.

    Args:
        correlation_id: The session-instance id (``x_correlation_id``). The same
            id always yields the same prefix; different ids yield different ones.
        num_tokens: Approximate number of tokens to emit. Each emitted word is a
            small integer that common tokenizers map to roughly one token. Values
            ``<= 0`` return an empty string (feature disabled).

    Returns:
        A space-joined string of pseudo-random small integers, or ``""`` when
        ``num_tokens <= 0``.
    """
    if num_tokens <= 0:
        return ""
    # random.Random seeds deterministically from a str (sha512-based, version 2),
    # independent of PYTHONHASHSEED, so the prefix is stable across processes.
    rng = random.Random(correlation_id)
    return " ".join(str(rng.randint(0, 9999)) for _ in range(num_tokens))


def prepend_unique_session_prefix(
    user_context_message: str | None, correlation_id: str, num_tokens: int
) -> str | None:
    """Prepend a per-session-instance unique prefix to a user-context message.

    Returns ``user_context_message`` unchanged when ``num_tokens <= 0`` (feature
    disabled). Otherwise prepends the prefix from ``make_unique_session_prefix``,
    keeping any existing context after it.

    Args:
        user_context_message: Existing per-conversation context, or ``None``.
        correlation_id: The session-instance id (``x_correlation_id``).
        num_tokens: Approximate prefix length in tokens; ``<= 0`` disables.

    Returns:
        The combined message, the bare prefix when there was no context, or the
        unchanged input when disabled.
    """
    if num_tokens <= 0:
        return user_context_message
    prefix = make_unique_session_prefix(correlation_id, num_tokens)
    if user_context_message:
        return f"{prefix}\n\n{user_context_message}"
    return prefix

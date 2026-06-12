# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in OpenAI tool-call wire shaping for weka trace replay deltas.

Kept in its own dependency-free module so both the serial loader
(:mod:`aiperf.dataset.loader.weka_trace`) and the parallel worker
(:mod:`aiperf.dataset.loader.weka_parallel_convert`) import one source of
truth without pulling either module into the other.
"""

from __future__ import annotations


def tool_shape_delta_messages(
    messages: list[dict], *, turn_index: int, is_tool_result: bool
) -> list[dict]:
    """Reshape a tool-result turn's delta into the OpenAI tool-call wire shape.

    The same-delta assistant segment (the one that "made the call") gains a
    synthetic ``tool_calls`` entry and the new-input user segment becomes a
    ``role: "tool"`` message referencing it; content text is unchanged, so
    only the message framing differs from the default plain-user shape.

    Only the current turn's trailing assistant -> user pair is shaped (on a
    ``reset_context`` full re-emit, earlier history keeps the plain shape).
    Guarded: requires an assistant segment immediately before the final user
    segment, so turn 0 (no prior assistant) and live-assistant deltas
    (assistant segments not emitted) fall through unchanged. The call id is
    deterministic per turn index for reproducible payloads.
    """
    if not is_tool_result or len(messages) < 2:
        return messages
    if messages[-1].get("role") != "user" or messages[-2].get("role") != "assistant":
        return messages
    call_id = f"call_turn_{turn_index}"
    shaped = list(messages)
    shaped[-2] = {
        **messages[-2],
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "recorded_tool", "arguments": "{}"},
            }
        ],
    }
    shaped[-1] = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": messages[-1].get("content", ""),
    }
    return shaped

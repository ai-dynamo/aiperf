# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in OpenAI tool-call wire shaping for weka trace replay.

Shaping operates on the reconstructor's emitted segment window so it applies
uniformly to append deltas AND ``reset_context`` full re-emissions -- a reset
replaces the wire context, so emission-time-only shaping would retroactively
unshape every previously-sent tool turn. Segments are duck-typed (``role`` +
``tool_result_turn`` attributes) to keep this module dependency-free for both
the serial loader and the parallel worker.
"""

from __future__ import annotations


def tool_shape_segment_messages(messages: list[dict], segments: list) -> list[dict]:
    """Shape an emitted message window from its source segments.

    ``messages[i]`` must correspond 1:1 to ``segments[i]``. A user segment
    marked with ``tool_result_turn`` becomes a ``role: "tool"`` message and
    the assistant segment immediately before it (in the same window) gains
    the matching synthetic ``tool_calls`` entry. The call id is keyed to the
    turn recorded on the segment, so re-emissions reproduce the id the turn
    was first sent with.

    Pairing is window-local and guarded: a marked segment without an
    assistant directly before it (turn 0, post-context-loss user heads,
    window boundaries) falls back to the plain user shape.
    """
    shaped = list(messages)
    for i, seg in enumerate(segments):
        turn = getattr(seg, "tool_result_turn", None)
        if turn is None or getattr(seg, "role", None) != "user":
            continue
        if i == 0 or getattr(segments[i - 1], "role", None) != "assistant":
            continue
        call_id = f"call_turn_{turn}"
        shaped[i - 1] = {
            **shaped[i - 1],
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "recorded_tool", "arguments": "{}"},
                }
            ],
        }
        shaped[i] = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": shaped[i].get("content", ""),
        }
    return shaped

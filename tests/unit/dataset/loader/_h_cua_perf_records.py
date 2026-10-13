# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.dataset.loader.h_cua_perf_processing import (
    IMAGE_OMITTED_TEXT,
    screenshot_slots,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }
]


def image(session_id: str, slot: int) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{session_id}-{slot}"},
        "uuid": f"{slot:016x}",
    }


def observation(
    session_id: str, slot: int, *, with_image: bool
) -> list[dict[str, Any]]:
    if with_image:
        return [
            {"type": "text", "text": f"obs {slot} before"},
            image(session_id, slot),
            {"type": "text", "text": f"obs {slot} after"},
        ]
    return [
        {
            "type": "text",
            "text": f"obs {slot} before{IMAGE_OMITTED_TEXT}obs {slot} after",
        }
    ]


def record(
    session_id: str,
    step: int,
    *,
    delay: int | None = None,
    image_slots: set[int] | None = None,
) -> dict[str, Any]:
    """Request at ``step`` with observation slots 0..step; the source keeps only the latest screenshot."""
    image_slots = {step} if image_slots is None else image_slots
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a computer-use agent."},
        {
            "role": "user",
            "content": observation(session_id, 0, with_image=0 in image_slots),
        },
    ]
    for slot in range(1, step + 1):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{slot}",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{slot}",
                "content": observation(
                    session_id, slot, with_image=slot in image_slots
                ),
            }
        )
    out: dict[str, Any] = {
        "session_id": session_id,
        "messages": messages,
        "tools": TOOLS,
        "output_length": 10 + step,
        "extra": {"tool_choice": "auto"},
    }
    if delay is not None:
        out["delay"] = delay
    return out


def trajectory(session_id: str, steps: int) -> list[dict[str, Any]]:
    """Requests of one trajectory; the first carries no delay, the rest the agent's think time."""
    return [
        record(session_id, step, delay=None if step == 0 else 2000)
        for step in range(steps)
    ]


def image_slots(messages: list[dict[str, Any]]) -> list[int]:
    """Slot indexes that currently hold a screenshot rather than the placeholder."""
    return [
        idx
        for idx, (parts, part_idx) in enumerate(screenshot_slots(messages))
        if parts[part_idx].get("type") == "image_url"
    ]

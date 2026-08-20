# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

OUTPUT_REPLAY_ID_ANNOTATION_KEY = "output_replay_id"
OUTPUT_REPLAY_ID_ANNOTATION_PREFIX = f"{OUTPUT_REPLAY_ID_ANNOTATION_KEY}:"


def effective_replay_key(
    request_id: str | None,
    session_id: str | None,
    turn_index: int,
    line_index: int,
) -> str:
    if request_id is not None and request_id.strip():
        return request_id.strip()
    if session_id is not None and session_id.strip():
        return f"{session_id.strip()}:{turn_index}"
    return f"line:{line_index}"


def output_replay_id_annotation(replay_key: str) -> str:
    return f"{OUTPUT_REPLAY_ID_ANNOTATION_PREFIX}{replay_key}"


def merge_output_replay_annotation(
    body: dict[str, Any] | None, replay_key: str | None
) -> dict[str, Any] | None:
    if replay_key is None:
        return body

    merged: dict[str, Any] = deepcopy(body) if body is not None else {}
    nvext = merged.get("nvext")
    if nvext is None:
        nvext = {}
        merged["nvext"] = nvext
    if not isinstance(nvext, dict):
        raise ValueError("nvext must be a dict to attach output replay annotations")

    annotations = nvext.get("annotations")
    if annotations is None:
        annotations = []
        nvext["annotations"] = annotations
    if not isinstance(annotations, list):
        raise ValueError(
            "nvext.annotations must be a list to attach output replay annotations"
        )

    annotation = output_replay_id_annotation(replay_key)
    if annotation not in annotations:
        annotations.append(annotation)

    return merged

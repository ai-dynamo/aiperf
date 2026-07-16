# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared dataset configuration constants."""

BASETEN_ONLY_REPLAY_FIELDS: tuple[str, ...] = (
    "trace_session_sample_ratio",
    "replay_speedup",
    "max_idle_gap_cap_seconds",
    "open_loop_replay",
    "open_loop_strict",
    "omit_kv_hints",
    "force_min_tokens",
)

BASETEN_ONLY_REPLAY_FIELD_FLAGS: tuple[tuple[str, str], ...] = (
    ("trace_session_sample_ratio", "--trace-session-sample-ratio"),
    ("replay_speedup", "--replay-speedup"),
    ("max_idle_gap_cap_seconds", "--max-idle-gap-cap-seconds"),
)

BASETEN_ONLY_REPLAY_BOOL_FIELD_FLAGS: tuple[tuple[str, str], ...] = (
    ("open_loop_replay", "--open-loop-replay/--no-open-loop-replay"),
    ("open_loop_strict", "--open-loop-strict"),
    ("omit_kv_hints", "--omit-kv-hints"),
    ("force_min_tokens", "--force-min-tokens/--no-force-min-tokens"),
)

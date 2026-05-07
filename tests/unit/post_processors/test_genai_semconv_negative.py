# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Negative-case smoke tests for GenAI semconv compliance.

Asserts that AIPerf never emits gen_ai.server.* metrics or any of the four
opt-in GenAI event names. These are explicitly out of scope per
Requirements 14.10 and 14.11.

# Feature: otel-mlflow-telemetry-takeover, Requirement 14.10, 14.11
"""

from __future__ import annotations

import inspect

from aiperf.post_processors.strategies import genai_semconv
from aiperf.post_processors.strategies.genai_semconv import METRIC_NAME_MAP

_FORBIDDEN_EVENT_NAMES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
    }
)


class TestNoServerMetrics:
    """No spec metric name in METRIC_NAME_MAP starts with 'gen_ai.server.'."""

    def test_no_server_metric_in_map_values(self) -> None:
        for aiperf_name, (spec_name, _unit, _buckets) in METRIC_NAME_MAP.items():
            assert not spec_name.startswith("gen_ai.server."), (
                f"METRIC_NAME_MAP[{aiperf_name!r}] maps to server metric "
                f"{spec_name!r} which is out of scope"
            )


class TestNoOptInGenAIEvents:
    """No string constant in the genai_semconv module equals a forbidden event name."""

    def test_no_forbidden_event_name_in_module(self) -> None:
        source = inspect.getsource(genai_semconv)
        for event_name in _FORBIDDEN_EVENT_NAMES:
            assert event_name not in source, (
                f"Forbidden GenAI event name {event_name!r} found in "
                f"genai_semconv module source"
            )

    def test_no_server_prefix_in_module_source(self) -> None:
        source = inspect.getsource(genai_semconv)
        assert "gen_ai.server." not in source, (
            "String 'gen_ai.server.' found in genai_semconv module source; "
            "server-side metrics are out of scope"
        )

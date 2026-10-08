# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fail-fast configuration and compatibility validation of finite replay."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aiperf.common.enums import AgenticReplayLifecycle
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.common.scenario.base import EmptyTracePoolError
from aiperf.config.phases import ConcurrencyPhase
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import (
    DatasetSamplingStrategy,
    PhaseType,
    TimingMode,
    TransportType,
)
from aiperf.timing.config import _validate_finite_replay_compatibility
from aiperf.timing.trajectory_source import TrajectorySource


def test_finite_lifecycle_rejects_non_profiling_phase() -> None:
    with pytest.raises(ValidationError, match="does not support warmup phases"):
        ConcurrencyPhase(
            name="warmup",
            kind="warmup",
            type=PhaseType.CONCURRENCY,
            concurrency=2,
            agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("requests", 10),
        ("duration", 30.0),
        ("sessions", 5),
        ("agentic_cache_warmup_duration", 10.0),
        ("agentic_warmup_grace_period", 5.0),
        ("system_idle_gap_cap_seconds", 1.0),
        ("concurrency_ramp", 1.0),
        ("prefill_ramp", 1.0),
    ],
)
def test_finite_lifecycle_rejects_conflicting_parameters(
    field_name: str, field_value: object
) -> None:
    kwargs = {
        "name": "profiling",
        "kind": "profiling",
        "type": PhaseType.CONCURRENCY,
        "concurrency": 2,
        "agentic_replay_lifecycle": AgenticReplayLifecycle.FINITE,
        field_name: field_value,
    }
    with pytest.raises(ValidationError, match=f"conflicts with {field_name}"):
        ConcurrencyPhase(**kwargs)


def test_finite_lifecycle_rejects_burst_and_seamless() -> None:
    with pytest.raises(
        ValidationError, match="does not support burst_phase_starts or seamless"
    ):
        ConcurrencyPhase(
            name="profiling",
            kind="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=2,
            agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
            burst_phase_starts=True,
        )

    with pytest.raises(
        ValidationError, match="does not support burst_phase_starts or seamless"
    ):
        ConcurrencyPhase(
            name="profiling",
            kind="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=2,
            agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
            seamless=True,
        )


def test_finite_lifecycle_rejects_conflicting_timing_mode() -> None:
    with pytest.raises(ValidationError, match="conflicts with timing_mode"):
        ConcurrencyPhase(
            name="profiling",
            kind="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=2,
            agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
            timing_mode=TimingMode.REQUEST_RATE,
        )


def test_compatibility_rejects_mixed_profiling_lifecycles() -> None:
    p1 = SimpleNamespace(agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE)
    p2 = SimpleNamespace(agentic_replay_lifecycle=AgenticReplayLifecycle.STEADY_STATE)
    cfg = SimpleNamespace()
    with pytest.raises(ValueError, match="same agentic replay lifecycle"):
        _validate_finite_replay_compatibility(cfg, [p1, p2])


def test_compatibility_rejects_warmup_phases() -> None:
    p1 = SimpleNamespace(agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE)
    cfg = SimpleNamespace(get_warmup_phases=lambda: ["warmup_phase"])
    with pytest.raises(ValueError, match="does not support explicit warmup phases"):
        _validate_finite_replay_compatibility(cfg, [p1])


def test_compatibility_rejects_non_http_transport() -> None:
    p1 = SimpleNamespace(
        agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        cancellation=None,
        agentic_warmup_grace_period=None,
    )
    cfg = SimpleNamespace(
        get_warmup_phases=lambda: [],
        endpoint=SimpleNamespace(transport="grpc", type="triton"),
        get_default_dataset=lambda: SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="requires the HTTP transport"):
        _validate_finite_replay_compatibility(cfg, [p1])


def test_compatibility_rejects_polling_endpoints() -> None:
    p1 = SimpleNamespace(
        agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        cancellation=None,
        agentic_warmup_grace_period=None,
    )
    cfg = SimpleNamespace(
        get_warmup_phases=lambda: [],
        endpoint=SimpleNamespace(transport=TransportType.HTTP, type="custom_polling"),
        get_default_dataset=lambda: SimpleNamespace(),
    )
    meta = SimpleNamespace(requires_polling=True)
    with (
        patch("aiperf.plugin.plugins.get_endpoint_metadata", return_value=meta),
        pytest.raises(ValueError, match="does not support polling endpoint"),
    ):
        _validate_finite_replay_compatibility(cfg, [p1])


def test_compatibility_rejects_request_cancellation() -> None:
    p1 = SimpleNamespace(
        agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        cancellation=SimpleNamespace(rate=0.5),
        agentic_warmup_grace_period=None,
    )
    cfg = SimpleNamespace(
        get_warmup_phases=lambda: [],
        endpoint=SimpleNamespace(transport=TransportType.HTTP, type="vllm"),
        get_default_dataset=lambda: SimpleNamespace(),
    )
    meta = SimpleNamespace(requires_polling=False)
    with (
        patch("aiperf.plugin.plugins.get_endpoint_metadata", return_value=meta),
        pytest.raises(ValueError, match="does not support request cancellation"),
    ):
        _validate_finite_replay_compatibility(cfg, [p1])


@pytest.mark.parametrize(
    "compression_field",
    [
        "trace_idle_gap_cap_seconds",
        "inter_turn_delay_cap_seconds",
        "max_idle_gap_cap_seconds",
        "replay_speedup",
    ],
)
def test_compatibility_rejects_dataset_delay_compression(
    compression_field: str,
) -> None:
    p1 = SimpleNamespace(
        agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        cancellation=None,
        agentic_warmup_grace_period=None,
    )
    dataset = SimpleNamespace(**{compression_field: 1.5})
    cfg = SimpleNamespace(
        get_warmup_phases=lambda: [],
        endpoint=SimpleNamespace(transport=TransportType.HTTP, type="vllm"),
        get_default_dataset=lambda: dataset,
    )
    meta = SimpleNamespace(requires_polling=False)
    with (
        patch("aiperf.plugin.plugins.get_endpoint_metadata", return_value=meta),
        pytest.raises(ValueError, match="delay compression"),
    ):
        _validate_finite_replay_compatibility(cfg, [p1])


def test_compatibility_rejects_agentic_warmup_grace_period() -> None:
    p1 = SimpleNamespace(
        agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        cancellation=None,
        agentic_warmup_grace_period=5.0,
    )
    cfg = SimpleNamespace(
        get_warmup_phases=lambda: [],
        endpoint=SimpleNamespace(transport=TransportType.HTTP, type="vllm"),
        get_default_dataset=lambda: SimpleNamespace(),
    )
    meta = SimpleNamespace(requires_polling=False)
    with (
        patch("aiperf.plugin.plugins.get_endpoint_metadata", return_value=meta),
        pytest.raises(
            ValueError, match="does not support agentic warmup grace settings"
        ),
    ):
        _validate_finite_replay_compatibility(cfg, [p1])


def test_compatibility_rejects_ignore_trace_delays() -> None:
    p1 = SimpleNamespace(
        agentic_replay_lifecycle=AgenticReplayLifecycle.FINITE,
        cancellation=None,
        agentic_warmup_grace_period=None,
    )
    dataset = SimpleNamespace(ignore_trace_delays=True)
    cfg = SimpleNamespace(
        get_warmup_phases=lambda: [],
        endpoint=SimpleNamespace(transport=TransportType.HTTP, type="vllm"),
        get_default_dataset=lambda: dataset,
    )
    meta = SimpleNamespace(requires_polling=False)
    with (
        patch("aiperf.plugin.plugins.get_endpoint_metadata", return_value=meta),
        pytest.raises(ValueError, match="requires authored absolute trace timestamps"),
    ):
        _validate_finite_replay_compatibility(cfg, [p1])


def test_trajectory_source_rejects_allow_dataset_wrap() -> None:
    metadata = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="root",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                is_root=True,
            )
        ],
    )
    with pytest.raises(ValueError, match="does not support dataset wrapping"):
        TrajectorySource(
            dataset_metadata=metadata,
            dataset_sampler=SequentialSampler(["root"]),
            concurrency=1,
            random_seed=42,
            allow_dataset_wrap=True,
            finite_replay=True,
        )


def test_trajectory_source_rejects_empty_root_pool() -> None:
    metadata = DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="child-only",
                turns=[TurnMetadata(timestamp_ms=0.0)],
                is_root=False,
                agent_depth=1,
            )
        ],
    )
    with pytest.raises(EmptyTracePoolError, match="no eligible root traces"):
        TrajectorySource(
            dataset_metadata=metadata,
            dataset_sampler=SequentialSampler(["child-only"]),
            concurrency=1,
            random_seed=42,
            allow_dataset_wrap=False,
            finite_replay=True,
        )

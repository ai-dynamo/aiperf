# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the worker-pod lifecycle messages and their service exceptions.

These messages carry Kubernetes pod-level worker state from a worker pod's
manager back to the SystemController, plus the real-time server-metrics
fan-out consumed by the ``/api/server-metrics`` router.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import (
    MessageType,
    WorkerStartupState,
    WorkerStatus,
)
from aiperf.common.exceptions import (
    AIPerfError,
    ServiceProcessDiedError,
    ServiceRegistrationTimeoutError,
)
from aiperf.common.messages import (
    RealtimeServerMetricsMessage,
    WorkerGroupStatsMessage,
    WorkerPodStateMessage,
    WorkerStartupStateMessage,
)
from aiperf.common.models import WorkerTaskStats


def make_pod_state(**overrides: object) -> WorkerPodStateMessage:
    """Construct a minimally-valid WorkerPodStateMessage."""
    kwargs: dict[str, object] = {
        "service_id": "worker-manager-0",
        "pod_index": "0",
        "declared_workers": 4,
        "declared_record_processors": 2,
        "pod_state": "ready",
        "admission_state": "admitted",
    }
    kwargs.update(overrides)
    return WorkerPodStateMessage(**kwargs)  # type: ignore[arg-type]


class TestWorkerPodStateMessage:
    def test_carries_message_type(self) -> None:
        assert make_pod_state().message_type == MessageType.WORKER_POD_STATE

    def test_counters_default_to_zero(self) -> None:
        msg = make_pod_state()
        assert msg.ready_workers == 0
        assert msg.ready_record_processors == 0
        assert msg.router_connected_workers == 0
        assert msg.dispatchable_workers == 0
        assert msg.degraded_workers == 0
        assert msg.degraded_record_processors == 0
        assert msg.benchmark_generation is None
        assert msg.dataset_generation is None

    def test_roundtrips_through_json(self) -> None:
        msg = make_pod_state(ready_workers=3, benchmark_generation="gen-7")
        restored = WorkerPodStateMessage.model_validate_json(msg.model_dump_json())
        assert restored.ready_workers == 3
        assert restored.benchmark_generation == "gen-7"
        assert restored.pod_index == "0"


class TestWorkerStartupStateMessage:
    def test_carries_message_type_and_state(self) -> None:
        msg = WorkerStartupStateMessage(
            service_id="worker-0", startup_state=WorkerStartupState.READY
        )
        assert msg.message_type == MessageType.WORKER_STARTUP_STATE
        assert msg.startup_state == WorkerStartupState.READY

    def test_request_ns_is_auto_populated(self) -> None:
        msg = WorkerStartupStateMessage(
            service_id="worker-0", startup_state=WorkerStartupState.READY
        )
        assert msg.request_ns > 0

    def test_roundtrips(self) -> None:
        msg = WorkerStartupStateMessage(
            service_id="worker-0", startup_state=WorkerStartupState.READY
        )
        assert WorkerStartupState.READY in msg.model_dump_json()


class TestWorkerGroupStatsMessage:
    def test_carries_message_type_and_defaults(self) -> None:
        msg = WorkerGroupStatsMessage(
            service_id="worker-manager-0",
            group_id="group-0",
            status=WorkerStatus.HEALTHY,
            task_stats=WorkerTaskStats(),
        )
        assert msg.message_type == MessageType.WORKER_GROUP_STATS
        assert msg.worker_statuses == {}
        assert msg.worker_startup_states == {}
        assert msg.worker_task_stats == {}
        assert msg.worker_health == {}
        assert msg.health is None
        assert msg.startup_state is None
        assert msg.last_update_ns > 0

    def test_per_worker_maps_roundtrip(self) -> None:
        msg = WorkerGroupStatsMessage(
            service_id="worker-manager-0",
            group_id="group-0",
            status=WorkerStatus.HEALTHY,
            task_stats=WorkerTaskStats(),
            worker_statuses={"worker-0": WorkerStatus.HEALTHY},
        )
        restored = WorkerGroupStatsMessage.model_validate_json(msg.model_dump_json())
        assert restored.worker_statuses == {"worker-0": WorkerStatus.HEALTHY}


class TestRealtimeServerMetricsMessage:
    def test_carries_message_type(self) -> None:
        msg = RealtimeServerMetricsMessage(
            service_id="server-metrics-manager", endpoint_summaries={}
        )
        assert msg.message_type == MessageType.REALTIME_SERVER_METRICS
        assert msg.endpoint_summaries == {}


class TestServiceLifecycleExceptions:
    def test_process_died_error_names_service_and_exit_code(self) -> None:
        err = ServiceProcessDiedError(
            service_id="worker-0", service_type="worker", exit_code=-9
        )
        text = str(err)
        assert "worker-0" in text
        assert "worker" in text
        assert "-9" in text
        assert err.exit_code == -9
        assert isinstance(err, AIPerfError)

    def test_process_died_error_without_exit_code_still_explains(self) -> None:
        err = ServiceProcessDiedError(service_id="worker-0", service_type="worker")
        assert "worker-0" in str(err)
        assert err.exit_code is None

    def test_registration_timeout_names_counts_and_missing(self) -> None:
        err = ServiceRegistrationTimeoutError(
            registered=3, expected=5, timeout_sec=60.0, missing={"worker": 2}
        )
        text = str(err)
        assert "3" in text
        assert "5" in text
        assert "worker" in text
        assert err.missing == {"worker": 2}

    def test_registration_timeout_is_both_aiperf_error_and_timeout_error(self) -> None:
        err = ServiceRegistrationTimeoutError(
            registered=0, expected=1, timeout_sec=1.0, missing={"worker": 1}
        )
        assert isinstance(err, AIPerfError)
        assert isinstance(err, TimeoutError)

    def test_registration_timeout_is_catchable_as_timeout_error(self) -> None:
        with pytest.raises(TimeoutError):
            raise ServiceRegistrationTimeoutError(
                registered=0, expected=2, timeout_sec=5.0, missing={"worker": 2}
            )

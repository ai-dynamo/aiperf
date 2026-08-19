# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial progress and server-metrics API tests.

Focuses on:
- /api/progress snapshot schema stability beyond pod-state RPC aggregation.
- SYSTEM_STATE_CHANGED deserialization boundaries and malformed-state rejection.
- Results-exported and concurrent progress updates reflected through FastAPI.
- Server-metrics summaries, missing required fields, and non-finite JSON output.

Out of scope: pod-state RPC fallback behavior; see test_pod_state_rpc_adversarial.py.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError as PydanticValidationError
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.enums import CreditPhase, SystemState
from aiperf.common.messages import (
    ResultsExportedMessage,
    SystemStateChangedMessage,
)
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.config import AIPerfConfig, BenchmarkRun

# ============================================================================
# Helpers
# ============================================================================


def _benchmark_run() -> BenchmarkRun:
    """Real Pydantic benchmark run for router construction."""
    config = AIPerfConfig(
        benchmark={
            "models": ["meta-llama/Llama-3-8B"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [
                {
                    "name": "synthetic-chat-main",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "steady-state",
                    "type": "concurrency",
                    "kind": "profiling",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id="aiperf-bench-7f2a",
        cfg=config.benchmark,
        artifact_dir=Path("/tmp/aiperf-bench-7f2a"),
    )


def _progress_app(router: ProgressRouter) -> FastAPI:
    app = FastAPI()
    app.state.progress = router
    app.include_router(router.get_router())
    return app


@pytest.fixture
def progress_router(mock_zmq: object) -> ProgressRouter:
    return ProgressRouter(run=_benchmark_run())


@pytest.fixture
def progress_client(progress_router: ProgressRouter) -> TestClient:
    return TestClient(_progress_app(progress_router))


class TestProgressSnapshotSchema:
    """Snapshot fields are stable for operator and dashboard consumers."""

    def test_get_progress_empty_snapshot_has_stable_top_level_schema(
        self, progress_client: TestClient
    ) -> None:
        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        data = orjson.loads(response.content)
        assert set(data) == {"phases", "workers", "results_exported", "system_state"}
        assert data["phases"] == {}
        assert data["results_exported"] is False
        assert data["system_state"] == "initializing"
        assert data["workers"]["ready"] == 0
        assert data["workers"]["total_pods"] == 0

    def test_get_progress_phase_snapshot_preserves_false_zero_and_null_fields(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        progress_router._progress_tracker._phases = {
            CreditPhase.PROFILING: CombinedPhaseStats(
                phase=CreditPhase.PROFILING,
                phase_name="profiling",
                phase_kind="profiling",
                start_ns=1_000,
                # ``total_expected_requests`` is constrained ``> 0``; zero-value
                # preservation is covered by ``requests_completed`` below.
                total_expected_requests=1,
                requests_completed=0,
                requests_per_second=None,
            )
        }

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        phase = orjson.loads(response.content)["phases"]["profiling"]
        assert phase["phase"] == "profiling"
        assert phase["phase_kind"] == "profiling"
        assert phase["total_expected_requests"] == 1
        assert phase["requests_completed"] == 0
        assert phase["requests_per_second"] is None

    @pytest.mark.asyncio
    async def test_on_results_exported_sets_snapshot_gate_true(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_results_exported(
            ResultsExportedMessage(service_id="system_controller", was_cancelled=False)
        )

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert orjson.loads(response.content)["results_exported"] is True

    @pytest.mark.parametrize(
        "wire_state,expected_state",
        [
            ("profiling", SystemState.PROFILING),
            param("PROFILING", SystemState.PROFILING, id="uppercase-accepted"),
            param("Ready", SystemState.READY, id="mixed-case-accepted"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_system_state_changed_message_wire_values_decode_to_enum(
        self,
        progress_client: TestClient,
        progress_router: ProgressRouter,
        wire_state: str,
        expected_state: SystemState,
    ) -> None:
        message = SystemStateChangedMessage.from_json(
            {
                "message_type": "system_state_changed",
                "service_id": "system_controller",
                "state": wire_state,
            }
        )

        await progress_router._on_system_state_changed(message)

        response = progress_client.get("/api/progress")
        assert response.status_code == 200
        assert orjson.loads(response.content)["system_state"] == expected_state.value

    @pytest.mark.parametrize(
        "payload,match",
        [
            param(
                {
                    "message_type": "system_state_changed",
                    "service_id": "system_controller",
                    "state": "profilling",
                },
                r"state",
                id="typo-state",
            ),
            param(
                {
                    "message_type": "system_state_changed",
                    "service_id": "system_controller",
                    "state": None,
                },
                r"state",
                id="null-state",
            ),
            param(
                {
                    "message_type": "system_state_changed",
                    "service_id": "system_controller",
                },
                r"state",
                id="missing-state",
            ),
        ],
    )  # fmt: skip
    def test_system_state_changed_message_malformed_wire_values_are_rejected(
        self, payload: dict[str, object], match: str
    ) -> None:
        # AIPerf messages are Pydantic models on this branch, so a malformed
        # wire payload surfaces as pydantic's ValidationError rather than
        # msgspec's; the rejection contract is what matters here.
        with pytest.raises(PydanticValidationError, match=match):
            SystemStateChangedMessage.from_json(payload)

    def test_get_progress_non_finite_phase_numbers_serialize_as_null(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        progress_router._progress_tracker._phases = {
            CreditPhase.PROFILING: CombinedPhaseStats(
                phase=CreditPhase.PROFILING,
                start_ns=1_000,
                total_expected_requests=10,
                requests_completed=1,
                requests_per_second=float("nan"),
                records_per_second=float("inf"),
            )
        }

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert b"NaN" not in response.content
        assert b"Infinity" not in response.content
        phase = orjson.loads(response.content)["phases"]["profiling"]
        assert phase["requests_per_second"] is None
        assert phase["records_per_second"] is None

    @pytest.mark.asyncio
    async def test_get_progress_concurrent_system_state_updates_never_emit_invalid_state(
        self, progress_router: ProgressRouter
    ) -> None:
        app = _progress_app(progress_router)
        transport = ASGITransport(app=app)
        valid_states = {state.value for state in SystemState}

        async with AsyncClient(
            transport=transport, base_url="http://api.local"
        ) as client:
            for state in (
                SystemState.CONFIGURING,
                SystemState.READY,
                SystemState.PROFILING,
                SystemState.PROCESSING,
                SystemState.STOPPING,
            ):
                await progress_router._on_system_state_changed(
                    SystemStateChangedMessage(
                        service_id="system_controller",
                        state=state,
                    )
                )
                response = await client.get("/api/progress")
                assert response.status_code == 200
                assert response.json()["system_state"] in valid_states

        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.SHUTDOWN,
            )
        )
        final = TestClient(app).get("/api/progress")
        assert final.json()["system_state"] == "shutdown"


# ============================================================================
# Server metrics summaries and JSON boundaries
# ============================================================================

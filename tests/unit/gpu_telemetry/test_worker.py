# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aiperf.common.models import GpuMetadata, TelemetryMetrics, TelemetryRecord
from aiperf.gpu_telemetry.worker import TelemetryWorker


class _FixtureCollector:
    instance: _FixtureCollector | None = None

    def __init__(self, *, record_callback, error_callback, **_kwargs) -> None:
        self.record_callback = record_callback
        self.error_callback = error_callback
        self.endpoint_url = "fixture://localhost"
        self.initialized = False
        self.stopped = False
        type(self).instance = self

    async def is_url_reachable(self) -> bool:
        return True

    async def initialize(self) -> None:
        self.initialized = True

    async def collect_and_process_metrics(self) -> None:
        await self.record_callback(
            [
                TelemetryRecord(
                    timestamp_ns=123,
                    dcgm_url=self.endpoint_url,
                    **GpuMetadata(
                        gpu_index=0,
                        gpu_uuid="GPU-fixture",
                        gpu_model_name="Fixture GPU",
                        hostname="node",
                    ).model_dump(),
                    telemetry_data=TelemetryMetrics(
                        gpu_power_usage=250.0,
                        energy_consumption=2.0,
                    ),
                )
            ],
            "fixture",
        )

    async def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_worker_adapts_registered_collectors_without_owning_cadence(
    monkeypatch,
) -> None:
    from aiperf.gpu_telemetry import worker as worker_module

    monkeypatch.setattr(
        worker_module.plugins,
        "get_class",
        lambda *_args, **_kwargs: _FixtureCollector,
    )
    worker = TelemetryWorker()

    configured = await worker.configure(
        {"collector": "fixture", "request_timeout_seconds": 1.0}
    )
    scraped = await worker.scrape({"boundary": True})

    assert configured == {
        "endpoint_url": "fixture://localhost",
        "reachable": True,
        "reason": None,
    }
    assert scraped["duplicate"] is False
    assert scraped["records"] == [
        {
            "gpu_index": 0,
            "gpu_uuid": "GPU-fixture",
            "gpu_model_name": "Fixture GPU",
            "hostname": "node",
            "dcgm_url": "fixture://localhost",
            "telemetry_data": {
                "gpu_power_usage": 250.0,
                "energy_consumption": 2.0,
            },
        }
    ]
    assert "timestamp_ns" not in scraped["records"][0]

    assert await worker.shutdown() == {"shutdown": True}
    assert _FixtureCollector.instance is not None
    assert _FixtureCollector.instance.initialized is True
    assert _FixtureCollector.instance.stopped is True


def test_worker_process_negotiates_strict_json_lines() -> None:
    requests = "\n".join(
        [
            json.dumps({"id": 1, "op": "hello", "protocol": 1}),
            json.dumps({"id": 2, "op": "shutdown"}),
            "",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-u", "-m", "aiperf.gpu_telemetry.worker"],
        input=requests,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == 2
    assert responses[0]["id"] == 1
    assert responses[0]["ok"] is True
    assert responses[0]["result"]["protocol"] == 1
    assert responses[0]["result"]["capabilities"] == [
        "configure",
        "scrape",
        "shutdown",
    ]
    assert responses[1] == {
        "id": 2,
        "ok": True,
        "result": {"shutdown": True},
    }

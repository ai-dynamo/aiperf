# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict stdio adapter for canonical Python GPU telemetry collectors.

Rust owns phase barriers, scrape cadence, the benchmark Clock, accumulation,
and result generation. This worker owns only Python implementations that are
already canonical for local GPU bindings, custom DCGM fields, or registered
user collectors. One process serves exactly one source and writes exactly one
JSON response line for every JSON request line.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.common.redact import redact_url
from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

PROTOCOL_VERSION = 1
WORKER_VERSION = "1.0"
CAPABILITIES = ["configure", "scrape", "shutdown"]


class TelemetryWorker:
    """Own one configured collector while Rust drives every scrape."""

    def __init__(self) -> None:
        self.collector: Any | None = None
        self.endpoint_url: str | None = None
        self._records: list[TelemetryRecord] = []
        self._errors: list[str] = []

    async def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        """Resolve, probe, and initialize one registered collector."""
        if self.collector is not None:
            raise RuntimeError("collector is already configured")
        collector_name = _required_string(request, "collector")
        metrics_file = request.get("metrics_file")
        if metrics_file is not None:
            if not isinstance(metrics_file, str) or not metrics_file:
                raise ValueError("metrics_file must be a non-empty string")
            _install_custom_metrics(Path(metrics_file))

        collector_class = plugins.get_class(
            PluginType.GPU_TELEMETRY_COLLECTOR,
            collector_name,
        )
        kwargs: dict[str, Any] = {
            "collection_interval": 3600.0,
            "record_callback": self._on_records,
            "error_callback": self._on_error,
            "collector_id": f"rust_{collector_name}_collector",
        }
        url = request.get("url")
        if url is not None:
            if not isinstance(url, str) or not url:
                raise ValueError("url must be a non-empty string")
            if issubclass(collector_class, DCGMTelemetryCollector):
                url = url.rstrip("/")
                if not url.endswith("/metrics"):
                    url = f"{url}/metrics"
            kwargs["dcgm_url"] = url
            if issubclass(collector_class, DCGMTelemetryCollector):
                timeout = request.get("request_timeout_seconds")
                if not isinstance(timeout, int | float) or isinstance(timeout, bool):
                    raise ValueError("request_timeout_seconds must be numeric")
                if not math.isfinite(float(timeout)) or timeout <= 0:
                    raise ValueError("request_timeout_seconds must be finite and positive")
                kwargs["reachability_timeout"] = float(timeout)

        collector = collector_class(**kwargs)
        endpoint_url = redact_url(str(collector.endpoint_url))
        reachable = await collector.is_url_reachable()
        if not reachable:
            return {
                "endpoint_url": endpoint_url,
                "reachable": False,
                "reason": f"{collector_name} source is unavailable",
            }

        await collector.initialize()
        self.collector = collector
        self.endpoint_url = endpoint_url
        return {
            "endpoint_url": endpoint_url,
            "reachable": True,
            "reason": None,
        }

    async def scrape(self, request: dict[str, Any]) -> dict[str, Any]:
        """Collect one source snapshot; boundary scrapes bypass DCGM dedup."""
        collector = self.collector
        endpoint_url = self.endpoint_url
        if collector is None or endpoint_url is None:
            raise RuntimeError("collector is not configured")
        boundary = request.get("boundary")
        if not isinstance(boundary, bool):
            raise ValueError("boundary must be a boolean")

        self._records.clear()
        self._errors.clear()
        duplicate = False
        if isinstance(collector, DCGMTelemetryCollector):
            records, duplicate = await collector.collect_records_once(
                bypass_dedup=boundary
            )
            self._records.extend(records)
        else:
            await collector.collect_and_process_metrics()

        if self._errors:
            raise RuntimeError("; ".join(self._errors))
        return {
            "endpoint_url": endpoint_url,
            "duplicate": duplicate and not boundary,
            "records": [_record_to_wire(record) for record in self._records],
        }

    async def shutdown(self) -> dict[str, bool]:
        """Run the collector's canonical lifecycle cleanup exactly once."""
        collector, self.collector = self.collector, None
        if collector is not None:
            await collector.stop()
        return {"shutdown": True}

    async def _on_records(
        self,
        records: list[TelemetryRecord],
        _collector_id: str,
    ) -> None:
        self._records.extend(records)

    async def _on_error(self, error: ErrorDetails, _collector_id: str) -> None:
        self._errors.append(str(error))


def _install_custom_metrics(path: Path) -> None:
    """Apply the same process-local custom-field mutation as service bootstrap."""
    if not path.is_file():
        raise FileNotFoundError(f"custom GPU metrics file not found: {path}")
    from aiperf.gpu_telemetry import constants
    from aiperf.gpu_telemetry.metrics_config import MetricsConfigLoader

    custom_metrics, mappings = MetricsConfigLoader().build_custom_metrics_from_csv(path)
    existing = {name for _, name, _ in constants.GPU_TELEMETRY_METRICS_CONFIG}
    constants.GPU_TELEMETRY_METRICS_CONFIG.extend(
        metric for metric in custom_metrics if metric[1] not in existing
    )
    constants.DCGM_TO_FIELD_MAPPING.update(mappings)


def _record_to_wire(record: TelemetryRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json", exclude_none=True)
    data.pop("timestamp_ns", None)
    data["dcgm_url"] = redact_url(str(data["dcgm_url"]))
    telemetry = data.get("telemetry_data", {})
    data["telemetry_data"] = {
        name: float(value)
        for name, value in telemetry.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }
    return data


def _required_string(request: dict[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


async def _dispatch(worker: TelemetryWorker, request: dict[str, Any]) -> Any:
    op = request.get("op")
    if op == "hello":
        protocol = request.get("protocol")
        if protocol != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported protocol {protocol!r}; expected {PROTOCOL_VERSION}"
            )
        return {
            "protocol": PROTOCOL_VERSION,
            "worker_version": WORKER_VERSION,
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "capabilities": CAPABILITIES,
        }
    if op == "configure":
        return await worker.configure(request)
    if op == "scrape":
        return await worker.scrape(request)
    if op == "shutdown":
        return await worker.shutdown()
    raise ValueError(f"unsupported operation {op!r}")


async def _main() -> int:
    worker = TelemetryWorker()
    while True:
        # The protocol is strictly request/response: no collector work is
        # active while waiting for the next command, so a blocking pipe read
        # is both simpler and portable across event-loop implementations.
        line = sys.stdin.buffer.readline()
        if not line:
            await worker.shutdown()
            return 0
        request_id: int | None = None
        shutdown_requested = False
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            if not isinstance(request_id, int) or isinstance(request_id, bool):
                raise ValueError("request id must be an integer")
            shutdown_requested = request.get("op") == "shutdown"
            result = await _dispatch(worker, request)
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {
                "id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        sys.stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")))
        sys.stdout.write("\n")
        sys.stdout.flush()
        if shutdown_requested:
            return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""npu-smi-based Ascend NPU telemetry collector.

Collects NPU metrics from Huawei Ascend devices (910B/910C training cards,
310P/300I inference cards, ...) by shelling out to ``npu-smi``, the device
management CLI shipped with the CANN driver.

Only the machine-readable ``npu-smi info -t <type> -i <id>`` subcommands are
parsed. The human-facing ``npu-smi info`` summary table changes layout between
product lines and driver releases (column order, extra counters) and is used
solely as a last-resort device enumeration fallback.

npu-smi exposes no cumulative energy counter, so this collector deliberately
emits no ``ascend_*`` energy field: ``EnergyEfficiencyAnalyzer`` integrates
``ascend_power`` over the profiling window instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    GpuMetadata,
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.gpu_telemetry.constants import (
    ASCEND_GPU_TELEMETRY_PLATFORM,
    NPU_SMI_SOURCE_IDENTIFIER,
)
from aiperf.gpu_telemetry.protocols import TErrorCallback, TRecordCallback

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["NPUSMITelemetryCollector"]

# One `npu-smi` process per NPU per scrape, so the requested collection
# interval is floored here: at the GPU-wide default of 333 ms an 8-card host
# would fork 24 short-lived processes per second for data that the driver only
# refreshes about once a second anyway.
MIN_COLLECTION_INTERVAL = 1.0

# A single scrape shells out per card; a hung driver must not wedge the
# telemetry service, so every invocation is bounded.
SCRAPE_TIMEOUT_SECONDS = 5.0
DISCOVERY_TIMEOUT_SECONDS = 10.0

# `npu-smi info -t <type>` prints `Key : Value` lines. Key spellings differ
# across product lines and driver releases, so each signal is resolved by
# trying the known prefixes in order (most specific first).
_KEY_LINE_RE = re.compile(r"^(?P<key>[^:]+?)\s*:\s*(?P<value>.*)$")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

_POWER_KEYS = ("NPU Real-time Power", "NPU Power", "Power")
_TEMPERATURE_KEYS = ("NPU Temperature", "Temperature", "HBM Temperature")
_AICORE_KEYS = ("Aicore Usage Rate", "AI Core Usage Rate")
_MEMORY_RATE_KEYS = ("HBM Usage Rate", "Memory Usage Rate")
_HBM_CAPACITY_KEYS = ("HBM Capacity",)
_HEALTH_KEYS = ("Chip Health", "Health")

# Names that contain "Power" but report a rated/limit value rather than the
# instantaneous draw. `-t common` reports both, and taking the rating as the
# live draw would inflate every power-derived metric.
_POWER_LIMIT_KEY_MARKERS = ("rated", "dissipation", "limit", "max", "cap")

# `npu-smi info` summary table row: `| 0     910B3  | OK  | 92.9  37  ...`. Used
# only to enumerate card IDs when `npu-smi info -m` is unavailable.
_TABLE_ROW_RE = re.compile(r"^\|\s*(\d+)\s+\S+")


class NpuSmiError(RuntimeError):
    """Raised when an ``npu-smi`` invocation fails, times out, or is missing."""


async def _run_npu_smi(*args: str, timeout: float) -> str:
    """Run ``npu-smi`` with ``args`` and return stdout.

    Args:
        *args: Arguments passed verbatim to the ``npu-smi`` binary.
        timeout: Seconds to wait before killing the process.

    Returns:
        The decoded standard output.

    Raises:
        NpuSmiError: If the binary is missing, cannot be executed, times out,
            or exits non-zero.
    """
    if shutil.which("npu-smi") is None:
        raise NpuSmiError("npu-smi not found on PATH")
    try:
        process = await asyncio.create_subprocess_exec(
            "npu-smi",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        raise NpuSmiError(f"failed to launch npu-smi: {e}") from e

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as e:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise NpuSmiError(
            f"'npu-smi {' '.join(args)}' timed out after {timeout}s"
        ) from e

    if process.returncode != 0:
        detail = (stderr or stdout).decode(errors="replace").strip()
        raise NpuSmiError(
            f"'npu-smi {' '.join(args)}' failed (rc={process.returncode}): "
            f"{detail[:200]}"
        )
    return stdout.decode(errors="replace")


def _parse_key_value_lines(text: str) -> list[tuple[str, str]]:
    """Split ``Key : Value`` output into ordered pairs, ignoring other lines."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = _KEY_LINE_RE.match(line.strip())
        if match:
            pairs.append((match.group("key").strip(), match.group("value").strip()))
    return pairs


def _lookup(pairs: Sequence[tuple[str, str]], patterns: Sequence[str]) -> str | None:
    """Return the first value whose key starts with one of ``patterns``.

    Patterns are tried in order so the most specific spelling wins, and a
    ``Power``-prefixed key that actually reports a rated or limit value is
    skipped rather than mistaken for the instantaneous draw.
    """
    for pattern in patterns:
        for key, value in pairs:
            if not key.startswith(pattern):
                continue
            if pattern == "Power" and any(
                marker in key.lower() for marker in _POWER_LIMIT_KEY_MARKERS
            ):
                continue
            return value
    return None


def _to_float(value: str | None) -> float | None:
    """Extract a finite number from a value that may carry units or a sentinel.

    ``npu-smi`` reports values such as ``"92.9"``, ``"37 C"``, ``"45%"``,
    ``"65536 MB"``, and ``"N/A"``. Anything without a number becomes ``None``
    rather than zero, so an unsupported sensor is absent instead of misreported.
    """
    if value is None or isinstance(value, bool):
        return None
    match = _NUMBER_RE.search(str(value))
    if match is None:
        return None
    number = float(match.group(0))
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


@dataclass(slots=True)
class _NpuDeviceState:
    """Per-NPU state for npu-smi telemetry collection.

    Args:
        npu_id: Card ID used with ``npu-smi -i``.
        metadata: Static GPU metadata emitted with every record.
        hbm_capacity_mb: HBM capacity in MiB, read once at init. ``None`` when
            the driver reports no capacity, in which case memory usage rate is
            still emitted but byte-level usage is not.
    """

    npu_id: int
    metadata: GpuMetadata
    hbm_capacity_mb: float | None = None


class NPUSMITelemetryCollector(AIPerfLifecycleMixin):
    """Collects Ascend NPU telemetry by invoking the ``npu-smi`` CLI.

    Direct collector for Huawei Ascend devices, the Ascend counterpart to the
    pynvml (NVIDIA) and amdsmi (AMD) collectors. Requires only the CANN driver;
    no HTTP exporter, CANN toolkit, or Python binding is needed.

    Features:
        - Automatic NPU discovery from ``npu-smi info -m``
        - One ``npu-smi info -t common`` invocation per card per scrape, run
          concurrently across cards
        - Layered fallback to ``-t usages`` / ``-t power`` / ``-t temp`` on
          drivers that do not implement ``-t common``
        - Same TelemetryRecord output format as the other collectors
        - Tolerant of partial sensor support: unsupported signals are absent
          from the record rather than reported as zero

    Requirements:
        - ``npu-smi`` on PATH (installed with the Ascend driver; in containers
          the Ascend container runtime must expose ``/dev/davinci*``)
        - At least one visible Ascend NPU

    Args:
        collection_interval: Requested interval in seconds between scrapes.
            Floored at :data:`MIN_COLLECTION_INTERVAL` because each scrape
            forks one process per card.
        record_callback: Async callback invoked with collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Async callback invoked on collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    @classmethod
    def validate_environment(cls) -> None:
        """Verify that the ``npu-smi`` CLI is available before config succeeds."""
        if shutil.which("npu-smi") is None:
            raise RuntimeError(
                "npu-smi not found on PATH. The npu-smi CLI ships with the Ascend "
                "driver (CANN); install the driver, or run inside a container that "
                "mounts /dev/davinci* and includes /usr/local/bin/npu-smi."
            )

    def __init__(
        self,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        collector_id: str = "npu_smi_collector",
    ) -> None:
        super().__init__(id=collector_id)
        self._requested_interval = collection_interval
        self._collection_interval = max(collection_interval, MIN_COLLECTION_INTERVAL)
        self._record_callback = record_callback
        self._error_callback = error_callback

        self._gpus: list[_NpuDeviceState] = []
        self._initialized = False
        # Probed once at init: whether `-t common` is implemented on this driver.
        self._common_query_supported = True

    @property
    def endpoint_url(self) -> str:
        """Source identifier for this collector ('npu-smi://localhost')."""
        return NPU_SMI_SOURCE_IDENTIFIER

    @property
    def collection_interval(self) -> float:
        """Collection interval in seconds, floored for subprocess cost."""
        return self._collection_interval

    async def is_url_reachable(self) -> bool:
        """Check whether ``npu-smi`` is present and enumerates at least one NPU.

        Returns:
            True if an Ascend NPU is visible, False otherwise.
        """
        if self._initialized:
            return bool(self._gpus)
        try:
            return bool(await self._discover_devices())
        except Exception:  # reachability probe must never raise
            return False

    @on_init
    async def _initialize_npu_smi(self) -> None:
        """Discover NPUs, read static HBM capacity, and probe ``-t common``.

        Raises:
            RuntimeError: If ``npu-smi`` is missing, fails, or reports no NPU.
        """
        try:
            devices = await self._discover_devices()
        except NpuSmiError as e:
            raise RuntimeError(f"Failed to enumerate Ascend NPUs: {e}") from e

        if not devices:
            raise RuntimeError("No Ascend NPUs detected via npu-smi")

        if self._collection_interval != self._requested_interval:
            self.info(
                f"npu-smi collector: collection interval raised from "
                f"{self._requested_interval}s to {self._collection_interval}s "
                f"(one npu-smi process per NPU per scrape)"
            )

        self._common_query_supported = await self._probe_common_query(devices[0][0])

        capacities = await asyncio.gather(
            *(self._read_hbm_capacity_mb(npu_id) for npu_id, _ in devices)
        )
        self._gpus = [
            self._build_device_state(npu_id, chip_name, capacity_mb)
            for (npu_id, chip_name), capacity_mb in zip(
                devices, capacities, strict=True
            )
        ]
        self._initialized = True
        self.info(
            f"npu-smi initialized with {len(self._gpus)} Ascend NPU(s): "
            f"{', '.join(gpu.metadata.gpu_model_name for gpu in self._gpus)}"
        )

    async def _discover_devices(self) -> list[tuple[int, str]]:
        """Return ``(npu_id, chip_name)`` for each AI chip reported by npu-smi.

        ``npu-smi info -m`` lists one row per chip; rows whose Chip Logic ID is
        ``-`` describe the MCU management chip and are skipped, as are repeated
        card IDs (a card contributes one telemetry record).
        """
        text = await _run_npu_smi("info", "-m", timeout=DISCOVERY_TIMEOUT_SECONDS)
        devices: list[tuple[int, str]] = []
        seen: set[int] = set()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                npu_id = int(parts[0])
            except ValueError:
                continue
            if parts[2] == "-" or npu_id in seen:
                continue
            seen.add(npu_id)
            devices.append((npu_id, " ".join(parts[3:])))
        if devices:
            return devices

        self.warning(
            "npu-smi info -m produced no parseable mapping rows; falling back to "
            "the npu-smi info summary table for device discovery"
        )
        return await self._discover_devices_from_table()

    async def _discover_devices_from_table(self) -> list[tuple[int, str]]:
        """Last-resort device enumeration from the human-facing summary table."""
        text = await _run_npu_smi("info", timeout=DISCOVERY_TIMEOUT_SECONDS)
        npu_ids: list[int] = []
        for line in text.splitlines():
            match = _TABLE_ROW_RE.match(line)
            if match is None:
                continue
            npu_id = int(match.group(1))
            if npu_id not in npu_ids:
                npu_ids.append(npu_id)
        return [(npu_id, "Ascend NPU") for npu_id in npu_ids]

    async def _probe_common_query(self, npu_id: int) -> bool:
        """Whether this driver implements ``npu-smi info -t common``."""
        try:
            await _run_npu_smi(
                "info",
                "-t",
                "common",
                "-i",
                str(npu_id),
                timeout=DISCOVERY_TIMEOUT_SECONDS,
            )
        except NpuSmiError as e:
            self.warning(
                f"'npu-smi info -t common' unavailable ({e}); falling back to "
                f"'npu-smi info -t usages|power|temp' for per-scrape collection"
            )
            return False
        return True

    async def _read_hbm_capacity_mb(self, npu_id: int) -> float | None:
        """Read the static HBM capacity for a card, in MiB.

        ``-t memory`` is the documented source; some inference products only
        expose the capacity through ``-t usages``. Both are probe-once queries.
        """
        for subcmd in ("memory", "usages"):
            try:
                text = await _run_npu_smi(
                    "info",
                    "-t",
                    subcmd,
                    "-i",
                    str(npu_id),
                    timeout=DISCOVERY_TIMEOUT_SECONDS,
                )
            except NpuSmiError as e:
                self.debug(lambda e=e, s=subcmd: f"'-t {s}' failed: {e}")
                continue
            capacity = _to_float(
                _lookup(_parse_key_value_lines(text), _HBM_CAPACITY_KEYS)
            )
            if capacity is not None:
                return capacity
        self.warning(
            f"NPU {npu_id}: HBM capacity not reported by npu-smi; "
            f"ascend_memory_used will be absent (usage rate still reported)"
        )
        return None

    def _build_device_state(
        self, npu_id: int, chip_name: str, hbm_capacity_mb: float | None
    ) -> _NpuDeviceState:
        """Build per-NPU state with static metadata.

        npu-smi exposes no device UUID (unlike NVML's ``GPU-<uuid>`` or
        amdsmi's device uuid), so the card ID is the stable identity here. It is
        stable for the life of a host boot and unique within the telemetry
        source, which is what the accumulator keys on.
        """
        return _NpuDeviceState(
            npu_id=npu_id,
            hbm_capacity_mb=hbm_capacity_mb,
            metadata=GpuMetadata(
                gpu_index=npu_id,
                gpu_uuid=f"NPU-{npu_id}",
                gpu_model_name=chip_name or "Ascend NPU",
                device=f"npu{npu_id}",
                hostname="localhost",
                platform=ASCEND_GPU_TELEMETRY_PLATFORM,
            ),
        )

    @on_stop
    async def _shutdown_npu_smi(self) -> None:
        """Drop device state. Invocations are awaited, so nothing to release."""
        self._gpus = []
        self._initialized = False
        self.debug("npu-smi collector shutdown complete")

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Periodic collection task that runs while the collector is RUNNING."""
        await self._collect_and_process_metrics()

    async def collect_and_process_metrics(self) -> None:
        """Public alias for one-shot scrape.

        ``GPUTelemetryManager`` calls this name during baseline and final-state
        capture (``manager.py`` :func:`_capture_collector_baseline` and
        :func:`_handle_profile_complete_command`).
        """
        await self._collect_and_process_metrics()

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics and dispatch via record/error callbacks."""
        try:
            records = await self._collect_records()
            if records and self._record_callback:
                await self._record_callback(records, self.id)
        except Exception as e:  # fault-tolerant telemetry
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as cb_err:  # callback failure must not propagate
                    self.error(f"Failed to send error via callback: {cb_err}")
            else:
                self.error(f"Metrics collection error: {e}")

    async def _collect_records(self) -> list[TelemetryRecord]:
        """Collect one record per NPU, scraping all cards concurrently."""
        if not self._initialized or not self._gpus:
            return []

        timestamp_ns = time.time_ns()
        results = await asyncio.gather(
            *(self._scrape_npu(gpu) for gpu in self._gpus),
            return_exceptions=True,
        )

        records: list[TelemetryRecord] = []
        for gpu, result in zip(self._gpus, results, strict=True):
            if isinstance(result, BaseException):
                self.debug(
                    lambda g=gpu, e=result: (
                        f"NPU {g.npu_id}: scrape failed, skipping this tick: {e!r}"
                    )
                )
                continue
            if result.model_fields_set:
                records.append(
                    TelemetryRecord(
                        timestamp_ns=timestamp_ns,
                        telemetry_source_url=NPU_SMI_SOURCE_IDENTIFIER,
                        **gpu.metadata.model_dump(),
                        telemetry_data=result,
                    )
                )
        return records

    async def _scrape_npu(self, gpu: _NpuDeviceState) -> TelemetryMetrics:
        """Read one card's metrics, preferring the batched ``-t common`` query."""
        if self._common_query_supported:
            try:
                text = await _run_npu_smi(
                    "info",
                    "-t",
                    "common",
                    "-i",
                    str(gpu.npu_id),
                    timeout=SCRAPE_TIMEOUT_SECONDS,
                )
            except NpuSmiError as e:
                # Transient failures (busy driver) fall through to the layered
                # queries rather than disabling `-t common` for the whole run.
                self.debug(
                    lambda e=e, n=gpu.npu_id: (
                        f"NPU {n}: 'npu-smi info -t common' failed: {e}"
                    )
                )
            else:
                return self._metrics_from_pairs(_parse_key_value_lines(text), gpu)

        pairs: list[tuple[str, str]] = []
        for subcmd in ("usages", "power", "temp"):
            with contextlib.suppress(NpuSmiError):
                text = await _run_npu_smi(
                    "info",
                    "-t",
                    subcmd,
                    "-i",
                    str(gpu.npu_id),
                    timeout=SCRAPE_TIMEOUT_SECONDS,
                )
                pairs.extend(_parse_key_value_lines(text))
        return self._metrics_from_pairs(pairs, gpu)

    def _metrics_from_pairs(
        self, pairs: Sequence[tuple[str, str]], gpu: _NpuDeviceState
    ) -> TelemetryMetrics:
        """Map ``npu-smi`` key/value pairs onto a ``TelemetryMetrics``.

        Signals the driver does not report stay unset, so downstream metrics
        treat them as unavailable instead of reading a fabricated zero.
        """
        metrics = TelemetryMetrics()

        power = _to_float(_lookup(pairs, _POWER_KEYS))
        if power is not None:
            metrics.ascend_power = power

        temperature = _to_float(_lookup(pairs, _TEMPERATURE_KEYS))
        if temperature is not None:
            metrics.ascend_temperature = temperature

        aicore = _to_float(_lookup(pairs, _AICORE_KEYS))
        if aicore is not None:
            metrics.ascend_aicore_utilization = aicore

        memory_rate = _to_float(_lookup(pairs, _MEMORY_RATE_KEYS))
        if memory_rate is not None:
            metrics.ascend_memory_utilization = memory_rate
            if gpu.hbm_capacity_mb:
                # npu-smi reports MiB; the field is GB, matching the other
                # collectors' MiB -> GB conversion (1 MiB = 1.048576e-3 GB).
                metrics.ascend_memory_used = (
                    gpu.hbm_capacity_mb * memory_rate / 100.0 * 1.048576e-3
                )

        health = _lookup(pairs, _HEALTH_KEYS)
        if health is not None:
            metrics.ascend_health = 1.0 if health.lower().startswith("ok") else 0.0

        return metrics

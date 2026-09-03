# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host energy telemetry from the Linux powercap RAPL interface.

RAPL exposes a cumulative energy counter per power domain under
`/sys/class/powercap/`. The tree looks like:

    intel-rapl:0/name              -> "package-0"
    intel-rapl:0/energy_uj         -> cumulative microjoules
    intel-rapl:0/max_energy_range_uj
    intel-rapl:0/intel-rapl:0:0/name -> "core"
    intel-rapl:0/intel-rapl:0:1/name -> "dram"

Two things about this interface decide most of the implementation.

The counter wraps. `max_energy_range_uj` is typically around 2^32 microjoules,
roughly 65 kJ, which a busy package can burn through in well under a minute.
A reader that subtracts consecutive samples without handling the wrap reports a
large negative delta once per cycle, so the wrap is corrected here and the
corrected total is what leaves the collector.

The counter is usually root-only. After CVE-2020-8694, which showed RAPL could
be used as a side channel to recover AES keys, distributions restricted
`energy_uj` to mode 0400. So an unprivileged process typically sees the domain
names and not the energy. That is reported as an actionable message rather than
as an empty result, because a silent zero is worse than a refusal.
"""

from __future__ import annotations

import asyncio
import platform
import re
import sys
import time
from pathlib import Path

from aiperf.common.hooks import background_task
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails
from aiperf.common.models.host_telemetry_models import (
    HostPowerDomainMetadata,
    HostTelemetryMetrics,
    HostTelemetryRecord,
)
from aiperf.host_telemetry.protocols import THostErrorCallback, THostRecordCallback

POWERCAP_ROOT = Path("/sys/class/powercap")

# Top-level domains are `intel-rapl:N`, subdomains `intel-rapl:N:M`. AMD exposes
# the same interface under the same `intel-rapl` prefix, which is a kernel
# naming artifact rather than a vendor claim.
_TOP_LEVEL = re.compile(r"^intel-rapl:\d+$")
_SUBDOMAIN = re.compile(r"^intel-rapl:\d+:\d+$")


class RAPLUnavailableError(RuntimeError):
    """Raised when RAPL cannot be read on this host."""


class RAPLDomain:
    """One RAPL power domain and its wraparound-corrected running total."""

    def __init__(self, path: Path, index: int, parent_id: str | None = None) -> None:
        """Initialize a domain from its sysfs directory."""
        self.path = path
        self.domain_id = path.name
        self.index = index
        self.parent_id = parent_id
        self.name = self._read_text("name") or path.name
        self.max_energy_uj = self._read_float("max_energy_range_uj")

        self._last_raw: float | None = None
        self._wrap_offset: float = 0.0

    def _read_text(self, filename: str) -> str | None:
        try:
            return (self.path / filename).read_text().strip()
        except OSError:
            return None

    def _read_float(self, filename: str) -> float | None:
        raw = self._read_text(filename)
        try:
            return float(raw) if raw is not None else None
        except ValueError:
            return None

    def read_energy_uj(self) -> float | None:
        """Return cumulative energy in microjoules, corrected for counter wraparound.

        Returns None if the counter cannot be read, which on most modern
        distributions means the process is not root.

        One bound worth stating, because it is not recoverable from the counter
        alone. A backwards step is the only evidence a wrap happened, so at most
        one wrap can be inferred per read. If the counter wraps more than once
        between two reads, the extra wraps are invisible and the total
        under-reports by that many ranges. The counter carries no timestamp, so
        nothing in the sample distinguishes one wrap from three.

        In practice this needs a very long stall. A package rated 250 W against
        the common 65.5 kJ range wraps about every 262 s, so a reader polling at
        any normal rate is several orders of magnitude clear of it. It is stated
        here because a caller that pauses collection could cross it, and a
        silent undercount is worse than a documented one.
        """
        raw = self._read_float("energy_uj")
        if raw is None:
            return None

        if self._last_raw is not None and raw < self._last_raw:
            # The counter went backwards, so it wrapped. Add one full range.
            # Without max_energy_range_uj the size of the wrap is unknown, and
            # inventing one would silently corrupt the total, so the jump is
            # absorbed instead: the interval is lost, the running total is not.
            if self.max_energy_uj:
                self._wrap_offset += self.max_energy_uj
            else:
                self._wrap_offset += self._last_raw - raw

        self._last_raw = raw
        return raw + self._wrap_offset

    def is_readable(self) -> bool:
        """Whether this domain's energy counter can actually be read."""
        return self._read_float("energy_uj") is not None

    def metadata(self) -> HostPowerDomainMetadata:
        """Static identity of this domain."""
        return HostPowerDomainMetadata(
            domain_index=self.index,
            domain_id=self.domain_id,
            domain_name=self.name,
            parent_domain_id=self.parent_id,
        )


def discover_domains(root: Path = POWERCAP_ROOT) -> list[RAPLDomain]:
    """Find every RAPL domain and subdomain under `root`, parents before children."""
    if not root.is_dir():
        return []

    domains: list[RAPLDomain] = []
    index = 0
    try:
        entries = sorted(p for p in root.iterdir() if _TOP_LEVEL.match(p.name))
    except OSError:
        return []

    for parent_path in entries:
        parent = RAPLDomain(parent_path, index)
        domains.append(parent)
        index += 1
        try:
            children = sorted(
                p for p in parent_path.iterdir() if _SUBDOMAIN.match(p.name)
            )
        except OSError:
            children = []
        for child_path in children:
            domains.append(RAPLDomain(child_path, index, parent_id=parent.domain_id))
            index += 1
    return domains


class RAPLTelemetryCollector(AIPerfLifecycleMixin):
    """Collects host energy telemetry from the Linux powercap RAPL interface.

    Reports the cumulative counter rather than a derived wattage. Deriving power
    needs two samples and a choice of interval, and that choice belongs to
    whatever is aggregating, not to the reader.

    Lifecycle and delivery follow the shipped GPU collectors: `start`/`stop`
    come from `AIPerfLifecycleMixin`, a background task samples every
    `collection_interval` seconds while running, and records leave through
    `record_callback` with failures through `error_callback`.
    """

    def __init__(
        self,
        root: Path = POWERCAP_ROOT,
        *,
        collection_interval: float = 1.0,
        record_callback: THostRecordCallback | None = None,
        error_callback: THostErrorCallback | None = None,
        collector_id: str = "rapl",
    ) -> None:
        """Initialize the collector against a powercap tree."""
        super().__init__(id=collector_id)
        self.root = Path(root)
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._domains: list[RAPLDomain] = []

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier."""
        return "rapl://localhost"

    @property
    def collection_interval(self) -> float:
        """Seconds between samples."""
        return self._collection_interval

    @classmethod
    def validate_environment(cls, root: Path = POWERCAP_ROOT) -> None:
        """Raise RAPLUnavailableError if RAPL cannot be used on this host."""
        if sys.platform != "linux":
            raise RAPLUnavailableError(
                f"RAPL is a Linux powercap interface and this host is {sys.platform}."
            )
        if not Path(root).is_dir():
            raise RAPLUnavailableError(
                f"{root} does not exist, so this kernel exposes no powercap interface. "
                "On a supported CPU this usually means the intel_rapl_common module is "
                "not loaded."
            )
        domains = discover_domains(Path(root))
        if not domains:
            raise RAPLUnavailableError(
                f"{root} exists but contains no intel-rapl domains. "
                f"Detected machine: {platform.machine()}."
            )
        if not any(d.is_readable() for d in domains):
            raise RAPLUnavailableError(
                f"Found {len(domains)} RAPL domain(s) but none has a readable energy_uj. "
                "Since CVE-2020-8694 most distributions restrict it to mode 0400, so this "
                "normally means the process is not running as root."
            )

    async def initialize(self) -> None:
        """Discover the readable RAPL domains on this host."""
        self.validate_environment(self.root)
        self._domains = [d for d in discover_domains(self.root) if d.is_readable()]

    async def is_url_reachable(self) -> bool:
        """Whether at least one RAPL domain can be read."""
        try:
            self.validate_environment(self.root)
        except RAPLUnavailableError:
            return False
        return True

    @property
    def domains(self) -> list[RAPLDomain]:
        """The readable domains found at initialize time."""
        return self._domains

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Sample every collection_interval seconds while the collector runs."""
        await self.collect_and_process_metrics()

    async def collect_and_process_metrics(self) -> None:
        """One-shot scrape, dispatched through the configured callback.

        The sysfs reads happen in a thread so a slow or hung read cannot stall
        the event loop, mirroring the GPU collectors.
        """
        try:
            records = await asyncio.to_thread(self.collect)
            if records and self._record_callback:
                await self._record_callback(records, self.id)
        except Exception as e:  # fault-tolerant telemetry
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:  # fault-tolerant telemetry
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Host telemetry collection error: {e}")

    def collect(self) -> list[HostTelemetryRecord]:
        """Take one sample across every readable domain."""
        timestamp_ns = time.time_ns()
        records: list[HostTelemetryRecord] = []
        for domain in self._domains:
            energy = domain.read_energy_uj()
            if energy is None:
                # A domain that became unreadable mid-run is skipped rather than
                # reported as zero, which would look like an idle package.
                continue
            records.append(
                HostTelemetryRecord(
                    timestamp_ns=timestamp_ns,
                    telemetry_source_url=self.endpoint_url,
                    domain=domain.metadata(),
                    telemetry_data=HostTelemetryMetrics(
                        energy_consumption_uj=energy,
                        energy_range_uj=domain.max_energy_uj,
                    ),
                )
            )
        return records


def rapl_is_available(root: Path = POWERCAP_ROOT) -> bool:
    """Whether this host can provide RAPL energy telemetry."""
    try:
        RAPLTelemetryCollector.validate_environment(root)
    except RAPLUnavailableError:
        return False
    return True


__all__ = [
    "POWERCAP_ROOT",
    "RAPLDomain",
    "RAPLTelemetryCollector",
    "RAPLUnavailableError",
    "discover_domains",
    "rapl_is_available",
]

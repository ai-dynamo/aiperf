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
import logging
import math
import platform
import re
import time
from pathlib import Path

from pydantic import ValidationError

from aiperf.common.constants import IS_LINUX
from aiperf.common.hooks import background_task, on_init
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

logger = logging.getLogger(__name__)


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
        raw_range = self._read_text("max_energy_range_uj")
        self.max_energy_uj = self._parse_float(raw_range)
        if raw_range is not None and self.max_energy_uj is None:
            # The file exists on every real intel-rapl domain, so a present but
            # unparseable range signals a fault, and taking the lossy no-range
            # wrap fallback silently would hide it.
            logger.warning(
                "%s: max_energy_range_uj is present but unparseable (%r); "
                "wrap correction degrades to the lossy no-range fallback",
                self.domain_id, raw_range,
            )

        self._last_raw: float | None = None
        self._wrap_offset: float = 0.0

    def _read_text(self, filename: str) -> str | None:
        try:
            return (self.path / filename).read_text().strip()
        except OSError as e:
            # EACCES (not root) and a transient EIO present identically to the
            # caller as None; the exception type is the only thing that tells
            # them apart, so it is worth a log line even at debug level.
            logger.debug("%s/%s: %s: %s", self.domain_id, filename,
                         type(e).__name__, e)
            return None

    @staticmethod
    def _parse_float(raw: str | None) -> float | None:
        try:
            value = float(raw) if raw is not None else None
        except ValueError:
            return None
        # 'inf' and 'nan' parse successfully, and a non-finite energy would
        # otherwise ride all the way to serialization before being lost.
        if value is not None and not math.isfinite(value):
            return None
        return value

    def _read_float(self, filename: str) -> float | None:
        return self._parse_float(self._read_text(filename))

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

        A backwards step is also not proof of a wrap. The counter resets to
        near zero on suspend/resume, an intel_rapl module reload, or a
        container restart, and crediting a reset as a wrap would inject a full
        range of phantom energy. So a backwards step only counts as a wrap
        when the previous reading was in the upper half of the declared range;
        otherwise the jump is absorbed, the interval is lost, and the running
        total stays nondecreasing. The same absorption is the fallback when no
        range is declared, and in both absorbed cases the total is permanently
        biased low by the unknowable lost interval, which compounds if it
        happens again. That bias is the price of never inventing energy.
        """
        raw = self._read_float("energy_uj")
        if raw is None:
            return None

        if self._last_raw is not None and raw < self._last_raw:
            if self.max_energy_uj and self._last_raw >= 0.5 * self.max_energy_uj:
                # Plausible wrap: the counter was near its ceiling.
                self._wrap_offset += self.max_energy_uj
            else:
                if self.max_energy_uj:
                    logger.warning(
                        "%s: counter went backwards from %.0f at %.0f%% of "
                        "range; treating as a reset, not a wrap",
                        self.domain_id, self._last_raw,
                        100.0 * self._last_raw / self.max_energy_uj,
                    )
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
        entries = sorted(
            (p for p in root.iterdir() if _TOP_LEVEL.match(p.name)),
            key=lambda p: int(p.name.rsplit(":", 1)[1]),
        )
    except OSError:
        return []

    for parent_path in entries:
        parent = RAPLDomain(parent_path, index)
        domains.append(parent)
        index += 1
        try:
            children = sorted(
                (p for p in parent_path.iterdir() if _SUBDOMAIN.match(p.name)),
                key=lambda p: int(p.name.rsplit(":", 1)[1]),
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
        """Raise RAPLUnavailableError if RAPL cannot be used on this host.

        This is a class-level preflight against the platform default tree, by
        design: a protocol-driven caller invokes it before any instance
        exists. An instance's configured ``root`` is honoured at initialize
        time, not here.
        """
        if not IS_LINUX:
            raise RAPLUnavailableError(
                f"RAPL is a Linux powercap interface and this host is "
                f"{platform.system()}."
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

    @on_init
    async def _discover_readable_domains(self) -> None:
        """Discover the readable RAPL domains on this host.

        Runs as an ON_INIT hook so the mixin's own ``initialize`` still drives
        the CREATED -> INITIALIZED transition that ``start`` requires;
        overriding ``initialize`` outright left the collector unstartable.
        Discovery happens in a thread for the same reason the sampling path
        does: sysfs reads against a wedged driver hang, and they would hang
        the event loop from here just as surely as from a sample.
        """
        self._domains = await asyncio.to_thread(self._discover_readable_sync)

    def _discover_readable_sync(self) -> list[RAPLDomain]:
        """One walk of the tree, validated and filtered to readable domains."""
        if not IS_LINUX:
            raise RAPLUnavailableError(
                f"RAPL is a Linux powercap interface and this host is "
                f"{platform.system()}."
            )
        if not self.root.is_dir():
            raise RAPLUnavailableError(
                f"{self.root} does not exist, so this kernel exposes no "
                "powercap interface. On a supported CPU this usually means "
                "the intel_rapl_common module is not loaded."
            )
        domains = discover_domains(self.root)
        if not domains:
            raise RAPLUnavailableError(
                f"{self.root} exists but contains no intel-rapl domains. "
                f"Detected machine: {platform.machine()}."
            )
        readable = [d for d in domains if d.is_readable()]
        if not readable:
            raise RAPLUnavailableError(
                f"Found {len(domains)} RAPL domain(s) but none has a readable "
                "energy_uj. Since CVE-2020-8694 most distributions restrict it "
                "to mode 0400, so this normally means the process is not "
                "running as root."
            )
        return readable

    async def is_url_reachable(self) -> bool:
        """Whether at least one RAPL domain can be read."""
        try:
            await asyncio.to_thread(self._discover_readable_sync)
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
            if not records and self._domains:
                # Total telemetry loss must not look like a healthy idle run:
                # permission revocation or driver teardown lands here.
                raise RAPLUnavailableError(
                    f"All {len(self._domains)} previously readable RAPL "
                    "domain(s) produced no sample."
                )
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
            try:
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
            except ValidationError as e:
                # One malformed domain must not discard the sample from the
                # domains that read perfectly; skipping matches the policy for
                # the unreadable case above.
                logger.warning("%s: sample failed validation, skipped: %s",
                               domain.domain_id, e)
                continue
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

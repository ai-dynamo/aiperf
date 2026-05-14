# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake node-exporter Prometheus endpoint for the AIPerf mock server.

The real node-exporter (https://github.com/prometheus/node_exporter) exposes
~336 metric families across gauge, counter, histogram, summary, AND ``untyped``
types. The ``untyped`` families (e.g. ``node_netstat_Icmp_InErrors``) are not
reliably reproducible from ``prometheus_client`` (which has no public
``Untyped`` primitive) — yet they are exactly the wire shape AIPerf's
server-metrics scraper must tolerate.

This module provides a small builder API that programmatically emits a
Prometheus exposition body. Tests parametrise it (number of untyped families,
random-walk seed, optional cross-scrape type collision, etc.); the mock server
serves a sensible default mix on ``/node_exporter/metrics`` so the AIPerf
scraper pipeline can be exercised end-to-end without pulling a Docker image.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# A value-producer takes the scrape index (0, 1, 2, ...) and a per-faker RNG
# and returns the next observation. Stateless helpers can ignore both args.
ValueFn = Callable[[int, random.Random], float]

# Label combinations are encoded as ordered tuples of (key, value) pairs so the
# rendered line order is deterministic across scrapes.
LabelTuple = tuple[tuple[str, str], ...]


# ============================================================================
# Family specs
# ============================================================================


@dataclass(slots=True)
class _ScalarFamily:
    """Spec for a single-sample family: gauge, counter, or untyped."""

    name: str
    help_text: str
    type_decl: str  # "gauge" | "counter" | "untyped" | "" (omit TYPE line)
    samples: list[tuple[LabelTuple, ValueFn]]


@dataclass(slots=True)
class _HistogramFamily:
    """Spec for a histogram family with explicit cumulative buckets."""

    name: str
    help_text: str
    base_labels: LabelTuple
    buckets: Sequence[tuple[str, ValueFn]]
    sum_fn: ValueFn
    count_fn: ValueFn


@dataclass(slots=True)
class _SummaryFamily:
    """Spec for a summary family (AIPerf skips these, but real exporters emit them)."""

    name: str
    help_text: str
    quantiles: Sequence[tuple[str, ValueFn]]
    sum_fn: ValueFn
    count_fn: ValueFn


# ============================================================================
# Builder
# ============================================================================


@dataclass(slots=True)
class NodeExporterFaker:
    """Build Prometheus exposition bodies that look like a node-exporter scrape.

    The faker is stateful only in the sense that it advances an internal
    scrape index every time ``render()`` is called, which value-functions can
    use to drift their observations (counters, random walks, etc.).

    Two convenience constructors are provided:

    - ``NodeExporterFaker.default()`` — a realistic mix mirroring what a real
      node-exporter exposes, including 7 untyped families.
    - ``NodeExporterFaker.collisions(...)`` — same mix plus one family whose
      ``# TYPE`` declaration flips between scrapes, exercising the
      cross-scrape type-reclassification edge case.

    Lower-level callers can pass arbitrary family specs to the constructor.
    """

    scalars: list[_ScalarFamily] = field(default_factory=list)
    histograms: list[_HistogramFamily] = field(default_factory=list)
    summaries: list[_SummaryFamily] = field(default_factory=list)
    seed: int = 0
    pre_render: Callable[[NodeExporterFaker, int], None] | None = None
    _scrape_index: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- public API ----------------------------------------------------------

    def add_untyped(
        self,
        name: str,
        help_text: str,
        value_fn: ValueFn,
        labels: LabelTuple = (),
        *,
        omit_type_line: bool = False,
    ) -> NodeExporterFaker:
        """Append an ``# TYPE foo untyped`` (or no-TYPE-declaration) family.

        ``omit_type_line=True`` reproduces the real-world quirk where some
        exporters emit samples with no ``# TYPE`` declaration at all; the
        Prometheus parser also classifies those as untyped.
        """
        self.scalars.append(
            _ScalarFamily(
                name=name,
                help_text=help_text,
                type_decl="" if omit_type_line else "untyped",
                samples=[(labels, value_fn)],
            )
        )
        return self

    def add_gauge(
        self,
        name: str,
        help_text: str,
        value_fn: Iterable[tuple[LabelTuple, ValueFn]] | ValueFn,
        labels: LabelTuple = (),
    ) -> NodeExporterFaker:
        """Append a ``# TYPE foo gauge`` family.

        ``value_fn`` is either a single ``ValueFn`` (the family has one sample
        with ``labels``) or a list of ``(labels, value_fn)`` pairs for
        multi-dimensional families.
        """
        self.scalars.append(
            _ScalarFamily(
                name=name,
                help_text=help_text,
                type_decl="gauge",
                samples=_normalize_samples(value_fn, labels),
            )
        )
        return self

    def add_counter(
        self,
        name: str,
        help_text: str,
        value_fn: Iterable[tuple[LabelTuple, ValueFn]] | ValueFn,
        labels: LabelTuple = (),
    ) -> NodeExporterFaker:
        """Append a ``# TYPE foo counter`` family.

        Accepts the same ``value_fn`` shapes as :meth:`add_gauge`.
        """
        self.scalars.append(
            _ScalarFamily(
                name=name,
                help_text=help_text,
                type_decl="counter",
                samples=_normalize_samples(value_fn, labels),
            )
        )
        return self

    def add_histogram(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[tuple[str, ValueFn]],
        sum_fn: ValueFn,
        count_fn: ValueFn,
        base_labels: LabelTuple = (),
    ) -> NodeExporterFaker:
        """Append a ``# TYPE foo histogram`` family with cumulative buckets.

        ``buckets`` is a list of ``(le, value_fn)`` pairs; include ``"+Inf"``
        explicitly. ``base_labels`` are applied to every bucket plus ``_sum``
        and ``_count`` lines.
        """
        self.histograms.append(
            _HistogramFamily(
                name=name,
                help_text=help_text,
                base_labels=base_labels,
                buckets=tuple(buckets),
                sum_fn=sum_fn,
                count_fn=count_fn,
            )
        )
        return self

    def add_summary(
        self,
        name: str,
        help_text: str,
        quantiles: Sequence[tuple[str, ValueFn]],
        sum_fn: ValueFn,
        count_fn: ValueFn,
    ) -> NodeExporterFaker:
        """Append a ``# TYPE foo summary`` family (AIPerf will skip it)."""
        self.summaries.append(
            _SummaryFamily(
                name=name,
                help_text=help_text,
                quantiles=tuple(quantiles),
                sum_fn=sum_fn,
                count_fn=count_fn,
            )
        )
        return self

    def render(self) -> str:
        """Return the next scrape's exposition body and advance internal state."""
        n = self._scrape_index
        if self.pre_render is not None:
            self.pre_render(self, n)
        rng = self._rng
        out: list[str] = []
        for fam in self.scalars:
            if fam.help_text:
                out.append(f"# HELP {fam.name} {fam.help_text}")
            if fam.type_decl:
                out.append(f"# TYPE {fam.name} {fam.type_decl}")
            for labels, value_fn in fam.samples:
                out.append(f"{fam.name}{_format_labels(labels)} {value_fn(n, rng)}")
        for hist in self.histograms:
            out.append(f"# HELP {hist.name} {hist.help_text}")
            out.append(f"# TYPE {hist.name} histogram")
            for le, value_fn in hist.buckets:
                labels = (*hist.base_labels, ("le", le))
                out.append(
                    f"{hist.name}_bucket{_format_labels(labels)} {value_fn(n, rng)}"
                )
            out.append(
                f"{hist.name}_sum{_format_labels(hist.base_labels)} {hist.sum_fn(n, rng)}"
            )
            out.append(
                f"{hist.name}_count{_format_labels(hist.base_labels)} {hist.count_fn(n, rng)}"
            )
        for summ in self.summaries:
            out.append(f"# HELP {summ.name} {summ.help_text}")
            out.append(f"# TYPE {summ.name} summary")
            for q, value_fn in summ.quantiles:
                out.append(f'{summ.name}{{quantile="{q}"}} {value_fn(n, rng)}')
            out.append(f"{summ.name}_sum {summ.sum_fn(n, rng)}")
            out.append(f"{summ.name}_count {summ.count_fn(n, rng)}")
        self._scrape_index += 1
        return "\n".join(out) + "\n"

    # -- convenience constructors --------------------------------------------

    @classmethod
    def default(cls, seed: int = 0) -> NodeExporterFaker:
        """A realistic mix covering every type AIPerf's scraper handles.

        Includes 7 ``untyped`` families (one with no ``# TYPE`` declaration),
        3 gauges (one with labels), 2 counters (multi-label), 1 histogram, and
        1 summary. Values drift per scrape so percentile stats are non-zero.
        """
        f = cls(seed=seed)
        # ---- untyped ----
        f.add_untyped(
            "node_netstat_Icmp_InErrors",
            "Statistic IcmpInErrors.",
            value_fn=lambda n, _r: float(n % 3),
        )
        f.add_untyped(
            "node_netstat_Icmp_InMsgs",
            "Statistic IcmpInMsgs.",
            value_fn=lambda n, _r: 42.0 + n,
        )
        f.add_untyped(
            "node_netstat_Tcp_InSegs",
            "Statistic TcpInSegs.",
            value_fn=lambda n, _r: 1024.0 + n * 7,
        )
        f.add_untyped(
            "node_netstat_Tcp_OutSegs",
            "Statistic TcpOutSegs.",
            value_fn=lambda n, _r: 2048.0 + n * 11,
        )
        f.add_untyped(
            "node_netstat_Tcp_RetransSegs",
            "Statistic TcpRetransSegs.",
            value_fn=lambda n, _r: float(n // 5),
        )
        f.add_untyped(
            "node_netstat_IpExt_InOctets",
            "Statistic IpExtInOctets.",
            value_fn=lambda n, r: 100_000.0 + n * 1500 + r.uniform(-50, 50),
        )
        # Real-world quirk: family with no `# TYPE` line. Parser treats as untyped.
        f.add_untyped(
            "node_legacy_no_type_metric",
            "Demonstrates missing TYPE declaration (untyped fallback).",
            value_fn=lambda n, _r: n * 0.5,
            omit_type_line=True,
        )
        # ---- gauge ----
        f.add_gauge(
            "node_load1",
            "1m load average.",
            value_fn=lambda n, r: 0.5 + (n % 7) * 0.1 + r.uniform(-0.05, 0.05),
        )
        f.add_gauge(
            "node_load5",
            "5m load average.",
            value_fn=lambda n, r: 0.4 + (n % 11) * 0.07 + r.uniform(-0.04, 0.04),
        )
        f.add_gauge(
            "node_arp_entries",
            "ARP entries by device.",
            value_fn=[
                ((("device", "eth0"),), lambda n, _r: 10 + n % 5),
                ((("device", "lo"),), lambda n, _r: 1.0),
            ],
        )
        # ---- counter ----
        f.add_counter(
            "node_cpu_seconds_total",
            "Seconds the CPUs spent in each mode.",
            value_fn=[
                (
                    (("cpu", "0"), ("mode", "idle")),
                    lambda n, _r: 10_000.0 + n * 0.95,
                ),
                (
                    (("cpu", "0"), ("mode", "user")),
                    lambda n, _r: 1_000.0 + n * 0.03,
                ),
            ],
        )
        f.add_gauge(
            "node_boot_time_seconds",
            "Node boot time, in unixtime.",
            value_fn=lambda _n, _r: 1_700_000_000.0,
        )
        # ---- histogram ----
        f.add_histogram(
            "node_scrape_collector_duration_seconds",
            "node_exporter: Duration of a collector scrape.",
            base_labels=(("collector", "cpu"),),
            buckets=[
                ("0.001", lambda n, _r: 10 + n),
                ("0.01", lambda n, _r: 20 + n),
                ("0.1", lambda n, _r: 22 + n),
                ("+Inf", lambda n, _r: 22 + n),
            ],
            sum_fn=lambda n, _r: 0.07 * (n + 1),
            count_fn=lambda n, _r: 22 + n,
        )
        # ---- summary (skipped by AIPerf) ----
        f.add_summary(
            "go_gc_duration_seconds",
            "Garbage-collection pause durations.",
            quantiles=[
                ("0.5", lambda _n, _r: 0.000015),
                ("0.99", lambda _n, _r: 0.00018),
            ],
            sum_fn=lambda _n, _r: 0.0021,
            count_fn=lambda n, _r: float(n + 1),
        )
        return f

    @classmethod
    def collisions(cls, seed: int = 0) -> NodeExporterFaker:
        """``default()`` plus a metric whose declared type flips every scrape.

        Useful for exercising the cross-scrape type-reclassification edge case:
        a Prometheus server reclassifying ``foo`` between histogram and gauge
        across consecutive scrapes.
        """
        f = cls.default(seed=seed)
        unstable = _ScalarFamily(
            name="node_unstable_type",
            help_text="Reclassifies between scrapes (type-collision scenario).",
            type_decl="gauge",
            samples=[((), lambda n, _r: float(n))],
        )
        f.scalars.append(unstable)

        def _flip(_faker: NodeExporterFaker, n: int) -> None:
            unstable.type_decl = "histogram" if n % 2 else "gauge"

        f.pre_render = _flip
        return f


# ============================================================================
# Helpers
# ============================================================================


def _normalize_samples(
    samples: Iterable[tuple[LabelTuple, ValueFn]] | ValueFn,
    fallback_labels: LabelTuple,
) -> list[tuple[LabelTuple, ValueFn]]:
    if callable(samples):
        return [(fallback_labels, samples)]
    return list(samples)


def _format_labels(labels: LabelTuple | Mapping[str, str]) -> str:
    if not labels:
        return ""
    pairs = labels.items() if isinstance(labels, Mapping) else labels
    body = ",".join(f'{k}="{v}"' for k, v in pairs)
    return "{" + body + "}"


# A process-wide default instance so the FastAPI route returns consistent
# scrape-to-scrape drift without callers having to instantiate one.
_DEFAULT = NodeExporterFaker.default()


def render_default() -> str:
    """Render the module-level default faker (one scrape, advances state)."""
    return _DEFAULT.render()


__all__ = [
    "NodeExporterFaker",
    "render_default",
]

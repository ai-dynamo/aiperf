# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-metrics realtime-snapshot residuals not covered by the pathological suite.

Four behaviors of ``ServerMetricsAccumulator``:

* SGLang ``num_retracted_reqs_total`` (parser-stripped to ``num_retracted_reqs``)
  feeds ``num_preemptions`` via a COUNTER-typed delta (``_add_preemptions`` /
  ``_counter_delta``).
* Counter lookups use the parser-stripped family name (no ``_total`` suffix),
  driven here through the *real* ``prometheus_client.parser`` so any drift
  between lookup names and the parser convention fails loudly.
* ``realtime_snapshot`` suppresses counter deltas that have only a single
  sample (``_counter_delta`` needs >= 2 points), so hit-rate / preemption rows
  never appear off one scrape.
* ``export_results`` widens the Parquet export window to the final
  per-endpoint ``last_update_ns`` (``_export_parquet_if_enabled``) so Parquet
  uses the same end window as the summaries.
"""

from __future__ import annotations

import pytest

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from tests.unit.conftest import make_run_from_cli


def _accumulator() -> ServerMetricsAccumulator:
    return ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                endpoint_type=EndpointType.CHAT,
                streaming=False,
            )
        )
    )


@pytest.mark.asyncio
async def test_realtime_snapshot_suppresses_single_sample_counter_deltas() -> None:
    """Realtime counter deltas require two samples; one scrape emits nothing."""
    acc = _accumulator()
    await acc.process_server_metrics_record(
        ServerMetricsRecord(
            endpoint_url="http://127.0.0.1:8000/metrics",
            timestamp_ns=1_000_000_000,
            metrics={
                "vllm:prefix_cache_hits": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Prefix cache hits.",
                    samples=[MetricSample(value=500.0)],
                ),
                "vllm:prefix_cache_queries": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Prefix cache queries.",
                    samples=[MetricSample(value=1000.0)],
                ),
                "vllm:num_preemptions": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Preemptions.",
                    samples=[MetricSample(value=7.0)],
                ),
            },
        )
    )

    snapshot = acc.realtime_snapshot()

    assert "prefix_cache_hit_rate" not in snapshot
    assert "num_preemptions" not in snapshot


@pytest.mark.asyncio
async def test_realtime_snapshot_uses_sglang_retracted_total_counter() -> None:
    """SGLang ``num_retracted_reqs_total`` (counter) feeds ``num_preemptions``.

    ``prometheus_client.parser`` strips ``_total`` so the family is stored as
    ``sglang:num_retracted_reqs``; the COUNTER-type filter in ``_counter_delta``
    keeps a same-named gauge from contaminating the lookup.
    """
    acc = _accumulator()
    for timestamp_ns, counter_value in (
        (1_000_000_000, 10.0),
        (2_000_000_000, 12.0),
    ):
        await acc.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url="http://127.0.0.1:8000/metrics",
                timestamp_ns=timestamp_ns,
                metrics={
                    "sglang:num_retracted_reqs": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        description="Total retracted requests.",
                        samples=[MetricSample(value=counter_value)],
                    ),
                },
            )
        )

    snapshot = acc.realtime_snapshot()

    assert snapshot["num_preemptions"] == 2.0


@pytest.mark.asyncio
async def test_realtime_snapshot_handles_parser_stripped_total_suffix() -> None:
    """Counter throughput resolves when families are named by the real parser.

    Regression for the ``_total`` parser-stripping bug: the data collector
    stores ``vllm:prompt_tokens_total`` as ``vllm:prompt_tokens``, and the
    snapshot must look it up under the stripped name. Without the fix the
    throughput rows silently vanished at runtime.
    """
    from prometheus_client.parser import text_string_to_metric_families

    text_t1 = """\
# HELP vllm:prompt_tokens_total Total prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 0.0
# HELP vllm:generation_tokens_total Total generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="m"} 0.0
"""
    text_t2 = """\
# HELP vllm:prompt_tokens_total Total prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="m"} 1000000.0
# HELP vllm:generation_tokens_total Total generation tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="m"} 5000.0
"""

    def parse(text: str) -> dict[str, MetricFamily]:
        out: dict[str, MetricFamily] = {}
        for family in text_string_to_metric_families(text):
            metric_type = PrometheusMetricType(family.type)
            samples = [
                MetricSample(labels=dict(s.labels) or None, value=s.value)
                for s in family.samples
            ]
            out[family.name] = MetricFamily(
                type=metric_type,
                description=family.documentation or "",
                samples=samples,
            )
        return out

    acc = _accumulator()
    for ts, text in (
        (1_000_000_000, text_t1),
        (2_000_000_000, text_t2),
    ):
        await acc.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url="http://127.0.0.1:8000/metrics",
                timestamp_ns=ts,
                metrics=parse(text),
            )
        )

    snapshot = acc.realtime_snapshot()

    # Rates over a 1-second window between samples.
    assert snapshot["input_token_throughput_srv"] == pytest.approx(1_000_000.0)
    assert snapshot["output_token_throughput_srv"] == pytest.approx(5_000.0)


@pytest.mark.asyncio
async def test_export_results_extends_parquet_filter_to_endpoint_last_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parquet export uses the same final-collection end window as summaries.

    ``export_results`` widens the export window to
    ``max(end_ns, last_update_ns)`` so a scrape landing after the profiling
    end still appears in the Parquet file.
    """
    acc = _accumulator()
    exported_filters = []

    async def capture_export_filter(time_filter):
        exported_filters.append(time_filter)

    monkeypatch.setattr(acc, "_export_parquet_if_enabled", capture_export_filter)

    for timestamp_ns in (1_000_000_000, 3_000_000_000):
        gauge = MetricFamily(
            type=PrometheusMetricType.GAUGE,
            description="Cache usage",
            samples=[MetricSample(labels=None, value=0.5)],
        )
        await acc.process_server_metrics_record(
            ServerMetricsRecord(
                endpoint_url="http://node1:8081/metrics",
                timestamp_ns=timestamp_ns,
                endpoint_latency_ns=5_000_000,
                metrics={"cache_usage": gauge},
            )
        )

    result = await acc.export_results(
        ExportContext(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        )
    )

    assert result is not None
    assert len(exported_filters) == 1
    assert exported_filters[0].start_ns == 1_000_000_000
    # end_ns widened from the 2e9 profiling end to the 3e9 last collection.
    assert exported_filters[0].end_ns == 3_000_000_000

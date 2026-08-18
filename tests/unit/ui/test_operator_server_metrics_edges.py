# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HELPERS = (
    REPO
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "server-metrics"
    / "helpers.js"
).as_uri()
SERVER_METRICS_COMPONENT = (
    REPO
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "components"
    / "server-metrics"
    / "index.js"
)
JOB_DETAIL = REPO / "src" / "aiperf" / "operator" / "ui" / "pages" / "job-detail.js"


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def test_curate_server_metrics_keeps_partial_nonzero_gpu_metric() -> None:
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            'vllm:kv_cache_usage_perc': {{
              type: 'gauge',
              series: [{{ endpoint_url: 'u', labels: {{ tp_rank: 0 }}, stats: {{ avg: 0, max: 0 }} }}],
            }},
            'sglang:token_usage': {{
              type: 'gauge',
              series: [{{ endpoint_url: 'u', labels: {{ gpu: '0' }}, stats: {{ avg: 0.25, max: 0.75 }} }}],
            }},
          }},
        }};
        const curated = curateServerMetrics(normalizeServerMetrics(snapshot));
        const kv = curated.kpis.find(k => k.id === 'kv-cache-pressure');
        console.log(JSON.stringify({{
          value: kv.value,
          sub: kv.sub,
          source: kv.source,
          detail: curated.detailRows.map(row => [row.backend, row.kvPressure]),
        }}));
    """
    assert _run_node(script) == (
        '{"value":75,"sub":"avg 25.0%","source":"sglang:token_usage",'
        '"detail":[["u",75]]}'
    )


def test_curate_server_metrics_omits_gpu_kpi_when_gpu_stats_missing() -> None:
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_requests: {{
              type: 'counter',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ rate: 9 }} }}],
            }},
            'vllm:kv_cache_usage_perc': {{
              type: 'gauge',
              series: [{{ endpoint_url: 'u', labels: {{ tp_rank: 0 }}, stats: {{}} }}],
            }},
          }},
        }};
        const curated = curateServerMetrics(normalizeServerMetrics(snapshot));
        console.log(JSON.stringify({{
          ids: curated.kpis.map(k => k.id).sort(),
          kvDetailValues: curated.detailRows.map(row => row.kvPressure),
        }}));
    """
    assert _run_node(script) == (
        '{"ids":["request-rate"],"kvDetailValues":[null,null]}'
    )


def test_curate_server_metrics_labels_backend_rows_from_label_columns() -> None:
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_requests: {{
              type: 'counter',
              series: [
                {{ endpoint_url: 'u', labels: {{ dynamo_component: 'prefill' }}, stats: {{ rate: 1 }} }},
                {{ endpoint_url: 'u', labels: {{ tp_rank: 1, pp_rank: 0 }}, stats: {{ rate: 2 }} }},
                {{ endpoint_url: 'u', labels: {{ engine: 3 }}, stats: {{ rate: 3 }} }},
              ],
            }},
          }},
        }};
        const rows = curateServerMetrics(normalizeServerMetrics(snapshot)).detailRows;
        console.log(JSON.stringify(rows.map(row => [row.backend, row.reqRate])));
    """
    assert _run_node(script) == ('[["engine-3",3],["prefill",1],["tp1/pp0",2]]')


def test_normalize_server_metrics_ignores_malformed_endpoint_series_shape() -> None:
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const normalized = normalizeServerMetrics({{
          endpoint_summaries: {{
            'http://host-a:9090/metrics': {{
              metrics: {{
                dynamo_frontend_requests: {{
                  type: 'counter',
                  series: {{ endpoint_url: 'bad-shape', labels: {{}}, stats: {{ rate: 99 }} }},
                }},
              }},
            }},
          }},
        }});
        const curated = curateServerMetrics(normalized);
        console.log(JSON.stringify({{
          configured: normalized.summary.endpoints_configured,
          successful: normalized.summary.endpoints_successful,
          series: normalized.metrics.dynamo_frontend_requests.series,
          curatedIsNull: curated === null,
        }}));
    """
    assert _run_node(script) == (
        '{"configured":["http://host-a:9090/metrics"],'
        '"successful":["http://host-a:9090/metrics"],'
        '"series":[],"curatedIsNull":false}'
    )


def test_aggregate_sparkline_snapshot_filters_non_finite_values() -> None:
    script = f"""
        import {{ normalizeServerMetrics, aggregateSparklineSnapshot }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_requests: {{
              type: 'counter',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ rate: Infinity }} }}],
            }},
            dynamo_frontend_output_tokens: {{
              type: 'counter',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ rate: 0 }} }}],
            }},
          }},
        }};
        const agg = aggregateSparklineSnapshot(normalizeServerMetrics(snapshot));
        console.log(JSON.stringify(agg));
    """
    assert _run_node(script) == (
        '{"values":{"generation-token-rate":0},"latencyKpiId":null}'
    )


def test_curate_server_metrics_latency_units_are_milliseconds() -> None:
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_time_to_first_token_seconds: {{
              type: 'histogram',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ count: 10, p99_estimate: 1.234 }} }}],
            }},
          }},
        }};
        const curated = curateServerMetrics(normalizeServerMetrics(snapshot));
        const kpi = curated.kpis.find(k => k.id === 'p99-ttft');
        console.log(JSON.stringify({{
          kpiValue: kpi.value,
          kpiUnit: kpi.unit,
          detailValue: curated.detailRows[0].latencyP99Ms,
        }}));
    """
    assert _run_node(script) == '{"kpiValue":1234,"kpiUnit":"ms","detailValue":1234}'


def test_live_and_final_source_chips_are_explicit_in_component() -> None:
    source = SERVER_METRICS_COMPONENT.read_text()
    assert "const sourceLabel = source === 'live' ? 'LIVE' : 'FINAL';" in source
    assert '<span class="metric-source-chip">${sourceLabel}</span>' in source


def test_job_detail_prefers_final_server_metrics_over_live_snapshot() -> None:
    source = JOB_DETAIL.read_text()
    assert (
        "const displayedServerMetrics = serverMetrics || liveServerMetrics;" in source
    )
    assert "const serverMetricsSource = serverMetrics ? 'final' : 'live';" in source


def test_malformed_server_metrics_artifact_json_shows_error_card() -> None:
    source = JOB_DETAIL.read_text()
    assert (
        "fetch(`${resultsBase}/server_metrics_export.json`, { signal: ac.signal })"
        in source
    )
    assert ".then(r => (r.ok ? r.json() : null))" in source
    assert (
        "setServerMetricsError(err?.message ?? 'Server metrics artifact could not be read.');"
        in source
    )
    assert "${isTerminal && serverMetricsError && html`" in source
    assert '<div class="card-title">Server Metrics</div>' in source


def test_server_metrics_component_formats_units_for_kpis_and_detail_rows() -> None:
    """Every unit the component knows routes to a formatter chosen for it, and
    the detail rows restate the unit they formatted for."""
    source = SERVER_METRICS_COMPONENT.read_text()

    assert "if (kpi.unit === 'req/s') return fmtReqPerSecond(kpi.value);" in source
    assert "if (kpi.unit === 'tok/s') return fmtThroughput(kpi.value);" in source
    assert "if (kpi.unit === '%') return fmtNumber(kpi.value, 1);" in source
    assert "if (kpi.unit === 'ms') return fmtMilliseconds(kpi.value);" in source

    assert "if (unit === 'rate') return fmtReqPerSecond(value);" in source
    assert "if (unit === 'percent') return `${fmtNumber(value, 1)}%`;" in source
    assert "if (unit === 'ms') return `${fmtMilliseconds(value)} ms`;" in source

    # A number that reached a formatter without a matching unit still has to be
    # rendered, and an absent one must not print as a zero.
    assert "return fmtInt(kpi.value);" in source
    assert "if (kpi.value == null) return '—';" in source
    assert "if (value == null) return '—';" in source

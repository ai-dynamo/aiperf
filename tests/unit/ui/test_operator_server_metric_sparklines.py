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


def test_aggregate_sparkline_snapshot_dynamo_frontend() -> None:
    """A dynamo-frontend snapshot resolves all five KPI ids and the
    p99-ttft latency id, and the values match what curateServerMetrics
    would produce from the same snapshot (curator scales latency to ms)."""
    script = f"""
        import {{
          normalizeServerMetrics, curateServerMetrics, aggregateSparklineSnapshot,
        }} from {HELPERS!r};
        const snapshot = {{
          summary: {{
            endpoints_configured: ['http://h1:9090/metrics'],
            endpoints_successful: ['http://h1:9090/metrics'],
          }},
          metrics: {{
            dynamo_frontend_requests: {{
              type: 'counter',
              series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ rate: 12.5 }} }}],
            }},
            dynamo_frontend_output_tokens: {{
              type: 'counter',
              series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ rate: 230 }} }}],
            }},
            dynamo_component_kvstats_gpu_cache_usage_percent: {{
              type: 'gauge',
              series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{ dynamo_component: 'worker-a' }}, stats: {{ avg: 0.42, max: 0.68 }} }}],
            }},
            dynamo_frontend_time_to_first_token_seconds: {{
              type: 'histogram',
              series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ count: 100, p99_estimate: 0.085 }} }}],
            }},
            dynamo_frontend_queued_requests: {{
              type: 'gauge',
              series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ avg: 3, max: 5 }} }}],
            }},
          }},
        }};
        const norm = normalizeServerMetrics(snapshot);
        const agg = aggregateSparklineSnapshot(norm);
        const curated = curateServerMetrics(norm);
        const curatedById = Object.fromEntries(curated.kpis.map(k => [k.id, k.value]));
        // Curator scales latency seconds → ms; aggregator keeps raw seconds.
        // Equality check rescales the curator value back to seconds for the
        // latency id so the assertion is comparing apples to apples.
        const curatedScaled = (id) =>
          (id === 'p99-ttft' || id === 'p99-e2e-latency')
            ? curatedById[id] / 1000
            : curatedById[id];
        console.log(JSON.stringify({{
          ids: Object.keys(agg.values).sort(),
          latencyKpiId: agg.latencyKpiId,
          equal: Object.keys(agg.values).every(id => agg.values[id] === curatedScaled(id)),
        }}));
    """
    assert _run_node(script) == (
        '{"ids":["generation-token-rate","kv-cache-pressure","p99-ttft",'
        '"request-rate","requests-waiting"],"latencyKpiId":"p99-ttft","equal":true}'
    )


def test_aggregate_sparkline_snapshot_e2e_latency_only() -> None:
    """When only e2e-latency histograms are present, the latency tile id
    flips to p99-e2e-latency and the aggregate value matches the curator."""
    script = f"""
        import {{
          normalizeServerMetrics, curateServerMetrics, aggregateSparklineSnapshot,
        }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            vllm: {{
              type: 'gauge',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ avg: 1, max: 1 }} }}],
            }},
            'vllm:e2e_request_latency_seconds': {{
              type: 'histogram',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ count: 50, p99_estimate: 0.42 }} }}],
            }},
          }},
        }};
        const norm = normalizeServerMetrics(snapshot);
        const agg = aggregateSparklineSnapshot(norm);
        const curated = curateServerMetrics(norm);
        const e2e = curated.kpis.find(k => k.id === 'p99-e2e-latency');
        console.log(JSON.stringify({{
          latencyKpiId: agg.latencyKpiId,
          aggregatorMs: agg.values['p99-e2e-latency'] != null ? +(agg.values['p99-e2e-latency'] * 1000).toFixed(3) : null,
          curatorMs: e2e ? +e2e.value.toFixed(3) : null,
        }}));
    """
    assert _run_node(script) == (
        '{"latencyKpiId":"p99-e2e-latency","aggregatorMs":420,"curatorMs":420}'
    )


def test_aggregate_sparkline_snapshot_emits_zero_waiting() -> None:
    """Zero queue depth still produces a `requests-waiting: 0` entry so the
    rolling buffer stays continuous; the curator hides the tile by snapshot
    gate, but Task 2 will extend the curator to use the buffer instead."""
    script = f"""
        import {{
          normalizeServerMetrics, aggregateSparklineSnapshot,
        }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_requests: {{
              type: 'counter',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ rate: 1 }} }}],
            }},
            dynamo_frontend_queued_requests: {{
              type: 'gauge',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ avg: 0, max: 0 }} }}],
            }},
          }},
        }};
        const agg = aggregateSparklineSnapshot(normalizeServerMetrics(snapshot));
        console.log(JSON.stringify({{
          waitingValue: agg.values['requests-waiting'],
          hasWaitingKey: 'requests-waiting' in agg.values,
        }}));
    """
    assert _run_node(script) == '{"waitingValue":0,"hasWaitingKey":true}'


def test_aggregate_sparkline_snapshot_ttft_beats_e2e_when_both_present() -> None:
    """When both TTFT and e2e-latency histograms are present, the latency
    tile id resolves to p99-ttft (precedence: ttftHit || e2eHit)."""
    script = f"""
        import {{
          normalizeServerMetrics, aggregateSparklineSnapshot,
        }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_time_to_first_token_seconds: {{
              type: 'histogram',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ count: 10, p99_estimate: 0.05 }} }}],
            }},
            'vllm:e2e_request_latency_seconds': {{
              type: 'histogram',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ count: 10, p99_estimate: 0.5 }} }}],
            }},
          }},
        }};
        const agg = aggregateSparklineSnapshot(normalizeServerMetrics(snapshot));
        console.log(JSON.stringify({{
          latencyKpiId: agg.latencyKpiId,
          // Aggregator picked the ttft histogram, so the value is 0.05 not 0.5.
          latencyValueIsTtft: Math.abs(agg.values[agg.latencyKpiId] - 0.05) < 1e-9,
        }}));
    """
    assert _run_node(script) == (
        '{"latencyKpiId":"p99-ttft","latencyValueIsTtft":true}'
    )


def test_curate_server_metrics_attaches_sparkline_points() -> None:
    """When a sparklines map is supplied, each KPI gets its `points` array."""
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_requests: {{
              type: 'counter',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ rate: 5 }} }}],
            }},
          }},
        }};
        const sparklines = {{ 'request-rate': [{{ t: 1, v: 4 }}, {{ t: 2, v: 5 }}] }};
        const curated = curateServerMetrics(normalizeServerMetrics(snapshot), sparklines);
        const reqRate = curated.kpis.find(k => k.id === 'request-rate');
        console.log(JSON.stringify({{
          hasPoints: Array.isArray(reqRate.points),
          n: reqRate.points.length,
          firstV: reqRate.points[0].v,
        }}));
    """
    assert _run_node(script) == '{"hasPoints":true,"n":2,"firstV":4}'


def test_curate_server_metrics_requests_waiting_gate_uses_buffer() -> None:
    """Requests-waiting tile stays visible after the queue drains if any
    rolling-buffer sample was non-zero, so a transient queue earlier in
    the run keeps its tile."""
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_queued_requests: {{
              type: 'gauge',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ avg: 0, max: 0 }} }}],
            }},
          }},
        }};
        const norm = normalizeServerMetrics(snapshot);
        // No sparklines: tile hidden (current behavior).
        const curatedNoBuf = curateServerMetrics(norm);
        const noTile = !curatedNoBuf || !curatedNoBuf.kpis.some(k => k.id === 'requests-waiting');
        // Buffer has a non-zero sample: tile stays visible.
        const sparklines = {{ 'requests-waiting': [{{ t: 1, v: 4 }}, {{ t: 2, v: 0 }}] }};
        const curatedBuf = curateServerMetrics(norm, sparklines);
        const tile = curatedBuf.kpis.find(k => k.id === 'requests-waiting');
        console.log(JSON.stringify({{ noTile, hasTile: tile != null, points: tile?.points?.length ?? 0 }}));
    """
    assert _run_node(script) == '{"noTile":true,"hasTile":true,"points":2}'


def test_curate_server_metrics_latency_points_scaled_to_ms() -> None:
    """Latency tile's `points` values are scaled seconds → milliseconds to
    match the curator's displayed `value`. Aggregator stores raw seconds."""
    script = f"""
        import {{ normalizeServerMetrics, curateServerMetrics }} from {HELPERS!r};
        const snapshot = {{
          summary: {{ endpoints_configured: ['u'], endpoints_successful: ['u'] }},
          metrics: {{
            dynamo_frontend_time_to_first_token_seconds: {{
              type: 'histogram',
              series: [{{ endpoint_url: 'u', labels: {{}}, stats: {{ count: 10, p99_estimate: 0.05 }} }}],
            }},
          }},
        }};
        const sparklines = {{ 'p99-ttft': [{{ t: 1, v: 0.05 }}, {{ t: 2, v: 0.06 }}] }};
        const curated = curateServerMetrics(normalizeServerMetrics(snapshot), sparklines);
        const ttft = curated.kpis.find(k => k.id === 'p99-ttft');
        console.log(JSON.stringify({{
          // Aggregator s=0.05 → curator ms=50.
          firstMs: ttft.points[0].v,
          secondMs: ttft.points[1].v,
          // Sanity: the headline `value` is also in ms.
          valueIsMs: ttft.value === 50,
        }}));
    """
    assert _run_node(script) == '{"firstMs":50,"secondMs":60,"valueIsMs":true}'


KPI_CARD = (
    REPO / "src" / "aiperf" / "operator" / "ui" / "components" / "kpi-card-tone.js"
).as_uri()


def test_kpi_card_spark_colors_by_tone() -> None:
    """sparkColors(tone) returns stroke/fill matching the existing tile
    color signal: accent for ok/default, red for warn/bad, dim for
    neutral/ok/null. Helper lives in kpi-card-tone.js (no preact import)
    so node can resolve it without the operator UI's importmap; kpi-card.js
    re-exports it for component callers."""
    script = f"""
        import {{ sparkColors }} from {KPI_CARD!r};
        console.log(JSON.stringify({{
          accent: sparkColors('accent'),
          warn: sparkColors('warn'),
          bad: sparkColors('bad'),
          ok: sparkColors('ok'),
          neutral: sparkColors('neutral'),
          undef: sparkColors(undefined),
          nul: sparkColors(null),
          unknown: sparkColors('something-else'),
        }}));
    """
    out = _run_node(script)
    import orjson

    data = orjson.loads(out)
    # The sparkline must carry the SAME signal as the value number, i.e. mirror
    # the `.metric-val--<tone>` rules in style.css:971-977.
    assert data["accent"] == {"stroke": "var(--accent)", "fill": "var(--accent-dim)"}
    assert data["bad"] == {"stroke": "var(--red)", "fill": "rgba(239,83,80,0.15)"}
    # `warn` is amber, NOT red. The old assertion pinned red, which drew a
    # failure-coloured sparkline under an amber caution number -- escalating a
    # warning into an error purely through colour.
    assert data["warn"] == {"stroke": "var(--amber)", "fill": "rgba(255,193,7,0.15)"}
    # `ok` is green, NOT grey. The old assertion pinned grey, discarding the
    # positive signal the number was already making in green.
    assert data["ok"] == {"stroke": "var(--green)", "fill": "var(--green-dim)"}
    # Genuinely neutral inputs stay grey.
    for key in ("neutral", "undef", "nul"):
        assert data[key] == {
            "stroke": "var(--sub)",
            "fill": "rgba(167,167,167,0.10)",
        }, key
    # An unrecognised tone must make NO claim. The old assertion pinned the
    # positive accent, so any misspelled or future tone silently asserted
    # "good" -- the one direction a fallback must never assert.
    assert data["unknown"] == {"stroke": "var(--sub)", "fill": "rgba(167,167,167,0.10)"}

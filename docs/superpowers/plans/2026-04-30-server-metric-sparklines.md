# Server-metric sparklines on job detail page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live sparklines to the five server-metric KPI tiles on the AIPerf operator job-detail page, fed off the existing `realtime_server_metrics` bus message via the per-job `/ws` proxy.

**Architecture:** A new client-side aggregator collapses each `realtime_server_metrics` frame's `endpoint_summaries` into the same five numbers the existing curator produces from a snapshot, keyed by the curator's KPI ids. `lib/job-ws.js` accumulates a rolling buffer (120 pts / 5 min) per KPI id and publishes it alongside the existing live state. `KpiCard` gains an optional `sparkline` prop. `ServerMetricsSection` threads the buffer through the curator so each rendered KPI carries its own points.

**Tech Stack:** Preact + htm/preact (operator UI), pure-JS modules in `src/aiperf/operator/ui-v1/`, pytest-driven node shell-out tests under `tests/unit/ui/` (see `test_operator_server_metrics_helpers.py` for the established pattern).

**Spec:** `docs/superpowers/specs/2026-04-30-server-metric-sparklines-design.md`

---

## File map

**Modified:**
- `src/aiperf/operator/ui-v1/components/server-metrics/helpers.js` — add `aggregateSparklineSnapshot`; extend `curateServerMetrics(serverMetrics, sparklines?)` to thread `points` into each KPI; relax "requests waiting" gate when buffer has non-zero history.
- `src/aiperf/operator/ui-v1/components/kpi-card.js` — new optional `sparkline` prop; render `<Sparkline>` slot in the rich-card branch with tone-derived stroke/fill defaults.
- `src/aiperf/operator/ui-v1/components/server-metrics/index.js` — accept and pass through `sparklines` prop to the curator and on to `KpiCard`.
- `src/aiperf/operator/ui-v1/lib/job-ws.js` — add `realtime_server_metrics` to `SUBSCRIBE_TYPES`; accumulator + buffer; extend `publish()` payload with `serverSummary` and `serverTimeseries`.
- `src/aiperf/operator/ui-v1/pages/job-detail.js` — initial `liveData` state shape gains `serverSummary`/`serverTimeseries`; live overlay prefers `liveData.serverSummary` over `status.serverMetrics`; new `sparklines` prop on `<ServerMetricsSection>`.

**Created (tests):**
- `tests/unit/ui/test_operator_server_metric_sparklines.py` — node-shell-out unit tests for `aggregateSparklineSnapshot`, curator thread-through, and the `KpiCard` sparkline-slot render.
- `tests/unit/ui/test_operator_job_ws_server_metrics.py` — node-shell-out unit test for the `lib/job-ws.js` accumulator under fake-WebSocket frames.

---

## Task 1: `aggregateSparklineSnapshot` in `helpers.js`

**Files:**
- Modify: `src/aiperf/operator/ui-v1/components/server-metrics/helpers.js`
- Test: `tests/unit/ui/test_operator_server_metric_sparklines.py` (create)

- [ ] **Step 1: Write the failing test (test file create)**

Create `tests/unit/ui/test_operator_server_metric_sparklines.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HELPERS = (
    REPO / "src" / "aiperf" / "operator" / "ui-v1" / "components" / "server-metrics" / "helpers.js"
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
    would produce from the same snapshot."""
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
        console.log(JSON.stringify({{
          ids: Object.keys(agg.values).sort(),
          latencyKpiId: agg.latencyKpiId,
          // Aggregator values must equal the curator's values for the same ids.
          equal: Object.keys(agg.values).every(id => agg.values[id] === curatedById[id]),
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py -n auto -v
```

Expected: both tests fail with "aggregateSparklineSnapshot is not exported" (SyntaxError from node).

- [ ] **Step 3: Implement `aggregateSparklineSnapshot` in `helpers.js`**

At the bottom of `src/aiperf/operator/ui-v1/components/server-metrics/helpers.js`, add:

```js
/**
 * Collapse a single normalized server-metrics snapshot into the five
 * numbers the server-metrics KPI tiles surface, keyed by the same KPI ids
 * `curateServerMetrics` produces. Used by the per-job WS layer to push
 * one sample per scrape into a per-KPI rolling buffer.
 *
 * The latency tile id flips between `'p99-ttft'` and `'p99-e2e-latency'`
 * depending on which histogram is present in the snapshot; the resolved id
 * is returned alongside the values so the WS layer can key its buffer
 * consistently.
 *
 * Note: the latency value is returned in seconds (the raw histogram unit);
 * the curator multiplies by 1000 for display. Keep the buffer in seconds
 * and let the rendering layer scale on read.
 *
 * @param {object} normalizedServerMetrics - shape produced by
 *   `normalizeServerMetrics`
 * @returns {{ values: Object<string, number>, latencyKpiId: string|null }}
 */
export function aggregateSparklineSnapshot(normalizedServerMetrics) {
  const out = { values: {}, latencyKpiId: null };
  if (!normalizedServerMetrics) return out;
  const metrics = normalizedServerMetrics.metrics ?? {};
  const backendsPresent = detectBackends(metrics);

  const reqHit = pickBestMetricHit(metrics, backendsPresent, 'reqRate');
  const genHit = pickBestMetricHit(metrics, backendsPresent, 'genTokRate');
  const kvHit = pickBestMetricHit(metrics, backendsPresent, 'kvCachePct');
  const ttftHit = pickBestMetricHit(metrics, backendsPresent, 'ttft');
  const e2eHit = pickBestMetricHit(metrics, backendsPresent, 'e2eLatency');
  const waitHit = pickBestMetricHit(metrics, backendsPresent, 'requestsWaiting');
  const latencyHit = ttftHit || e2eHit;

  const reqRate = reqHit ? sumOf(metrics[reqHit.name], reqHit.statField) : null;
  const genRate = genHit ? aggregateForHit(metrics, genHit) : null;
  const kvPeak = kvHit ? normalizePercent(maxOf(metrics[kvHit.name], 'max')) : null;
  const latencyP99 = latencyHit ? histogramStat(metrics[latencyHit.name], 'p99_estimate') : null;
  const waitingAvg = waitHit ? avgOf(metrics[waitHit.name], 'avg') : null;

  // For aggregator → curator equality the latency value must be in the same
  // unit as `curateServerMetrics` returns it. The curator multiplies by 1000
  // for ms display, so the *aggregator* keeps the raw seconds and the
  // sparkline buffer stays in seconds; consumers that want ms scale on read.
  const ifNum = (v) => (typeof v === 'number' && isFinite(v) ? v : null);

  if (ifNum(reqRate) != null) out.values['request-rate'] = reqRate;
  if (ifNum(genRate) != null) out.values['generation-token-rate'] = genRate;
  if (ifNum(kvPeak) != null) out.values['kv-cache-pressure'] = kvPeak;
  if (ifNum(waitingAvg) != null) out.values['requests-waiting'] = waitingAvg;
  if (ifNum(latencyP99) != null) {
    out.latencyKpiId = latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency';
    out.values[out.latencyKpiId] = latencyP99;
  }
  return out;
}
```

- [ ] **Step 4: Verify test 2 (e2e-latency) passes; fix curator/aggregator unit mismatch for test 1**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py::test_aggregate_sparkline_snapshot_e2e_latency_only -n auto -v
```

Test 1 will fail because `curateServerMetrics` returns `latencyP99 * 1000` (ms) while the aggregator returns seconds. The test was written to assert seconds for both. Fix the test 1 assertion path: the test compares aggregator value to `curatedById[id]` directly — for the latency id those won't match. Update the test 1 assertion to scale curator latency back to seconds before comparing:

```js
// at the start of the equality check, replace curatedById[id] with:
const curatedScaled = (id) =>
  (id === 'p99-ttft' || id === 'p99-e2e-latency') ? curatedById[id] / 1000 : curatedById[id];
// and change the .every() to use curatedScaled(id):
equal: Object.keys(agg.values).every(id => agg.values[id] === curatedScaled(id)),
```

Apply the fix to the test 1 script string in `tests/unit/ui/test_operator_server_metric_sparklines.py`.

- [ ] **Step 5: Run both tests to verify pass**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py -n auto -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/server-metrics/helpers.js \
        tests/unit/ui/test_operator_server_metric_sparklines.py
git commit -s -m "feat(ui): aggregateSparklineSnapshot for server-metrics

Per-snapshot collapse of endpoint_summaries to the same five KPI numbers
curateServerMetrics produces, keyed by the curator's KPI ids so the
client-side rolling buffer can join cleanly."
```

---

## Task 2: Thread `sparklines` through `curateServerMetrics`

**Files:**
- Modify: `src/aiperf/operator/ui-v1/components/server-metrics/helpers.js`
- Test: `tests/unit/ui/test_operator_server_metric_sparklines.py` (extend)

- [ ] **Step 1: Add failing test for sparklines pass-through**

Append to `tests/unit/ui/test_operator_server_metric_sparklines.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py -n auto -v -k "attaches_sparkline_points or requests_waiting_gate"
```

Expected: both FAIL — `points` is `undefined` on KPIs and the gate ignores the second arg.

- [ ] **Step 3: Extend `curateServerMetrics` in `helpers.js`**

Modify the existing signature and gate logic. In `helpers.js`:

a. Update `makeKpi` to accept and pass through `points`:

```js
function makeKpi({ id, label, value, unit, source, stat, icon, tone = 'accent', progress = null, sub = null, points = null }) {
  if (value == null) return null;
  return { id, label, value, unit, source, stat, icon, tone, progress, sub, points };
}
```

b. Change `curateServerMetrics(serverMetrics)` to `curateServerMetrics(serverMetrics, sparklines = null)`:

```js
export function curateServerMetrics(serverMetrics, sparklines = null) {
```

c. Inside the function, after computing `waitingAvg`/`waitingPeak`, derive a buffer-aware "ever non-zero" signal for the gate:

```js
  const waitingPoints = sparklines?.['requests-waiting'] ?? null;
  const waitingBufHasNonzero = Array.isArray(waitingPoints)
    && waitingPoints.some(p => typeof p?.v === 'number' && p.v > 0);
```

d. Update the gate at the existing tile:

```js
    waitingAvg != null && (((waitingPeak ?? waitingAvg) > 0) || waitingBufHasNonzero) ? makeKpi({
      id: 'requests-waiting',
      label: 'Requests waiting',
      value: waitingAvg,
      // ...existing fields unchanged...
      points: waitingPoints,
    }) : null,
```

e. Pass `points` into each existing `makeKpi` call:

```js
    makeKpi({ id: 'request-rate', /* ...existing... */, points: sparklines?.['request-rate'] ?? null }),
    makeKpi({ id: 'generation-token-rate', /* ...existing... */, points: sparklines?.['generation-token-rate'] ?? null }),
    makeKpi({ id: 'kv-cache-pressure', /* ...existing... */, points: sparklines?.['kv-cache-pressure'] ?? null }),
    makeKpi({
      id: latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency',
      // ...existing fields unchanged...
      points: latencyHit
        ? (sparklines?.[latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency'] ?? null)
        : null,
    }),
```

f. The latency tile's points are stored as raw seconds in the buffer (per Task 1 unit choice); the curator's `value` field stays in ms. Sparkline rendering will scale at draw time. Add a transform here so points-on-the-KPI are also in ms for consistency with the displayed value:

Replace the latency `points` line in (e) with:

```js
      points: (() => {
        const latId = latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency';
        const raw = sparklines?.[latId];
        if (!Array.isArray(raw)) return null;
        return raw.map(p => ({ t: p.t, v: p.v * 1000 }));
      })(),
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py -n auto -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/server-metrics/helpers.js \
        tests/unit/ui/test_operator_server_metric_sparklines.py
git commit -s -m "feat(ui): thread sparkline points through curateServerMetrics

Optional second arg, keyed by KPI id. Latency points scaled s→ms to match
the displayed value. Requests-waiting tile stays visible if any buffered
sample was non-zero (transient queue earlier in the run)."
```

---

## Task 3: `KpiCard` sparkline slot

**Files:**
- Modify: `src/aiperf/operator/ui-v1/components/kpi-card.js`
- Test: `tests/unit/ui/test_operator_server_metric_sparklines.py` (extend)

- [ ] **Step 1: Add failing render test**

Append to `tests/unit/ui/test_operator_server_metric_sparklines.py`:

```python
KPI_CARD = (
    REPO / "src" / "aiperf" / "operator" / "ui-v1" / "components" / "kpi-card.js"
).as_uri()


def test_kpi_card_renders_sparkline_when_points_provided() -> None:
    """KpiCard with `sparkline` prop emits a Sparkline child whose stroke
    follows the tile tone."""
    script = f"""
        import {{ render }} from 'preact-render-to-string';
        import {{ html }} from 'htm/preact';
        import {{ KpiCard }} from {KPI_CARD!r};
        const points = [{{ t: 1, v: 1 }}, {{ t: 2, v: 2 }}];
        const ok = render(html`<${{KpiCard}} label="Req/s" value="5" unit="req/s"
                                              icon="speed" tone="accent"
                                              sparkline=${{ {{ points }} }} />`);
        const warn = render(html`<${{KpiCard}} label="KV" value="92" unit="%"
                                                icon="goodput" tone="warn"
                                                sparkline=${{ {{ points }} }} />`);
        console.log(JSON.stringify({{
          okHasSpark: ok.includes('class="sparkline"'),
          okStroke: /stroke="var\\(--accent\\)"/.test(ok),
          warnStroke: /stroke="var\\(--red\\)"/.test(warn),
        }}));
    """
    # Run with npm-installed deps already in node_modules; fail with a clear
    # message if preact-render-to-string isn't available.
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO / "src" / "aiperf" / "operator" / "ui-v1"),
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    assert result.stdout.strip() == '{"okHasSpark":true,"okStroke":true,"warnStroke":true}'
```

Note: the existing `helpers.js` tests use bare `node` because they only import standalone modules; this test imports `htm/preact` and `preact-render-to-string`, which need a `node_modules`. The `cwd=` points at the UI module so node's resolution finds the operator UI's `node_modules`.

- [ ] **Step 2: Verify render dependency exists**

```bash
ls src/aiperf/operator/ui-v1/node_modules/preact-render-to-string 2>&1 | head -1
```

If the directory is missing, the test setup must install it. Check `src/aiperf/operator/ui-v1/package.json` for the dep; if absent:

```bash
cd src/aiperf/operator/ui-v1 && npm install --save-dev preact-render-to-string
```

If `package.json` doesn't exist either, fall back to a string-content test that re-implements rendering by checking the `KpiCard` return value's tag tree — but only if needed. Run `ls src/aiperf/operator/ui-v1/package.json` first; if it exists, prefer the install path.

- [ ] **Step 3: Run test to verify failure**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py::test_kpi_card_renders_sparkline_when_points_provided -n auto -v
```

Expected: FAIL — KpiCard ignores `sparkline` prop.

- [ ] **Step 4: Add `sparkline` prop to `KpiCard`**

In `src/aiperf/operator/ui-v1/components/kpi-card.js`:

a. Add the import at the top:

```js
import { Sparkline } from './sparkline.js';
```

b. Helper function (above `KpiCard`):

```js
function sparkColors(tone) {
  switch (tone) {
    case 'warn': return { stroke: 'var(--red)', fill: 'rgba(239,83,80,0.15)' };
    case 'bad':  return { stroke: 'var(--red)', fill: 'rgba(239,83,80,0.15)' };
    case 'ok':
    case 'neutral':
    case undefined:
    case null:
      return { stroke: 'var(--sub)', fill: 'rgba(167,167,167,0.10)' };
    default:
      return { stroke: 'var(--accent)', fill: 'var(--accent-dim)' };
  }
}
```

c. Extend the `KpiCard` signature and body:

```js
export function KpiCard({
  label,
  value,
  unit,
  color,
  sub,
  title,
  icon,
  tone,
  progress,
  progressTone,
  sparkline,
}) {
```

d. In the rich-card branch (the second `return html`...), insert the sparkline below `metric-card__row`:

```js
      ${sparkline?.points?.length > 1 && (() => {
        const sc = sparkColors(tone);
        return html`<${Sparkline}
                      points=${sparkline.points}
                      stroke=${sparkline.stroke ?? sc.stroke}
                      fill=${sparkline.fill ?? sc.fill}
                      width=${140} height=${26} />`;
      })()}
```

e. The legacy bare-card branch is intentionally left untouched (no caller passes `sparkline` without `icon`).

- [ ] **Step 5: Run test to verify pass**

```bash
uv run pytest tests/unit/ui/test_operator_server_metric_sparklines.py::test_kpi_card_renders_sparkline_when_points_provided -n auto -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/kpi-card.js \
        tests/unit/ui/test_operator_server_metric_sparklines.py
[ -f src/aiperf/operator/ui-v1/package.json ] && git add src/aiperf/operator/ui-v1/package.json src/aiperf/operator/ui-v1/package-lock.json 2>/dev/null
git commit -s -m "feat(ui): KpiCard optional sparkline slot

Renders a 140x26 sparkline below the value row when \`sparkline.points\`
is provided. Stroke/fill default off the tile tone (accent / warn / red)
so the live trend tracks the existing color signal."
```

---

## Task 4: `lib/job-ws.js` accumulator for `realtime_server_metrics`

**Files:**
- Modify: `src/aiperf/operator/ui-v1/lib/job-ws.js`
- Test: `tests/unit/ui/test_operator_job_ws_server_metrics.py` (create)

- [ ] **Step 1: Write the failing accumulator test**

Create `tests/unit/ui/test_operator_job_ws_server_metrics.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drives lib/job-ws.js with a fake WebSocket implementation under node and
asserts the per-KPI rolling buffer gains samples for each `realtime_server_metrics`
frame, keyed by curator KPI ids.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JOB_WS = (
    REPO / "src" / "aiperf" / "operator" / "ui-v1" / "lib" / "job-ws.js"
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


def test_job_ws_accumulates_server_metric_samples() -> None:
    """Two `realtime_server_metrics` frames produce two-point buffers per KPI."""
    script = f"""
        // Stub the global WebSocket so openJobWs's connect() is exercised
        // synchronously; we drive onmessage manually.
        const fakeSockets = [];
        globalThis.WebSocket = class {{
          constructor(url) {{ this.url = url; fakeSockets.push(this); }}
          send() {{}}
          close() {{ this.onclose && this.onclose(); }}
        }};
        globalThis.window = {{ location: {{ protocol: 'http:', host: 'x' }} }};
        const {{ openJobWs }} = await import({JOB_WS!r});
        let last = null;
        const handle = openJobWs('ns', 'name', (snap) => {{ last = snap; }});
        const sock = fakeSockets[0];
        sock.onopen && sock.onopen();
        // Hand two minimal realtime_server_metrics frames, ~1 s apart.
        const frame1 = {{
          type: 'realtime_server_metrics',
          endpoint_summaries: {{
            'h1:9090': {{
              endpoint_url: 'http://h1:9090/metrics',
              metrics: {{
                dynamo_frontend_requests: {{
                  type: 'counter',
                  series: [{{ endpoint_url: 'http://h1:9090/metrics', labels: {{}}, stats: {{ rate: 4 }} }}],
                }},
              }},
            }},
          }},
        }};
        const frame2 = JSON.parse(JSON.stringify(frame1));
        frame2.endpoint_summaries['h1:9090'].metrics.dynamo_frontend_requests
              .series[0].stats.rate = 6;
        sock.onmessage({{ data: JSON.stringify(frame1) }});
        sock.onmessage({{ data: JSON.stringify(frame2) }});
        const ts = last.serverTimeseries['request-rate'];
        console.log(JSON.stringify({{
          hasSummary: last.serverSummary != null && typeof last.serverSummary === 'object',
          n: ts.length,
          values: ts.map(p => p.v),
          monotonic: ts[0].t <= ts[1].t,
        }}));
        handle.close();
    """
    assert _run_node(script) == '{"hasSummary":true,"n":2,"values":[4,6],"monotonic":true}'
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/ui/test_operator_job_ws_server_metrics.py -n auto -v
```

Expected: FAIL — `serverTimeseries` is `undefined` on the published snapshot.

- [ ] **Step 3: Wire the new subscribe + accumulator in `lib/job-ws.js`**

In `src/aiperf/operator/ui-v1/lib/job-ws.js`:

a. At the top, alongside the existing imports, add:

```js
import {
  normalizeServerMetrics,
  aggregateSparklineSnapshot,
} from '../components/server-metrics/helpers.js';
```

b. Update the subscribe list:

```js
const SUBSCRIBE_TYPES = ['realtime_metrics', 'realtime_server_metrics'];
```

c. Inside `openJobWs`, alongside `summary` / `timeseries` declarations, add:

```js
  let serverSummary = null;
  const serverTimeseries = {};
```

d. Add an accumulator helper next to `pushSample`:

```js
  function pushServerSample(kpiId, t, v) {
    const series = serverTimeseries[kpiId] ?? [];
    const cutoff = t - MAX_AGE_MS;
    const next = series.filter(s => s.t >= cutoff);
    next.push({ t, v });
    if (next.length > MAX_POINTS) next.splice(0, next.length - MAX_POINTS);
    serverTimeseries[kpiId] = next;
  }

  function applyRealtimeServerMetrics(payload) {
    if (!payload || typeof payload !== 'object') return;
    serverSummary = payload;
    const normalized = normalizeServerMetrics(payload);
    const { values } = aggregateSparklineSnapshot(normalized);
    const t = Date.now();
    for (const [kpiId, v] of Object.entries(values)) {
      if (typeof v === 'number' && isFinite(v)) pushServerSample(kpiId, t, v);
    }
  }
```

e. Extend `handleMessage` to dispatch the new type:

```js
    if (type === 'realtime_metrics' && Array.isArray(msg.metrics)) {
      applyRealtimeMetrics(msg.metrics);
      publish(true);
    } else if (type === 'realtime_server_metrics') {
      applyRealtimeServerMetrics(msg.endpoint_summaries
        ? msg
        : msg.payload ?? null);
      publish(true);
    }
```

f. Extend `publish()` to expose the new fields:

```js
  function publish(connected) {
    onUpdate({
      summary: { ...summary },
      timeseries: { ...timeseries },
      serverSummary,
      serverTimeseries: { ...serverTimeseries },
      connected,
    });
  }
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/unit/ui/test_operator_job_ws_server_metrics.py -n auto -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui-v1/lib/job-ws.js \
        tests/unit/ui/test_operator_job_ws_server_metrics.py
git commit -s -m "feat(ui): job-ws accumulator for realtime_server_metrics

Subscribes to realtime_server_metrics, normalizes each frame, runs
aggregateSparklineSnapshot, and pushes per-KPI samples into a 120pt /
5min rolling buffer. Published as serverTimeseries alongside the
existing summary/timeseries state."
```

---

## Task 5: Wire sparklines through `ServerMetricsSection` and the page

**Files:**
- Modify: `src/aiperf/operator/ui-v1/components/server-metrics/index.js`
- Modify: `src/aiperf/operator/ui-v1/pages/job-detail.js`

- [ ] **Step 1: Thread `sparklines` through `ServerMetricsSection`**

In `src/aiperf/operator/ui-v1/components/server-metrics/index.js`:

a. Update signature:

```js
export function ServerMetricsSection({ serverMetrics, source = 'final', sparklines = null }) {
```

b. Update the curator call:

```js
  const curated = curateServerMetrics(normalizeServerMetrics(serverMetrics), sparklines);
```

c. In the KPI loop, pass the sparkline points to `KpiCard`:

```js
            ${curated.kpis.map(kpi => html`
              <${KpiCard}
                key=${kpi.id}
                label=${kpi.label}
                icon=${kpi.icon}
                tone=${kpi.tone}
                value=${formatKpiValue(kpi)}
                unit=${kpi.unit}
                sub=${kpi.sub}
                progress=${kpi.progress}
                title=${kpi.source ? `Source: ${kpi.source} (${kpi.stat})` : ''}
                sparkline=${kpi.points && kpi.points.length > 1 ? { points: kpi.points } : null}
              />
            `)}
```

- [ ] **Step 2: Update `pages/job-detail.js` live state shape and overlay**

In `src/aiperf/operator/ui-v1/pages/job-detail.js`:

a. Update the initial `liveData` (line ~1651):

```js
  const [liveData, setLiveData] = useState({
    summary: {}, timeseries: {}, serverSummary: null, serverTimeseries: {}, connected: false,
  });
```

b. Update the WS-deactivation reset block (line ~1663):

```js
      setLiveData({ summary: {}, timeseries: {}, serverSummary: null, serverTimeseries: {}, connected: false });
```

c. Find the existing `liveServerMetrics` line (~1919):

```js
  const liveServerMetrics = epoch === undefined ? status.serverMetrics : null;
```

Replace with WS-preferred overlay:

```js
  const liveServerMetricsBase = epoch === undefined ? status.serverMetrics : null;
  const liveServerMetrics = (liveData.connected && liveData.serverSummary)
    ? liveData.serverSummary
    : liveServerMetricsBase;
```

d. Find the `<ServerMetricsSection>` render (line ~2265) and pass the new prop:

```js
        ? html`<${ServerMetricsSection}
                 serverMetrics=${displayedServerMetrics}
                 source=${serverMetricsSource}
                 sparklines=${epoch === undefined ? liveData.serverTimeseries : null} />`
```

- [ ] **Step 3: Run all UI unit tests + a quick lint**

```bash
uv run pytest tests/unit/ui/ -n auto -v
ruff format src/aiperf/operator/ui-v1/ tests/unit/ui/ && ruff check --fix src/aiperf/operator/ui-v1/ tests/unit/ui/
```

Expected: all tests PASS; ruff clean. (Ruff only touches the Python test files; the JS isn't ruff-managed.)

- [ ] **Step 4: Manual UI verification (browser)**

The harness can't render the live UI; ask the user to open a running AIPerfJob detail page and confirm:
- Five server-metric tiles each show a sparkline that updates on the WS cadence (~1 Hz).
- KV cache tile retains the progress bar AND shows a sparkline.
- Tile colors match tone (KV cache flips red over 90%, sparkline follows).
- After the run completes and the page navigates to an epoch, the strip shows static numbers and empty sparkline placeholders (no errors in console).

Tell the user explicitly: "I cannot test the live UI from here — please verify the five sparklines render and update on a running job."

- [ ] **Step 5: Commit**

```bash
git add src/aiperf/operator/ui-v1/components/server-metrics/index.js \
        src/aiperf/operator/ui-v1/pages/job-detail.js
git commit -s -m "feat(ui): live server-metric sparklines on job detail

Routes liveData.serverTimeseries from the per-job WS through
ServerMetricsSection into each KpiCard. WS overlay also takes precedence
over the polled status.serverMetrics for the headline numbers when
connected, matching the AIPerf-side strip's live behavior."
```

---

## Self-review

Spec coverage:
- WS subscribe + accumulator: Task 4. ✓
- `aggregateSparklineSnapshot`: Task 1. ✓
- `curateServerMetrics(serverMetrics, sparklines?)` thread + requests-waiting gate + latency-id pick + s→ms scaling: Task 2. ✓
- `KpiCard` sparkline slot + tone-derived colors: Task 3. ✓
- `ServerMetricsSection` plumbing + page-level overlay + epoch-view gating: Task 5. ✓
- Edge cases (KV bar+sparkline coexistence, requests-waiting gate, latency-id flip, WS reconnect, page hide, run termination): all surfaced in tasks (Task 2 gate test; Task 5 epoch-gating wiring; Task 4 reuses the existing reconnect loop).
- Testing scope (aggregator equality, gate, KpiCard render, accumulator): Tasks 1–4. ✓

Placeholder scan: no TBD, no "implement later", every code step shows code.

Type/name consistency:
- KPI ids `'request-rate'`, `'generation-token-rate'`, `'kv-cache-pressure'`, `'p99-ttft'` / `'p99-e2e-latency'`, `'requests-waiting'` are used identically in Tasks 1, 2, 4.
- `sparkline` prop shape `{ points: [{t,v}], stroke?, fill? }` is used identically in Tasks 3 and 5.
- `liveData` field names `serverSummary` / `serverTimeseries` match across Tasks 4 and 5.

No issues found.

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-04-30-server-metric-sparklines.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?

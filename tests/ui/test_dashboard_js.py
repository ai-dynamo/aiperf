# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node-based tests for the AIPerf API dashboards: the legacy
``static/dashboard.html`` and the modular ``static-v2/`` dashboard.

These are the cheap layers - ``node --check`` on the extracted inline script
and on every ``static-v2/`` ES module, Node-executed adversarial coverage of
the v2 modules, and FastAPI ``TestClient`` coverage of the static-asset route.
No DOM and no browser; they skip when ``node`` is not on PATH.

The browser-driven (Playwright/Chromium) dashboard tests live in
``tests/ui_e2e/dashboard/test_dashboard_render.py``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import param

from aiperf.api.routers.static import static_router

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_HTML = _REPO_ROOT / "src" / "aiperf" / "api" / "static" / "dashboard.html"


# -----------------------------------------------------------------------------
# Runtime availability
# -----------------------------------------------------------------------------


def _node_binary() -> str | None:
    return shutil.which("node")


_NODE_REASON = "node binary not on PATH"


# -----------------------------------------------------------------------------
# Inline-script helpers
# -----------------------------------------------------------------------------


def _extract_inline_js(html: str) -> str:
    """Return the content of the single inline ``<script>...</script>`` block."""
    match = re.search(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    assert match is not None, "dashboard.html must contain exactly one inline <script>"
    return match.group(1)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestDashboardInlineJS:
    """Inline-JS checks that don't need a DOM."""

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_inline_js_parses(self, tmp_path: Path) -> None:
        """``node --check`` on the extracted inline script."""
        html = _DASHBOARD_HTML.read_text()
        js = _extract_inline_js(html)
        js_path = tmp_path / "dashboard_inline.js"
        js_path.write_text(js)
        proc = subprocess.run(
            [_node_binary(), "--check", str(js_path)],
            capture_output=True,
            timeout=15,
        )
        assert proc.returncode == 0, (
            f"inline JS failed `node --check`:\n{proc.stderr.decode(errors='replace')}"
        )


# -----------------------------------------------------------------------------
# v2 dashboard (src/aiperf/api/static-v2/) - Preact/htm/signals stack
# -----------------------------------------------------------------------------

_STATIC_V2_DIR = _REPO_ROOT / "src" / "aiperf" / "api" / "static-v2"


def _v2_js_files() -> list[Path]:
    """All ES modules shipped by the v2 dashboard."""
    return sorted(_STATIC_V2_DIR.rglob("*.js"))


def _run_v2_node_script(tmp_path: Path, script: str) -> dict[str, Any]:
    """Run a dashboard-v2 ES-module script with tiny browser-dependency stubs."""
    node = _node_binary()
    assert node is not None, _NODE_REASON

    sandbox = tmp_path / "dashboard-v2-node"
    shutil.copytree(_STATIC_V2_DIR, sandbox)
    (sandbox / "package.json").write_text('{"type":"module"}\n')

    signals_dir = sandbox / "node_modules" / "@preact" / "signals"
    signals_dir.mkdir(parents=True)
    (signals_dir / "package.json").write_text('{"type":"module","main":"./index.js"}\n')
    (signals_dir / "index.js").write_text(
        "export function signal(value) { return { value }; }\n"
    )

    htm_dir = sandbox / "node_modules" / "htm"
    htm_dir.mkdir(parents=True)
    (htm_dir / "package.json").write_text(
        '{"type":"module","exports":{"./preact":"./preact.js"}}\n'
    )
    (htm_dir / "preact.js").write_text(
        "export function html(strings, ...values) { return { strings, values }; }\n"
    )

    preact_dir = sandbox / "node_modules" / "preact"
    preact_dir.mkdir(parents=True)
    (preact_dir / "package.json").write_text(
        '{"type":"module","exports":{"./hooks":"./hooks.js"}}\n'
    )
    (preact_dir / "hooks.js").write_text(
        "export function useState(value) { return [value, () => {}]; }\n"
    )

    script_path = sandbox / "adversarial-test.mjs"
    script_path.write_text(script)
    proc = subprocess.run(
        [node, str(script_path)],
        capture_output=True,
        timeout=15,
        text=True,
        cwd=sandbox,
    )
    assert proc.returncode == 0, (
        f"node dashboard-v2 module test failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout)


@pytest.fixture
def _static_client() -> TestClient:
    app = FastAPI(title="aiperf-static-test")
    app.include_router(static_router)
    return TestClient(app)


class TestDashboardV2StaticServing:
    """Route-level coverage for the API server's static-v2 asset handler."""

    def test_dashboard_v2_without_trailing_slash_redirects(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/dashboard-v2/"

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            param("/dashboard-v2/", "text/html", id="index"),
            param("/dashboard-v2/app.js", "application/javascript", id="js"),
            param("/dashboard-v2/style.css", "text/css", id="css"),
        ],
    )  # fmt: skip
    def test_dashboard_v2_serves_expected_content_types(
        self, _static_client: TestClient, path: str, expected: str
    ) -> None:
        response = _static_client.get(path)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(expected)

    @pytest.mark.parametrize(
        "path",
        [
            param("/dashboard-v2/%2E%2E/static/dashboard.html", id="encoded-dot-dot"),
            param("/dashboard-v2/%2e%2e/static/dashboard.html", id="encoded-dot-dot-lowercase"),
        ],
    )  # fmt: skip
    def test_dashboard_v2_rejects_path_traversal(
        self, _static_client: TestClient, path: str
    ) -> None:
        response = _static_client.get(path)

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid asset path"

    def test_dashboard_v2_missing_asset_returns_404(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2/no-such-asset.js")

        assert response.status_code == 404
        assert response.json()["detail"] == "no-such-asset.js not found"

    def test_dashboard_v2_index_serves_app_shell(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<title>AIPerf Dashboard</title>" in response.text
        assert 'src="./app.js"' in response.text
        assert 'href="./style.css"' in response.text

    def test_dashboard_v2_does_not_serve_operator_ui_assets(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2/components/artifacts-card.js")

        assert response.status_code == 404
        assert response.json()["detail"] == "components/artifacts-card.js not found"


class TestDashboardV2InlineJS:
    """Cheap syntax gates that don't need a browser."""

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_v2_js_modules_parse(self) -> None:
        """``node --check`` each .js file under ``static-v2/``.

        Catches syntax regressions across the lib/ and components/ split
        without needing jsdom or a browser.
        """
        files = _v2_js_files()
        assert files, "no v2 JS modules found; static-v2/ is missing files"
        failures: list[str] = []
        for path in files:
            proc = subprocess.run(
                [_node_binary(), "--check", str(path)],
                capture_output=True,
                timeout=15,
            )
            if proc.returncode != 0:
                failures.append(
                    f"{path.relative_to(_STATIC_V2_DIR)}:\n  "
                    + proc.stderr.decode(errors="replace").replace("\n", "\n  ")
                )
        assert not failures, "v2 JS files failed node --check:\n" + "\n\n".join(
            failures
        )


class TestDashboardV2ModuleAdversarial:
    """Node-backed adversarial coverage for dashboard-v2 modules."""

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_formatters_use_fallbacks_for_non_finite_values(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { fmtInt, fmtNumber, fmtPercent } from './lib/format.js';
            console.log(JSON.stringify({
              numberNaN: fmtNumber(Number.NaN),
              numberInfinity: fmtNumber(Number.POSITIVE_INFINITY),
              numberString: fmtNumber('42'),
              intNaN: fmtInt(Number.NaN),
              percentNaN: fmtPercent(Number.NaN),
              valid: fmtNumber(1234.567, 2),
            }));
            """,
        )

        assert result == {
            "numberNaN": "---",
            "numberInfinity": "---",
            "numberString": "---",
            "intNaN": "---",
            "percentNaN": "---",
            "valid": "1,234.57",
        }

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_timeseries_and_sparkline_ignore_non_finite_inputs(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { pushSample, pluck } from './lib/timeseries.js';
            import { Sparkline } from './components/sparkline.js';
            const series = pushSample([
              { t: 3000, values: { avg: 3 } },
              { t: Number.NaN, values: { avg: 99 } },
              { t: 1000, values: { avg: 1 } },
            ], { t: 2000, values: { avg: Number.POSITIVE_INFINITY, current: 2 } });
            const points = pluck(series, 'avg');
            const spark = Sparkline({ points: [
              { t: 2, v: 2 },
              { t: Number.NaN, v: 5 },
              { t: 1, v: 1 },
              { t: 3, v: Number.POSITIVE_INFINITY },
            ] });
            console.log(JSON.stringify({
              series: series.map(s => s.t),
              points,
              hasNaN: JSON.stringify(spark).includes('NaN'),
              hasInfinity: JSON.stringify(spark).includes('Infinity'),
            }));
            """,
        )

        assert result["series"] == [1000, 2000, 3000]
        assert result["points"] == [{"t": 1000, "v": 1}, {"t": 3000, "v": 3}]
        assert result["hasNaN"] is False
        assert result["hasInfinity"] is False

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_phase_dispatch_preserves_terminal_and_failed_states(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { phases, logs } from './lib/state.js';
            phases.value = {};
            logs.value = [];
            handleWsMessage({
              type: 'credit_phase_complete',
              phase: 'profiling',
              stats: { start_ns: 1, requests_end_ns: 10, total_expected_requests: 100, final_requests_completed: 100 },
            });
            handleWsMessage({
              type: 'credit_phase_progress',
              phase: 'profiling',
              stats: { start_ns: 1, total_expected_requests: 100, requests_completed: 60 },
            });
            handleWsMessage({
              type: 'credit_phase_failed',
              phase: 'cleanup',
              stats: { start_ns: 1, requests_end_ns: 2, request_errors: 28 },
            });
            console.log(JSON.stringify({ phases: phases.value, logs: logs.value }));
            """,
        )

        assert result["phases"]["profiling"]["complete"] is True
        assert result["phases"]["profiling"]["active"] is False
        assert result["phases"]["profiling"]["final_requests_completed"] == 100
        assert result["phases"]["cleanup"]["failed"] is True
        assert result["phases"]["cleanup"]["complete"] is False
        assert any(
            "Phase failed: cleanup" in entry["message"] for entry in result["logs"]
        )

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_server_metrics_render_non_finite_values_as_fallback(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { ServerMetrics } from './components/server-metrics.js';
            import { serverMetrics } from './lib/state.js';
            serverMetrics.value = [{ endpoint: 'srv-a', metrics: [
              { name: 'kv_cache_utilization', value: Number.POSITIVE_INFINITY, unit: '%' },
              { name: 'queue_depth', value: Number.NaN, unit: 'requests' },
              { name: 'goodput', value: 1234.56, unit: 'req/s' },
            ] }];
            const rendered = JSON.stringify(ServerMetrics());
            console.log(JSON.stringify({
              rendered,
              hasInfinity: rendered.includes('Infinity'),
              hasNaN: rendered.includes('NaN'),
              fallbackCount: (rendered.match(/---/g) || []).length,
            }));
            """,
        )

        assert result["hasInfinity"] is False
        assert result["hasNaN"] is False
        assert result["fallbackCount"] >= 2
        assert "1,235 req/s" in result["rendered"]

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_server_metrics_normalization_preserves_full_stats(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { normalizeEndpointSummaries } from './lib/ws-dispatch.js';
            const summaries = normalizeEndpointSummaries({
              'http://srv:8000': {
                metrics: {
                  kv_cache_utilization: {
                    unit: 'ratio',
                    series: [{ stats: { avg: 0.92, min: 0.70, max: 0.99, p99: 0.98, p90: 0.95, p50: 0.90 } }],
                  },
                  tokens_total: {
                    unit: 'tokens',
                    series: [{ stats: { value: 125000 } }],
                  },
                },
              },
            });
            console.log(JSON.stringify(summaries));
            """,
        )

        metrics = {m["name"]: m for m in result[0]["metrics"]}
        assert metrics["kv_cache_utilization"] == {
            "name": "kv_cache_utilization",
            "value": 0.92,
            "unit": "ratio",
            "avg": 0.92,
            "min": 0.70,
            "max": 0.99,
            "p99": 0.98,
            "p90": 0.95,
            "p50": 0.90,
        }
        assert metrics["tokens_total"] == {
            "name": "tokens_total",
            "value": 125000,
            "unit": "tokens",
        }

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_full_metrics_table_formats_stats_and_fallbacks(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { FullMetricsTable } from './components/full-metrics-table.js';
            function flatten(node, acc = { text: [], templates: [] }) {
              if (node == null || typeof node === 'boolean' || typeof node === 'function') return acc;
              if (Array.isArray(node)) { for (const item of node) flatten(item, acc); return acc; }
              if (typeof node === 'string' || typeof node === 'number') { acc.text.push(String(node)); return acc; }
              if (node.strings) {
                acc.templates.push(Array.from(node.strings).join(''));
                for (const value of node.values ?? []) flatten(value, acc);
              }
              return acc;
            }
            const rendered = FullMetricsTable({
              title: 'Full Benchmark Metrics',
              rows: [
                { key: 'latency', metric: 'Request Latency', unit: 'ms', avg: 510.25, min: 120, max: 9000, p99: 812, p90: Number.NaN, p50: null },
              ],
            });
            const flat = flatten(rendered);
            const text = flat.text.join('|');
            console.log(JSON.stringify({
              text,
              templates: flat.templates.join('|'),
              fallbackCount: (text.match(/---/g) || []).length,
            }));
            """,
        )

        assert "Full Benchmark Metrics" in result["text"]
        assert "Request Latency" in result["text"]
        assert "ms" in result["text"]
        assert "510.25" in result["text"]
        assert "9,000" in result["text"]
        assert result["fallbackCount"] >= 2
        assert "full-metrics-table" in result["templates"]

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_full_metrics_adapters_normalize_three_sources(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import {
              rowsFromMetrics,
              rowsFromServerMetrics,
            } from './components/full-metrics-table.js';
            const benchmarkRows = rowsFromMetrics([
              { tag: 'request_latency', header: 'Request Latency', unit: 'ms', avg: 1, min: 2, max: 3, p99: 4, p90: 5, p50: 6 },
              { tag: 'bad_metric', header: null, unit: 'x', avg: Number.POSITIVE_INFINITY },
              null,
            ]);
            const gpuRows = rowsFromMetrics([
              {
                tag: 'gpu_utilization_dcgm_gpu_0',
                header: 'GPU Utilization | http://srv:9400 | GPU 0 | NVIDIA H100',
                unit: '%',
                current: 96.0,
              },
            ]);
            const serverRows = rowsFromServerMetrics([
              { endpoint: 'http://srv:8000', metrics: [
                { name: 'kv_cache_utilization', unit: 'ratio', avg: 0.5, min: 0.1, max: 0.9, p99: 0.8, p90: 0.7, p50: 0.4 },
              ]},
            ]);
            console.log(JSON.stringify({ benchmarkRows, gpuRows, serverRows }));
            """,
        )

        assert result["benchmarkRows"][0] == {
            "key": "request_latency",
            "metric": "Request Latency",
            "unit": "ms",
            "avg": 1,
            "min": 2,
            "max": 3,
            "p99": 4,
            "p90": 5,
            "p50": 6,
        }
        assert result["benchmarkRows"][1]["metric"] == "bad_metric"
        assert result["benchmarkRows"][1]["avg"] is None
        assert result["gpuRows"][0]["key"] == "gpu_utilization_dcgm_gpu_0"
        assert (
            result["gpuRows"][0]["metric"]
            == "GPU Utilization | http://srv:9400 | GPU 0 | NVIDIA H100"
        )
        assert result["gpuRows"][0]["unit"] == "%"
        assert result["gpuRows"][0]["avg"] == 96.0
        assert result["serverRows"][0]["key"] == "http://srv:8000::kv_cache_utilization"
        assert (
            result["serverRows"][0]["metric"]
            == "http://srv:8000 · kv_cache_utilization"
        )

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_records_dispatch_reads_real_wire_keys(self, tmp_path: Path) -> None:
        """RecordsProcessingStatsMessage / AllRecordsReceivedMessage nest their
        counters under ``processing_stats`` / ``final_processing_stats`` on the
        wire (never ``stats``); the dispatcher must read those keys."""
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { records } from './lib/state.js';
            handleWsMessage({
              type: 'processing_stats',
              processing_stats: {
                success_records: 97,
                error_records: 1,
                final_requests_completed: 98,
                start_ns: 1000,
              },
            });
            const mid = { ...records.value };
            handleWsMessage({
              type: 'all_records_received',
              final_processing_stats: {
                success_records: 99,
                error_records: 1,
                final_requests_completed: 100,
                records_end_ns: 2000,
              },
            });
            console.log(JSON.stringify({ mid, final: records.value }));
            """,
        )

        assert result["mid"]["successRecords"] == 97
        assert result["mid"]["errorRecords"] == 1
        assert result["mid"]["finalRequestsCompleted"] == 98
        assert result["mid"]["active"] is True
        assert result["final"]["successRecords"] == 99
        assert result["final"]["errorRecords"] == 1
        assert result["final"]["finalRequestsCompleted"] == 100
        assert result["final"]["endNs"] == 2000
        assert result["final"]["complete"] is True

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_worker_group_in_flight_derived_from_wire_counters(
        self, tmp_path: Path
    ) -> None:
        """WorkerTaskStats.in_progress is a non-serialized @property; the
        dispatcher derives in-flight as total - completed - failed for both
        the group row and each per-worker child."""
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { workerGroups } from './lib/state.js';
            handleWsMessage({
              type: 'worker_group_stats',
              group_id: 'wg-primary',
              status: 'healthy',
              task_stats: { total: 101, completed: 97, failed: 1 },
              worker_statuses: { 'w-a': 'healthy', 'w-b': 'healthy' },
              worker_task_stats: {
                'w-a': { total: 52, completed: 51, failed: 0 },
                'w-b': { total: 49, completed: 46, failed: 1 },
              },
            });
            const g = workerGroups.value['wg-primary'];
            console.log(JSON.stringify({
              group: g.inFlight,
              a: g.workers['w-a'].inFlight,
              b: g.workers['w-b'].inFlight,
            }));
            """,
        )

        assert result == {"group": 3, "a": 1, "b": 2}

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_unknown_websocket_messages_log_once(self, tmp_path: Path) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { logs } from './lib/state.js';
            logs.value = [];
            handleWsMessage(null);
            handleWsMessage('bad');
            handleWsMessage({ type: 'future_payload' });
            handleWsMessage({ type: 'future_payload' });
            console.log(JSON.stringify(logs.value));
            """,
        )

        matching = [entry for entry in result if "future_payload" in entry["message"]]
        assert len(matching) == 1
        assert matching[0]["category"] == "ws"

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_gpu_telemetry_accepts_headers_without_model_suffix(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { GpuTelemetryCard } from './components/gpu-telemetry.js';
            import { telemetryMetrics } from './lib/state.js';
            telemetryMetrics.value = [
              {
                tag: 'gpu_utilization_dcgm_http___node1_9401_metrics_gpu0_uuid',
                header: 'GPU Utilization | node1:9401 | GPU 0',
                unit: '%',
                current: 88.5,
                avg: 87.0,
              },
            ];
            const rendered = JSON.stringify(GpuTelemetryCard());
            console.log(JSON.stringify({
              rendered,
              hasGpuCard: rendered.includes('node1:9401 | GPU 0'),
              hasUtilization: rendered.includes('Utilization'),
              hasValue: rendered.includes('88.5'),
            }));
            """,
        )

        assert result["hasGpuCard"] is True, result["rendered"]
        assert result["hasUtilization"] is True, result["rendered"]
        assert result["hasValue"] is True, result["rendered"]

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_worker_table_tolerates_malformed_missing_worker_state(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { workerGroups } from './lib/state.js';
            import { WorkerTable } from './components/worker-table.js';
            function flatten(node, acc = { text: [], templates: [] }) {
              if (node == null || typeof node === 'boolean' || typeof node === 'function') return acc;
              if (Array.isArray(node)) { for (const item of node) flatten(item, acc); return acc; }
              if (typeof node === 'string' || typeof node === 'number') { acc.text.push(String(node)); return acc; }
              if (node.strings) {
                acc.templates.push(Array.from(node.strings).join(''));
                for (const value of node.values ?? []) flatten(value, acc);
              }
              return acc;
            }
            workerGroups.value = {
              'worker-group-b': null,
              'worker-group-a': {
                status: null,
                workers: {
                  'worker-child-b': null,
                  'worker-child-a': { completed: 7 },
                },
              },
              'worker-group-c': 'bad-state',
            };
            const flat = flatten(WorkerTable());
            console.log(JSON.stringify({
              text: flat.text.join('|'),
              rowCount: flat.templates.filter(t => t.includes('<tr')).length,
              hasTitle: flat.templates.join('|').includes('Worker Groups'),
              hasUndefined: flat.text.includes('undefined'),
              hasNull: flat.text.includes('null'),
            }));
            """,
        )

        assert result["rowCount"] >= 3
        assert result["hasTitle"] is True
        assert result["hasUndefined"] is False
        assert result["hasNull"] is False

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_worker_table_sorts_many_workers_and_escapes_ids(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { workerGroups } from './lib/state.js';
            import { WorkerTable } from './components/worker-table.js';
            function flatten(node, acc = { text: [], templates: [] }) {
              if (node == null || typeof node === 'boolean' || typeof node === 'function') return acc;
              if (Array.isArray(node)) { for (const item of node) flatten(item, acc); return acc; }
              if (typeof node === 'string' || typeof node === 'number') { acc.text.push(String(node)); return acc; }
              if (node.strings) {
                acc.templates.push(Array.from(node.strings).join(''));
                for (const value of node.values ?? []) flatten(value, acc);
              }
              return acc;
            }
            const dangerousId = 'worker-zz-<img src=x onerror=alert(1)>-&';
            const workers = {};
            for (let i = 20; i >= 0; i -= 1) {
              const suffix = String(i).padStart(3, '0');
              workers[`worker-child-${suffix}`] = { status: i % 2 ? 'healthy' : 'high_load', completed: i };
            }
            workers[dangerousId] = { status: 'healthy' };
            workerGroups.value = {
              'worker-group-main': { status: 'healthy', declaredWorkers: 22, readyWorkers: 22, workers },
            };
            const flat = flatten(WorkerTable());
            const text = flat.text.join('|');
            const templates = flat.templates.join('|');
            console.log(JSON.stringify({
              rowCount: flat.templates.filter(t => t.includes('<tr')).length,
              firstIndex: text.indexOf('child-000'),
              middleIndex: text.indexOf('child-010'),
              lastIndex: text.indexOf('child-020'),
              dangerousInText: text.includes('<img src=x onerror=alert(1)>'),
              dangerousInTemplates: templates.includes('<img src=x onerror=alert(1)>'),
            }));
            """,
        )

        assert result["rowCount"] == 24
        assert 0 <= result["firstIndex"] < result["middleIndex"] < result["lastIndex"]
        assert result["dangerousInText"] is True
        assert result["dangerousInTemplates"] is False

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_websocket_error_and_teardown_do_not_lose_status_or_reconnect(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { connection, logs } from './lib/state.js';
            import { connectWebSocket, teardownWebSocket } from './lib/ws.js';
            const timers = [];
            globalThis.window = { location: { protocol: 'http:', host: 'example.test' } };
            globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length - 1; };
            globalThis.clearTimeout = () => {};
            class FakeWebSocket {
              static instances = [];
              constructor(url) { this.url = url; FakeWebSocket.instances.push(this); }
              send() {}
              close() { this.closed = true; this.onclose?.({ code: 1000 }); }
            }
            globalThis.WebSocket = FakeWebSocket;

            connectWebSocket();
            const first = FakeWebSocket.instances[0];
            first.onerror(new Error('boom'));
            first.onclose({ code: 1006 });
            const statusAfterErrorClose = connection.value;
            const reconnectsAfterErrorClose = timers.length;
            const errorLogged = logs.value.some(entry => entry.severity === 'error' && entry.message === 'WebSocket error');
            timers[0].fn();
            teardownWebSocket();
            console.log(JSON.stringify({
              statusAfterErrorClose,
              reconnectsAfterErrorClose,
              errorLogged,
              secondClosed: FakeWebSocket.instances[1].closed,
              reconnectsAfterTeardown: timers.length,
            }));
            """,
        )

        assert result == {
            "statusAfterErrorClose": "error",
            "reconnectsAfterErrorClose": 1,
            "errorLogged": True,
            "secondClosed": True,
            "reconnectsAfterTeardown": 1,
        }

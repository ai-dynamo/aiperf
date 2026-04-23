// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AIPerf Dashboard v2 entrypoint.
 *
 * Preact + htm + @preact/signals, same stack as src/aiperf/operator/ui.
 * Single-page (no router): the API dashboard is one live view of the
 * current benchmark run.
 */

import { html, render } from 'htm/preact';
import { useEffect } from 'preact/hooks';

import { connection, config, log } from './lib/state.js';
import { api } from './lib/api.js';
import { connectWebSocket, currentEpoch, teardownWebSocket } from './lib/ws.js';
import { bootstrapProgress } from './lib/ws-dispatch.js';

import { ConfigBar } from './components/config-bar.js';
import { StatusBar } from './components/status-bar.js';
import { HeroStrip } from './components/hero-strip.js';
import { PhaseCards } from './components/phase-cards.js';
import { RealtimeMetricsCard } from './components/realtime-metrics.js';
import { ThroughputLatencyChart } from './components/throughput-latency-chart.js';
import { GpuTelemetryCard } from './components/gpu-telemetry.js';
import { WorkerTable } from './components/worker-table.js';
import { LogPane } from './components/log-pane.js';
import { ServerMetrics } from './components/server-metrics.js';

function Dashboard() {
  useEffect(() => {
    const ac = new AbortController();

    connectWebSocket({
      onOpen: async (epoch) => {
        // Bootstrap three HTTP endpoints in parallel; ignore responses from
        // stale connections so a reconnect doesn't clobber fresh state.
        const safe = async (fn, apply) => {
          try {
            const data = await fn(ac.signal);
            if (epoch === currentEpoch()) apply(data);
          } catch (e) {
            if (e?.name === 'AbortError') return;
            log(`fetch failed: ${e.message}`);
          }
        };

        await Promise.all([
          safe((s) => api.getConfig(s),       (d) => { if (d) config.value = d; }),
          safe((s) => api.getProgress(s),     (d) => { if (d) bootstrapProgress(d); }),
          safe((s) => api.getServerMetrics(s), () => { /* WS drives this; HTTP is a warm start */ }),
        ]);
      },
    });

    const onBeforeUnload = () => teardownWebSocket();
    window.addEventListener('beforeunload', onBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', onBeforeUnload);
      ac.abort();
      teardownWebSocket();
    };
  }, []);

  // Force re-render on every signal change via .value reads:
  connection.value; config.value;

  return html`
    <div class="app">
      <div class="topbar">
        <div class="logo">
          <span class="logo-badge">AI</span>
          <span>AIPerf Dashboard</span>
        </div>
        <div style="font-size: 11px; color: var(--dim); font-family: var(--font-mono)">
          v2
        </div>
      </div>
      <div class="content">
        <${StatusBar} />
        <${HeroStrip} />
        <${ConfigBar} />
        <${RealtimeMetricsCard} />
        <${ThroughputLatencyChart} />
        <${PhaseCards} />
        <${GpuTelemetryCard} />
        <${WorkerTable} />
        <${ServerMetrics} />
        <${LogPane} />
      </div>
    </div>
  `;
}

render(html`<${Dashboard} />`, document.getElementById('app'));

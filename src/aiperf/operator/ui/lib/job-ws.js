// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-job WebSocket subscription manager.
 *
 * Connects to ``/api/v1/jobs/{ns}/{name}/ws`` (proxied by the operator into
 * the controller pod's ``:API_SERVICE/ws``), subscribes to the realtime
 * message types, and accumulates two views the KPI grid consumes:
 *
 *   - ``summary``: ``{ [tag]: {avg, p99, current, ...} }`` — last-seen stats
 *     per metric, mirroring the REST snapshot shape so callers can swap in
 *     live data without reshaping.
 *   - ``timeseries``: ``{ [tag]: [{t, values}, ...] }`` — bounded rolling
 *     window for sparklines (~2 minutes / 120 points).
 *
 * Browser owns the subscribe protocol (controller speaks
 * ``{"type":"subscribe","message_types":[...]}``). The proxy is transparent.
 */

import {
  normalizeServerMetrics,
  aggregateSparklineSnapshot,
} from '../components/server-metrics/helpers.js';

const SUBSCRIBE_TYPES = ['realtime_metrics', 'realtime_server_metrics'];
const RECONNECT_DELAY_MS = 2000;
const MAX_POINTS = 120;
const MAX_AGE_MS = 5 * 60_000;

/**
 * Open a per-job WebSocket subscription. Returns a handle whose ``close()``
 * tears down the WS and stops auto-reconnect.
 *
 * @param {string} ns - AIPerfJob namespace
 * @param {string} name - AIPerfJob name
 * @param {(state: {summary: object, timeseries: object, connected: boolean}) => void} onUpdate
 *   Invoked on every state change. Receives a fresh shallow-copied snapshot
 *   so consumers can store-and-compare without aliasing.
 * @returns {{ close: () => void }}
 */
export function openJobWs(ns, name, onUpdate) {
  let ws = null;
  let reconnectTimer = null;
  let closed = false;

  const summary = {};
  const timeseries = {};
  let serverSummary = null;
  const serverTimeseries = {};

  function publish(connected) {
    onUpdate({
      summary: { ...summary },
      timeseries: { ...timeseries },
      serverSummary: serverSummary ? { ...serverSummary } : null,
      serverTimeseries: { ...serverTimeseries },
      connected,
    });
  }

  function pushSample(tag, t, statsObj) {
    const series = timeseries[tag] ?? [];
    const cutoff = t - MAX_AGE_MS;
    const next = series.filter(s => s.t >= cutoff);
    next.push({ t, values: statsObj });
    if (next.length > MAX_POINTS) next.splice(0, next.length - MAX_POINTS);
    timeseries[tag] = next;
  }

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

  function applyRealtimeMetrics(metrics) {
    if (!Array.isArray(metrics)) return;
    const t = Date.now();
    for (const m of metrics) {
      const tag = m?.tag;
      if (!tag) continue;
      const stats = {};
      for (const [k, v] of Object.entries(m)) {
        if (typeof v === 'number' && isFinite(v)) stats[k] = v;
      }
      summary[tag] = { ...summary[tag], ...stats, unit: m.unit ?? summary[tag]?.unit };
      pushSample(tag, t, stats);
    }
  }

  function handleMessage(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    const type = msg?.type ?? msg?.message_type;
    if (type === 'realtime_metrics' && Array.isArray(msg.metrics)) {
      applyRealtimeMetrics(msg.metrics);
      publish(true);
    } else if (type === 'realtime_server_metrics') {
      applyRealtimeServerMetrics(msg.endpoint_summaries ? msg : msg.payload ?? null);
      publish(true);
    }
  }

  function connect() {
    if (closed) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/api/v1/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}/ws`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      try {
        ws.send(JSON.stringify({ type: 'subscribe', message_types: SUBSCRIBE_TYPES }));
      } catch { /* socket may have closed between open and send */ }
      publish(true);
    };

    ws.onmessage = (event) => handleMessage(event.data);

    ws.onclose = () => {
      ws = null;
      if (closed) return;
      publish(false);
      reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => { /* onclose follows; reconnect path lives there */ };
  }

  connect();

  return {
    close() {
      closed = true;
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        try { ws.close(1000, 'page leaving'); } catch { /* best effort */ }
        ws = null;
      }
    },
  };
}

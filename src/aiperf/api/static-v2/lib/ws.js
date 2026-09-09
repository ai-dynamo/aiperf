// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * WebSocket lifecycle manager for the dashboard.
 *
 * Owns: connect / reconnect / close, subscribe on open, dispatch raw messages
 * to handlers in ./ws-dispatch.js, and surface state transitions via
 * ./state.js signals.
 */

import { connection, log } from './state.js';
import { handleWsMessage } from './ws-dispatch.js';

const SUBSCRIBE_TYPES = [
  'realtime_metrics',
  'realtime_telemetry_metrics',
  'realtime_server_metrics',
  'credit_phase_progress',
  'credit_phase_start',
  'credit_phase_sending_complete',
  'credit_phase_complete',
  'processing_stats',
  'all_records_received',
  'worker_group_stats',
];

const RECONNECT_DELAY_MS = 2000;

let currentWs = null;
let connectionEpoch = 0;
let reconnectTimer = null;
let reconnectEnabled = true;

/** Bump before each connect; stale async fetches check this to avoid clobbering fresh state. */
export function currentEpoch() {
  return connectionEpoch;
}

export function connectWebSocket({ onOpen } = {}) {
  reconnectEnabled = true;
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  const epoch = ++connectionEpoch;

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${window.location.host}/ws`;

  connection.value = 'connecting';
  log('Connecting...');

  const ws = new WebSocket(url);
  currentWs = ws;

  ws.onopen = () => {
    connection.value = 'connected';
    log('Connected');
    try {
      ws.send(JSON.stringify({ type: 'subscribe', message_types: SUBSCRIBE_TYPES }));
    } catch (e) { log(`subscribe failed: ${e.message}`); }
    if (onOpen) onOpen(epoch);
  };

  ws.onclose = () => {
    if (currentWs !== ws) return;
    if (connection.value !== 'error') connection.value = 'disconnected';
    currentWs = null;
    if (!reconnectEnabled) {
      log('Disconnected');
      return;
    }
    log('Disconnected, reconnecting...');
    reconnectTimer = setTimeout(() => connectWebSocket({ onOpen }), RECONNECT_DELAY_MS);
  };

  ws.onerror = () => {
    connection.value = 'error';
    log({ severity: 'error', category: 'connection', message: 'WebSocket error' });
  };

  ws.onmessage = (event) => {
    try {
      handleWsMessage(JSON.parse(event.data));
    } catch (e) {
      log(`parse error: ${e.message}`);
    }
  };
}

/** Tear down the current WS (used by window beforeunload). */
export function teardownWebSocket() {
  reconnectEnabled = false;
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (currentWs) {
    try { currentWs.close(); } catch (_) { /* best effort */ }
    currentWs = null;
  }
}

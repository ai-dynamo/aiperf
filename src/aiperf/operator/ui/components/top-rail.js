// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TOP RAIL — the persistent command strip at the top of the Flight Deck.
 *
 * Carries (left → right):
 *   ┌ logo + station callsign  (AIPERF // FLIGHT DECK)
 *   ├ view selector            (OVERVIEW / FLEET / ANALYSIS / LOG)
 *   ├ contextual path          (when a run is pinned to main viewport)
 *   ├ fleet telemetry strip    (live RUNS / GPUs / FLEET R/S / WORST P99)
 *   ├ UTC clock + NET LED
 *   ├ pane toggles             (left · right · log) with kbd hints
 *   └ search button            (Ctrl+K)
 *
 * Nothing here initiates API calls — the rail reads from `jobs` / `clusterInfo`
 * signals that ``app.js`` polls globally.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { jobs, clusterInfo } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { fmtInt, fmtNumber } from '../lib/format.js';

const VIEWS = [
  { kind: 'overview', path: '/overview', label: 'OVERVIEW', icon: 'ph-crosshair' },
  { kind: 'fleet',    path: '/fleet',    label: 'FLEET',    icon: 'ph-list-numbers' },
  { kind: 'analysis', path: '/analysis', label: 'ANALYSIS', icon: 'ph-scales' },
  { kind: 'log',      path: '/log',      label: 'LOG',      icon: 'ph-clock-counter-clockwise' },
];

const pad = n => String(n).padStart(2, '0');

function useUtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())}`;
}

function useNetStatus() {
  const [status, setStatus] = useState('ok');
  useEffect(() => {
    let failures = 0;
    let aborted = false;
    async function tick() {
      try {
        const r = await fetch('/healthz', { method: 'GET', cache: 'no-store' });
        if (aborted) return;
        if (r.ok) { failures = 0; setStatus('ok'); }
        else { failures += 1; setStatus(failures >= 2 ? 'err' : 'warn'); }
      } catch (_e) {
        if (aborted) return;
        failures += 1;
        setStatus(failures >= 2 ? 'err' : 'warn');
      }
    }
    tick();
    const id = setInterval(tick, 6000);
    return () => { aborted = true; clearInterval(id); };
  }, []);
  return status;
}

/** Aggregate live run stats for the fleet telemetry strip. */
function useFleetSnapshot() {
  const js = jobs.value ?? [];
  const ci = clusterInfo.value;
  const running = js.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'running' || p === 'initializing' || p === 'pending';
  });
  const failed = js.filter(j => {
    const p = (j.phase ?? '').toLowerCase();
    return p === 'failed' || p === 'error';
  });
  let fleetRps = 0;
  let rpsKnown = false;
  let worstP99 = null;
  for (const j of running) {
    if (j.throughputRps != null) { fleetRps += Number(j.throughputRps); rpsKnown = true; }
    if (j.latencyP99Ms != null) worstP99 = worstP99 == null ? j.latencyP99Ms : Math.max(worstP99, j.latencyP99Ms);
  }
  const gpus = ci?.gpus ?? ci?.gpuCount ?? ci?.gpu_count ?? null;
  const gpuCap = ci?.gpuCapacity ?? ci?.gpu_capacity ?? null;
  return { running, failed, fleetRps, rpsKnown, worstP99, gpus, gpuCap };
}

export function TopRail({
  viewKind, runParams,
  onSearchClick, onToggleLog, onToggleLeft, onToggleRight,
  logOpen, leftOpen, rightOpen,
}) {
  const clock = useUtcClock();
  const net = useNetStatus();
  const snap = useFleetSnapshot();

  const netLabel = net === 'ok' ? 'UP' : net === 'warn' ? 'RETRY' : 'DOWN';

  return html`
    <header class="rail-top" data-testid="top-nav">
      <div class="rail-top-left">
        <button
          class="pane-toggle pane-toggle--left"
          aria-pressed=${leftOpen}
          onclick=${onToggleLeft}
          title="Toggle fleet rail (Ctrl+B)"
          data-testid="toggle-left"
        ><i class="ph ph-sidebar-simple"></i></button>

        <div class="callsign" data-testid="callsign">
          <span class="callsign-light" aria-hidden="true"></span>
          <span class="callsign-head">AIPERF</span>
          <span class="callsign-sep">//</span>
          <span class="callsign-tail">FLIGHT DECK</span>
        </div>

        <nav class="view-switch" aria-label="View">
          ${VIEWS.map(v => html`
            <button
              key=${v.kind}
              class=${'view-switch-tab' + (v.kind === viewKind ? ' active' : '')}
              onclick=${() => navigate(v.path)}
              aria-current=${v.kind === viewKind ? 'page' : undefined}
              data-testid=${'nav-link-' + (v.kind === 'overview' ? 'dashboard' : v.kind === 'analysis' ? 'leaderboard' : v.kind === 'log' ? 'history' : v.kind === 'fleet' ? 'jobs' : v.kind)}
            >
              <i class=${'ph ' + v.icon}></i>
              <span>${v.label}</span>
            </button>
          `)}
        </nav>

        ${viewKind === 'run' && runParams && html`
          <div class="rail-path" data-testid="rail-path">
            <span class="rail-path-caret">▸</span>
            <span class="rail-path-ns">${runParams.ns}</span>
            <span class="rail-path-sep">/</span>
            <span class="rail-path-name">${runParams.name}</span>
          </div>
        `}
      </div>

      <div class="rail-top-right">
        <div class="tele-strip" role="group" aria-label="Fleet telemetry">
          <div class="tele">
            <span class="tele-label">LIVE</span>
            <span class=${'tele-val' + (snap.running.length > 0 ? ' is-hot' : '')}>
              ${snap.running.length}
            </span>
          </div>
          <div class="tele">
            <span class="tele-label">R/S</span>
            <span class=${'tele-val' + (snap.rpsKnown ? ' is-amber' : '')}>
              ${snap.rpsKnown ? fmtNumber(snap.fleetRps, 0) : '—'}
            </span>
          </div>
          <div class="tele">
            <span class="tele-label">P99</span>
            <span class=${'tele-val' + (snap.worstP99 != null && snap.worstP99 > 500 ? ' is-red' : '')}>
              ${snap.worstP99 != null ? fmtInt(snap.worstP99) : '—'}
            </span>
          </div>
          <div class="tele">
            <span class="tele-label">GPU</span>
            <span class="tele-val">
              ${snap.gpus != null ? fmtInt(snap.gpus) : '—'}${snap.gpuCap != null ? html`<small>/${fmtInt(snap.gpuCap)}</small>` : null}
            </span>
          </div>
          ${snap.failed.length > 0 && html`
            <div class="tele tele--bad">
              <span class="tele-label">FAIL</span>
              <span class="tele-val is-red">${snap.failed.length}</span>
            </div>
          `}
        </div>

        <div class=${'rail-net rail-net--' + net} title=${'NET ' + netLabel} data-testid="net-status">
          <span class="rail-net-dot"></span>
          NET · ${netLabel}
        </div>

        <div class="rail-clock" data-testid="topbar-clock">
          <span class="rail-clock-label">UTC</span>
          <span class="rail-clock-val">${clock}</span>
        </div>

        <button
          class="rail-search"
          onclick=${onSearchClick}
          title="Search (Ctrl+K)"
          data-testid="nav-search"
        >
          <i class="ph ph-magnifying-glass"></i>
          <span>Search</span>
          <kbd>⌘K</kbd>
        </button>

        <div class="pane-toggles">
          <button
            class=${'pane-toggle' + (logOpen ? ' is-on' : '')}
            aria-pressed=${logOpen}
            onclick=${onToggleLog}
            title="Toggle log strip (Ctrl + grave)"
            data-testid="toggle-log"
          ><i class="ph ph-terminal-window"></i></button>
          <button
            class=${'pane-toggle' + (rightOpen ? ' is-on' : '')}
            aria-pressed=${rightOpen}
            onclick=${onToggleRight}
            title="Toggle inspector (Ctrl+I)"
            data-testid="toggle-right"
          ><i class="ph ph-columns"></i></button>
        </div>
      </div>
    </header>
  `;
}

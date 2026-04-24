// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TOP RAIL — the only persistent element of the Workbench shell.
 *
 * Left:  station callsign, breadcrumb trail for the current view.
 * Right: LAUNCH primary CTA (⌘N), ARCHIVE + LOG secondary buttons, ⌘K
 *        search, UTC clock, NET heartbeat LED.
 *
 * Everything else — run listing, per-run inspector, event log — either
 * lives inside the viewport itself or is reachable via ⌘K, so the rail stays
 * thin and out of the way.
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { navigate } from '../lib/router.js';

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

/** Derive breadcrumb segments from the current view + params. */
function breadcrumbFor(viewKind, runParams) {
  if (viewKind === 'run' && runParams) {
    return [
      { label: 'Run', path: null },
      { label: runParams.ns, path: '/archive' },
      { label: runParams.name, path: null, emphasise: true },
    ];
  }
  if (viewKind === 'compare' && runParams) {
    return [
      { label: 'Compare', path: null },
      { label: runParams.ns, path: '/archive' },
      { label: runParams.name, path: `/run/${runParams.ns}/${runParams.name}`, emphasise: true },
    ];
  }
  if (viewKind === 'launch')   return [{ label: 'Launch', path: null, emphasise: true }];
  if (viewKind === 'archive')  return [{ label: 'Archive', path: null, emphasise: true }];
  if (viewKind === 'analysis') return [{ label: 'Compare', path: null, emphasise: true }];
  if (viewKind === 'log')      return [{ label: 'Log',    path: null, emphasise: true }];
  return [];
}

export function TopRail({ viewKind, runParams, onSearchClick }) {
  const clock = useUtcClock();
  const net = useNetStatus();
  const crumbs = breadcrumbFor(viewKind, runParams);

  const netLabel = net === 'ok' ? 'UP' : net === 'warn' ? 'RETRY' : 'DOWN';

  return html`
    <header class="rail" data-testid="top-nav">
      <div class="rail-left">
        <button
          class="rail-logo"
          onclick=${() => navigate('/')}
          title="Home"
          data-testid="callsign"
        >
          <span class="rail-logo-light"></span>
          <span class="rail-logo-head">AIPERF</span>
          <span class="rail-logo-sep">//</span>
          <span class="rail-logo-tail">WORKBENCH</span>
        </button>

        ${crumbs.length > 0 && html`
          <nav class="rail-crumbs" aria-label="Breadcrumb" data-testid="breadcrumb">
            ${crumbs.map((c, i) => html`
              <span key=${i} class="rail-crumb-sep">▸</span>
              ${c.path
                ? html`<a class="rail-crumb" href=${'#' + c.path} onclick=${(e) => { e.preventDefault(); navigate(c.path); }}>${c.label}</a>`
                : html`<span class=${'rail-crumb' + (c.emphasise ? ' rail-crumb--strong' : '')}>${c.label}</span>`}
            `)}
          </nav>
        `}
      </div>

      <div class="rail-right">
        <button
          class=${'rail-launch' + (viewKind === 'launch' ? ' is-active' : '')}
          onclick=${() => navigate('/launch')}
          data-testid="rail-launch"
          title="Launch a new run (⌘N)"
        >
          <i class="ph ph-plus"></i>
          <span>Launch</span>
          <kbd>⌘N</kbd>
        </button>

        <button
          class=${'rail-btn' + (viewKind === 'archive' ? ' is-active' : '')}
          onclick=${() => navigate('/archive')}
          data-testid="rail-archive"
          title="Browse past runs"
        >
          <i class="ph ph-archive"></i>
          <span>Archive</span>
        </button>

        <button
          class=${'rail-btn' + (viewKind === 'analysis' ? ' is-active' : '')}
          onclick=${() => navigate('/compare')}
          data-testid="rail-compare"
          title="Compare runs"
        >
          <i class="ph ph-scales"></i>
          <span>Compare</span>
        </button>

        <div class="rail-sep" aria-hidden="true"></div>

        <button
          class="rail-search"
          onclick=${onSearchClick}
          title="Search runs (⌘K)"
          data-testid="nav-search"
        >
          <i class="ph ph-magnifying-glass"></i>
          <kbd>⌘K</kbd>
        </button>

        <div class=${'rail-net rail-net--' + net} title=${'NET ' + netLabel} data-testid="net-status">
          <span class="rail-net-dot"></span>
          NET · ${netLabel}
        </div>

        <div class="rail-clock" data-testid="topbar-clock" title="UTC">
          ${clock}
        </div>
      </div>
    </header>
  `;
}

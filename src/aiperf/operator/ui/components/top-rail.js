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
import { route, matchRoute, navigate } from '../lib/router.js';
import { launchDivergence } from '../lib/state.js';
import { NamespaceSwitcher } from './namespace-switcher.js';

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

/** Derive the current namespace from the route, or null on cross-namespace tier. */
function deriveNamespace(currentRoute) {
  const patterns = [
    '/ns/:ns',
    '/ns/:ns/launch',
    '/ns/:ns/archive',
    '/ns/:ns/run/:name',
    '/ns/:ns/run/:name/runs/:epoch',
  ];
  for (const p of patterns) {
    const m = matchRoute(p, currentRoute);
    if (m) return m.ns;
  }
  return null;
}

/** Derive trailing breadcrumb segments (after the namespace pill, if any). */
function trailingCrumbs(viewKind, runParams, ns) {
  if (viewKind === 'run' && runParams) {
    return [{ label: runParams.name, path: null, emphasise: true }];
  }
  if (viewKind === 'compare' && runParams) {
    return [{ label: runParams.name, path: `/run/${runParams.ns}/${runParams.name}`, emphasise: true }];
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
  const currentRoute = route.value;
  const ns = deriveNamespace(currentRoute);
  const [switcherOpen, setSwitcherOpen] = useState(false);

  // Close the switcher whenever the route changes (covers item-click +
  // any other in-app nav).
  useEffect(() => { setSwitcherOpen(false); }, [currentRoute]);

  const crumbs = trailingCrumbs(viewKind, runParams, ns);
  const netLabel = net === 'ok' ? 'UP' : net === 'warn' ? 'RETRY' : 'DOWN';
  const launchTarget = ns ? `/ns/${encodeURIComponent(ns)}/launch` : null;
  const archiveTarget = ns ? `/ns/${encodeURIComponent(ns)}/archive` : null;
  // Reading the signal in render subscribes the component, so the pill
  // re-renders automatically when the launch view writes/clears divergence.
  const pillClass = 'ns-switcher-pill' + (launchDivergence.value ? ' ns-switcher-pill--bad' : '');

  return html`
    <header class="topbar" data-testid="top-nav">
      <div class="topbar-left">
        <a
          class="topbar-logo"
          href="#/"
          onclick=${(e) => { e.preventDefault(); navigate('/'); }}
          data-testid="callsign"
        >
          <span class="topbar-logo-badge">AI</span>
          <span>AIPerf Operator</span>
        </a>
        <nav class="topbar-crumbs" aria-label="Breadcrumb" data-testid="breadcrumb">
          ${ns && html`
            <span class="topbar-crumb topbar-crumb--ns">
              <button
                class=${pillClass}
                data-testid="ns-switcher-pill"
                onclick=${() => setSwitcherOpen(v => !v)}
                title="Switch namespace"
              >${ns}</button>
              ${switcherOpen && html`<${NamespaceSwitcher}
                currentNs=${ns}
                onClose=${() => setSwitcherOpen(false)}
              />`}
            </span>
          `}
          ${crumbs.map((c, i) => html`
            ${(ns || i > 0) && html`<span class="topbar-crumb-sep">/</span>`}
            <span class=${'topbar-crumb' + (i === crumbs.length - 1 ? ' topbar-crumb--current' : '')}>
              ${c.path
                ? html`<a href=${'#' + c.path} onclick=${(e) => { e.preventDefault(); navigate(c.path); }}>${c.label}</a>`
                : c.label}
            </span>
          `)}
        </nav>
      </div>

      <div class="topbar-right">
        <button
          class="btn btn--ghost"
          onclick=${() => navigate('/compare')}
          data-testid="rail-compare"
          title="Compare"
        >Compare</button>
        ${archiveTarget && html`
          <button
            class="btn btn--ghost"
            onclick=${() => navigate(archiveTarget)}
            data-testid="rail-archive"
            title="Archive"
          >Archive</button>
        `}
        <button
          class="btn btn--ghost"
          onclick=${onSearchClick}
          data-testid="nav-search"
          title="Open command palette"
        >Search <span class="kbd">⌘ K</span></button>
        ${launchTarget && html`
          <button
            class="btn btn--primary"
            onclick=${() => navigate(launchTarget)}
            data-testid="rail-launch"
            title="Launch new run (⌘N)"
          >+ Launch</button>
        `}
        <div class=${'topbar-net topbar-net--' + net} title=${'Network ' + netLabel} data-testid="net-status">
          <span class="status-dot"></span>
          <span>${netLabel}</span>
        </div>
        <div class="topbar-clock" data-testid="topbar-clock" title="UTC">${clock} UTC</div>
      </div>
    </header>
  `;
}

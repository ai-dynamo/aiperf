// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AIPERF // FLIGHT DECK — new three-pane operator shell.
 *
 * Layout (persistent across every view):
 *
 *   ┌──────────────────────────────────────────────────┐ 48 px
 *   │                  TOP RAIL                         │
 *   ├──────────┬──────────────────────┬──────────────┤
 *   │          │                       │                │
 *   │  LEFT    │        MAIN           │   RIGHT        │
 *   │  RAIL    │        VIEWPORT       │   INSPECTOR    │
 *   │  (260)   │        (flex)         │   (340)        │
 *   ├──────────┴──────────────────────┴──────────────┤
 *   │                  LOG STRIP                        │ 0 / 240 px
 *   └──────────────────────────────────────────────────┘
 *
 * Views are swapped inside the viewport without unmounting the rails, so the
 * user stays oriented (no page flash, no state loss). Routes become viewport
 * selectors — `/run/:ns/:name` swaps in a RUN view, `/fleet` swaps in FLEET,
 * etc. Default route is OVERVIEW.
 */

import { html, render } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { route, matchRoute, navigate } from './lib/router.js';
import { api, poll } from './lib/api.js';
import { jobs, clusterInfo, globalError } from './lib/state.js';
import { TopRail } from './components/top-rail.js';
import { LeftRail } from './components/left-rail.js';
import { RightInspector } from './components/right-inspector.js';
import { LogStrip } from './components/log-strip.js';
import { CommandPalette } from './components/command-palette.js';
import { Overview } from './views/overview.js';
import { Run } from './views/run.js';
import { Fleet } from './views/fleet.js';
import { Analysis } from './views/analysis.js';
import { Log } from './views/log.js';

function resolveView(currentRoute) {
  const runMatch = matchRoute('/run/:ns/:name', currentRoute)
    ?? matchRoute('/jobs/:ns/:name', currentRoute);        // legacy
  if (runMatch) return { kind: 'run', params: runMatch };
  if (currentRoute === '/fleet' || currentRoute === '/jobs')         return { kind: 'fleet' };
  if (currentRoute === '/analysis'
    || currentRoute === '/leaderboard'
    || currentRoute === '/compare')                                  return { kind: 'analysis' };
  if (currentRoute === '/log' || currentRoute === '/history')        return { kind: 'log' };
  return { kind: 'overview' };
}

function App() {
  const [showPalette, setShowPalette] = useState(false);
  const [logOpen, setLogOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(true);
  const [leftOpen, setLeftOpen] = useState(true);

  const currentRoute = route.value;
  const resolved = resolveView(currentRoute);
  const error = globalError.value;

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      try {
        const data = await api.listJobs();
        jobs.value = data?.jobs ?? [];
      } catch (_e) { /* left rail shows empty until recovery */ }
    }, 5000, ac.signal);
    poll(async () => {
      try { clusterInfo.value = await api.getCluster(); } catch (_e) { /* ignore */ }
    }, 10000, ac.signal);
    return () => ac.abort();
  }, []);

  useEffect(() => {
    function onKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setShowPalette(v => !v);
      } else if ((e.ctrlKey || e.metaKey) && e.key === '`') {
        e.preventDefault();
        setLogOpen(v => !v);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        setLeftOpen(v => !v);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'i') {
        e.preventDefault();
        setRightOpen(v => !v);
      } else if (e.key === 'Escape' && !showPalette) {
        // Escape backs out of a selected run to overview.
        if (resolved.kind === 'run') navigate('/overview');
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showPalette, resolved.kind]);

  let mainView;
  if (resolved.kind === 'run') {
    mainView = html`<${Run} ns=${resolved.params.ns} name=${resolved.params.name} />`;
  } else if (resolved.kind === 'fleet') {
    mainView = html`<${Fleet} />`;
  } else if (resolved.kind === 'analysis') {
    mainView = html`<${Analysis} />`;
  } else if (resolved.kind === 'log') {
    mainView = html`<${Log} />`;
  } else {
    mainView = html`<${Overview} />`;
  }

  const deckCls = [
    'deck',
    logOpen   ? 'deck--log-open'         : '',
    rightOpen ? ''                       : 'deck--right-collapsed',
    leftOpen  ? ''                       : 'deck--left-collapsed',
  ].filter(Boolean).join(' ');

  return html`
    <div class=${deckCls} data-route=${resolved.kind}>
      <${TopRail}
        viewKind=${resolved.kind}
        runParams=${resolved.params}
        onSearchClick=${() => setShowPalette(true)}
        onToggleLog=${() => setLogOpen(v => !v)}
        onToggleLeft=${() => setLeftOpen(v => !v)}
        onToggleRight=${() => setRightOpen(v => !v)}
        logOpen=${logOpen}
        leftOpen=${leftOpen}
        rightOpen=${rightOpen}
      />
      <${LeftRail}
        viewKind=${resolved.kind}
        runParams=${resolved.params}
        open=${leftOpen}
        onToggle=${() => setLeftOpen(v => !v)}
      />
      <main class="deck-main" data-testid="deck-main">
        ${error && html`
          <div class="deck-error-flash" data-testid="global-error">
            <strong>FAULT</strong>
            ${error}
          </div>
        `}
        ${mainView}
      </main>
      <${RightInspector}
        viewKind=${resolved.kind}
        runParams=${resolved.params}
        open=${rightOpen}
        onToggle=${() => setRightOpen(v => !v)}
      />
      <${LogStrip}
        open=${logOpen}
        onClose=${() => setLogOpen(false)}
      />
      ${showPalette && html`<${CommandPalette} onClose=${() => setShowPalette(false)} />`}
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));

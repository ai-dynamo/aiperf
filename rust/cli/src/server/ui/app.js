// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App shell: mounts the top nav, an app-level error strip, and the hash-routed
// page. Polls ``/api/meta`` + ``/api/runs`` every 5s into shared signals so a
// live ``aiperf profile --serve`` session shows new runs appear without a
// manual reload. There is no live-streaming view in this cut — that is a later
// addition; everything here is browse-first over completed reports.

import { html, render } from 'htm/preact';
import { useEffect } from 'preact/hooks';
import { route, matchRoute } from './lib/router.js';
import { globalError, refreshRuns } from './lib/state.js';
import { TopNav } from './components/top-nav.js';
import { Runs } from './pages/runs.js';
import { RunDetail } from './pages/run-detail.js';
import { Live } from './pages/live.js';
import { Compare } from './pages/compare.js';
import { Leaderboard } from './pages/leaderboard.js';
import { Sweeps } from './pages/sweeps.js';

const POLL_MS = 5000;

function resolvePage(current) {
  const runMatch = matchRoute('/runs/:id', current);
  if (current === '/' || current === '') return html`<${Runs} />`;
  if (runMatch) return html`<${RunDetail} id=${runMatch.id} />`;
  if (current === '/live') return html`<${Live} />`;
  if (current === '/compare') return html`<${Compare} />`;
  if (current === '/leaderboard') return html`<${Leaderboard} />`;
  if (current === '/sweeps') return html`<${Sweeps} />`;
  return html`<div class="page"><div class="empty">Not found: <code>${current}</code></div></div>`;
}

function App() {
  const current = route.value;
  const error = globalError.value;

  useEffect(() => {
    let stopped = false;
    let handle = null;
    async function tick() {
      try {
        await refreshRuns();
      } catch (err) {
        // Keep the shell and last-good data; just surface the strip.
        globalError.value = `Dashboard API unreachable — ${err.message}`;
      }
      if (!stopped) handle = setTimeout(tick, POLL_MS);
    }
    tick();
    return () => {
      stopped = true;
      if (handle) clearTimeout(handle);
    };
  }, []);

  return html`
    <div class="app">
      <${TopNav} />
      ${error && html`<div class="error-banner"><strong>Error:</strong> ${error}</div>`}
      <div class="content">${resolvePage(current)}</div>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));

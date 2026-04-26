// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AIPERF // WORKBENCH — single-viewport operator UI.
 *
 * Unlike the previous "Flight Deck" incarnation (always-on left run rail +
 * right inspector + log strip), this shell is deliberately spare: one top
 * strip carrying the callsign + breadcrumb + LAUNCH CTA + ⌘K search, and one
 * full-bleed main viewport below. The operator is a task-focused tool, not a
 * monitoring console — the user comes here with intent (launch this run,
 * watch that run finish, grab its results, compare two runs) and the UI gets
 * out of the way of that intent.
 *
 * Routes:
 *   /                → smart Home (active run / pick-one cards / launch CTA)
 *   /launch          → YAML editor + templates, POST to the operator
 *   /run/:ns/:name   → the rich workbench for a single run
 *   /archive         → past-runs browser
 *   /compare         → side-by-side chart overlay
 *   /log             → durable run log (kept but de-emphasized)
 *
 * Legacy paths (`/jobs`, `/leaderboard`, `/compare`, `/history`, and
 * `/jobs/:ns/:name`) still resolve to the new views so deep links keep
 * working.
 */

import { html, render } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { route, matchRoute, navigate } from './lib/router.js';
import { api, poll } from './lib/api.js';
import { jobs, clusterInfo, globalError } from './lib/state.js';
import { TopRail } from './components/top-rail.js';
import { LogStrip } from './components/log-strip.js';
import { CommandPalette } from './components/command-palette.js';
import { Home } from './views/home.js';
import { Launch } from './views/launch.js';
import { Run } from './views/run.js';
import { Archive } from './views/archive.js';
import { Analysis } from './views/analysis.js';
import { Compare } from './views/compare.js';
import { Log } from './views/log.js';

function resolveView(currentRoute) {
  const runEpochMatch = matchRoute('/run/:ns/:name/runs/:epoch', currentRoute)
    ?? matchRoute('/jobs/:ns/:name/runs/:epoch', currentRoute);
  if (runEpochMatch)                                   return { kind: 'run', params: runEpochMatch };
  const runMatch = matchRoute('/run/:ns/:name', currentRoute)
    ?? matchRoute('/jobs/:ns/:name', currentRoute);
  if (runMatch)                                        return { kind: 'run', params: runMatch };
  const compareDiffMatch = matchRoute('/compare/:ns/:name/:epochA/:epochB', currentRoute);
  if (compareDiffMatch)                                return { kind: 'compare', params: compareDiffMatch };
  if (currentRoute === '/launch')                      return { kind: 'launch' };
  if (currentRoute === '/archive'
    || currentRoute === '/fleet'
    || currentRoute === '/jobs')                       return { kind: 'archive' };
  if (currentRoute === '/compare'
    || currentRoute === '/leaderboard'
    || currentRoute === '/analysis')                   return { kind: 'analysis' };
  if (currentRoute === '/log'
    || currentRoute === '/history')                    return { kind: 'log' };
  return { kind: 'home' };
}

function App() {
  const [showPalette, setShowPalette] = useState(false);

  const currentRoute = route.value;
  const resolved = resolveView(currentRoute);
  const error = globalError.value;

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      try {
        const data = await api.listJobs();
        jobs.value = data?.jobs ?? [];
      } catch (_e) { /* ignore until recovery */ }
    }, 5000, ac.signal);
    poll(async () => {
      try { clusterInfo.value = await api.getCluster(); } catch (_e) { /* ignore */ }
    }, 15000, ac.signal);
    return () => ac.abort();
  }, []);

  useEffect(() => {
    function onKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setShowPalette(v => !v);
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        navigate('/launch');
      } else if (e.key === 'Escape' && !showPalette) {
        if (resolved.kind === 'run') navigate('/');
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showPalette, resolved.kind]);

  let mainView;
  if (resolved.kind === 'run') {
    mainView = html`<${Run}
      ns=${resolved.params.ns}
      name=${resolved.params.name}
      epoch=${resolved.params.epoch ?? null}
    />`;
  } else if (resolved.kind === 'launch') {
    mainView = html`<${Launch} />`;
  } else if (resolved.kind === 'archive') {
    mainView = html`<${Archive} />`;
  } else if (resolved.kind === 'compare') {
    mainView = html`<${Compare}
      ns=${resolved.params.ns}
      name=${resolved.params.name}
      epochA=${resolved.params.epochA}
      epochB=${resolved.params.epochB}
    />`;
  } else if (resolved.kind === 'analysis') {
    mainView = html`<${Analysis} />`;
  } else if (resolved.kind === 'log') {
    mainView = html`<${Log} />`;
  } else {
    mainView = html`<${Home} />`;
  }

  return html`
    <div class="bench" data-route=${resolved.kind}>
      <${TopRail}
        viewKind=${resolved.kind}
        runParams=${resolved.params}
        onSearchClick=${() => setShowPalette(true)}
      />
      <main class="bench-main" data-testid="bench-main">
        <div class="alpha-banner" role="status" data-testid="alpha-banner">
          <span class="alpha-banner-tag">ALPHA</span>
          <span>Developer testing ground — features here are experimental and unverified. Most will change, break, or be cut before any release.</span>
        </div>
        ${error && html`
          <div class="bench-error-flash" data-testid="global-error">
            <strong>Error</strong> ${error}
          </div>
        `}
        ${mainView}
      </main>
      <${LogStrip} />
      ${showPalette && html`<${CommandPalette} onClose=${() => setShowPalette(false)} />`}
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html, render } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { route, matchRoute } from './lib/router.js';
import { globalError } from './lib/state.js';
import { initTheme } from './lib/theme-switch.js';
import { TopNav } from './components/top-nav.js';
import { Breadcrumb } from './components/breadcrumb.js';
import { CommandPalette } from './components/command-palette.js';
import { LogStrip } from './components/log-strip.js';
import { FreshnessStrip } from './components/freshness.js';
import { Dashboard } from './pages/dashboard.js';
import { Jobs } from './pages/jobs.js';
import { JobDetail } from './pages/job-detail.js';
import { Leaderboard } from './pages/leaderboard.js';
import { Compare } from './pages/compare.js';
import { CompareEpochs } from './pages/compare-epochs.js';
import { History } from './pages/history.js';
import { Sweeps } from './pages/sweeps.js';
import { SweepDetail } from './pages/sweep-detail.js';
import { Launch } from './pages/launch.js';

function App() {
  const [showPalette, setShowPalette] = useState(false);
  const [features, setFeatures] = useState({ dashboard_enabled: false });
  const currentRoute = route.value;
  const error = globalError.value;

  // Ctrl+K to open command palette
  useEffect(() => {
    function handleKey(e) {
      if ((e.ctrlKey || e.metaKey) && typeof e.key === 'string' && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setShowPalette((v) => !v);
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/v1/config/features')
      .then((r) => (r.ok ? r.json() : null))
      .then((f) => { if (!cancelled && f) setFeatures(f); })
      .catch(() => { /* features stay default — no Plots link */ });
    return () => { cancelled = true; };
  }, []);

  // Route matching
  let page;
  const jobRunMatch = matchRoute('/jobs/:ns/:name/runs/:epoch', currentRoute);
  const sweepRunMatch = matchRoute('/sweeps/:ns/:name/runs/:epoch', currentRoute);
  const jobDetailMatch = matchRoute('/jobs/:ns/:name', currentRoute);
  const sweepDetailMatch = matchRoute('/sweeps/:ns/:name', currentRoute);
  const compareEpochsMatch = matchRoute('/compare/:ns/:name/:epochA/:epochB', currentRoute);

  if (currentRoute === '/' || currentRoute === '') {
    page = html`<${Dashboard} />`;
  } else if (currentRoute === '/jobs') {
    page = html`<${Jobs} />`;
  } else if (jobRunMatch) {
    page = html`<${JobDetail} namespace=${jobRunMatch.ns} name=${jobRunMatch.name} epoch=${jobRunMatch.epoch} />`;
  } else if (jobDetailMatch) {
    page = html`<${JobDetail} namespace=${jobDetailMatch.ns} name=${jobDetailMatch.name} />`;
  } else if (currentRoute === '/sweeps') {
    page = html`<${Sweeps} />`;
  } else if (sweepRunMatch) {
    page = html`<${SweepDetail} namespace=${sweepRunMatch.ns} name=${sweepRunMatch.name} epoch=${sweepRunMatch.epoch} />`;
  } else if (sweepDetailMatch) {
    page = html`<${SweepDetail} namespace=${sweepDetailMatch.ns} name=${sweepDetailMatch.name} />`;
  } else if (currentRoute === '/leaderboard') {
    page = html`<${Leaderboard} />`;
  } else if (compareEpochsMatch) {
    page = html`<${CompareEpochs} namespace=${compareEpochsMatch.ns} name=${compareEpochsMatch.name} epochA=${compareEpochsMatch.epochA} epochB=${compareEpochsMatch.epochB} />`;
  } else if (currentRoute === '/compare') {
    page = html`<${Compare} />`;
  } else if (currentRoute === '/history') {
    page = html`<${History} />`;
  } else if (currentRoute === '/launch') {
    page = html`<${Launch} />`;
  } else {
    page = html`<div class="page-stub"><h2>Not Found</h2><p class="text-dim">${currentRoute}</p></div>`;
  }

  return html`
    <div class="app">
      <${TopNav} onSearchClick=${() => setShowPalette(true)} features=${features} />
      <${Breadcrumb} />
      <div class="alpha-banner" role="status" data-testid="alpha-banner">
        <span class="alpha-banner-tag">ALPHA</span>
        <span>Developer testing ground — features here are experimental and unverified. Most will change, break, or be cut before any release.</span>
      </div>
      <${FreshnessStrip} />
      ${error && html`
        <div class="error-banner">
          <strong>Error:</strong> ${error}
        </div>
      `}
      <div class="content">${page}</div>
      <${LogStrip} />
      ${showPalette && html`<${CommandPalette} onClose=${() => setShowPalette(false)} />`}
    </div>
  `;
}

initTheme();
render(html`<${App} />`, document.getElementById('app'));

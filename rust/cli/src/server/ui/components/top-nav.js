// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Top navigation bar with the AIPerf logo, page tabs, and a live-session
// indicator. Modeled on the operator UI's ``top-nav.js`` but wired to this
// dashboard's hash routes and shared ``meta``/``compareSel`` signals.

import { html } from 'htm/preact';
import { route, navigate } from '../lib/router.js';
import { meta, compareSel } from '../lib/state.js';

const NAV_ITEMS = [
  { path: '/', label: 'Runs' },
  { path: '/live', label: 'Live' },
  { path: '/compare', label: 'Compare' },
  { path: '/leaderboard', label: 'Leaderboard' },
  { path: '/sweeps', label: 'Sweeps' },
];

function isActive(itemPath, current) {
  if (itemPath === '/') return current === '/' || current === '' || current.startsWith('/runs');
  return current === itemPath || current.startsWith(itemPath + '/');
}

export function TopNav() {
  const current = route.value;
  const m = meta.value;
  const selCount = compareSel.value.length;
  const liveCount = m?.session_runs ?? 0;

  return html`
    <header class="topbar">
      <div class="topbar-left">
        <div class="logo" onClick=${() => navigate('/')} title="AIPerf Cross-Run Dashboard">
          <div class="logo-icon">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#0e0e10" stroke-width="2.6">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
          </div>
          <span>AIPerf <span class="logo-sub">Cross-Run</span></span>
        </div>
        <nav class="nav" aria-label="Main navigation">
          ${NAV_ITEMS.map(
            (item) => html`
              <button
                type="button"
                key=${item.path}
                class=${'nav-tab' + (isActive(item.path, current) ? ' active' : '')}
                aria-current=${isActive(item.path, current) ? 'page' : undefined}
                onClick=${() => navigate(item.path)}
              >
                ${item.label}
                ${item.path === '/compare' && selCount > 0 && html`<span class="nav-badge">${selCount}</span>`}
              </button>
            `,
          )}
        </nav>
      </div>
      <div class="topbar-right">
        ${liveCount > 0
          ? html`<span class="live-pill" title=${liveCount + ' run(s) from this live session'}>
              <span class="live-dot"></span>${liveCount} live
            </span>`
          : html`<span class="dim caption">browse mode</span>`}
      </div>
    </header>
  `;
}

// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { route, navigate } from '../lib/router.js';
import { cycleTheme, getResolvedTheme } from '../lib/theme-switch.js';

const PRIMARY_GROUP = {
  items: [
    { path: '/', label: 'Dashboard' },
    { path: '/jobs', label: 'Jobs' },
    { path: '/sweeps', label: 'Sweeps' },
    { path: '/launch', label: 'Launch' },
  ],
};

const ANALYTICS_GROUP = {
  items: [
    { path: '/leaderboard', label: 'Leaderboard' },
    { path: '/compare', label: 'Compare' },
    { path: '/history', label: 'History' },
  ],
};

function buildNavGroups(features) {
  const groups = [PRIMARY_GROUP, ANALYTICS_GROUP];
  if (features && features.dashboard_enabled) {
    groups.push({
      items: [
        {
          path: '/dashboard/',
          label: 'Plots ↗',
          external: true,
          testId: 'nav-link-plots',
        },
      ],
    });
  }
  return groups;
}

function isActive(itemPath, currentRoute) {
  if (itemPath === '/') return currentRoute === '/' || currentRoute === '';
  return currentRoute === itemPath || currentRoute.startsWith(itemPath + '/');
}

function routeSlug(path) {
  if (path === '/' || path === '') return 'dashboard';
  return path.replace(/^\//, '').replace(/\/$/, '').replace(/\//g, '-');
}

/**
 * Top navigation bar with logo, grouped tabs, and search trigger.
 * @param {{ onSearchClick: () => void, features?: { dashboard_enabled: boolean } }} props
 */
export function TopNav({ onSearchClick, features }) {
  const currentRoute = route.value;
  const navGroups = buildNavGroups(features);
  const [resolved, setResolved] = useState(getResolvedTheme());

  useEffect(() => {
    const onThemeChange = () => setResolved(getResolvedTheme());
    window.addEventListener('themechange', onThemeChange);
    return () => window.removeEventListener('themechange', onThemeChange);
  }, []);

  return html`
    <header class="topbar" data-testid="top-nav">
      <div class="topbar-left">
        <div class="logo">
          <div class="logo-icon">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          AIPerf Operator
        </div>
        <nav class="nav" aria-label="Main navigation">
          ${navGroups.map((group, gi) => html`
            ${gi > 0 && html`<span class="nav-sep" />`}
            ${group.items.map((item) => item.external ? html`
              <a
                key=${item.path}
                href=${item.path}
                target="_blank"
                rel="noopener noreferrer"
                class="nav-tab"
                data-testid=${item.testId || ('nav-link-' + routeSlug(item.path))}
              >
                ${item.label}
              </a>
            ` : html`
              <button type="button"
                key=${item.path}
                class=${'nav-tab' + (isActive(item.path, currentRoute) ? ' active' : '')}
                aria-current=${isActive(item.path, currentRoute) ? 'page' : undefined}
                onclick=${() => navigate(item.path)}
                data-testid=${item.testId || ('nav-link-' + routeSlug(item.path))}
              >
                ${item.label}
              </button>
            `)}
          `)}
        </nav>
      </div>
      <div class="topbar-right">
        <button type="button"
          class="topbar-icon-btn"
          data-testid="topbar-theme-toggle"
          title=${`Theme: ${resolved}`}
          aria-label="Toggle theme"
          onClick=${cycleTheme}
        >
          ${resolved === 'dark' ? '☾' : '☀'}
        </button>
        <button type="button"
          class="search-btn"
          onclick=${onSearchClick}
          title="Search (Ctrl+K)"
          aria-label="Open search"
          data-testid="nav-search"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Search
          <kbd>Ctrl+K</kbd>
        </button>
      </div>
    </header>
  `;
}

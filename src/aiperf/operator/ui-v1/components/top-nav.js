import { html } from 'htm/preact';
import { route, navigate } from '../lib/router.js';

const NAV_GROUPS = [
  {
    items: [
      { path: '/', label: 'Dashboard' },
      { path: '/jobs', label: 'Jobs' },
    ],
  },
  {
    items: [
      { path: '/leaderboard', label: 'Leaderboard' },
      { path: '/compare', label: 'Compare' },
      { path: '/history', label: 'History' },
    ],
  },
];

const PLOTS_LINK = { path: '/dashboard/', label: 'Plots', external: true };

function isActive(itemPath, currentRoute) {
  if (itemPath === '/') return currentRoute === '/' || currentRoute === '';
  return currentRoute.startsWith(itemPath);
}

function routeSlug(path) {
  if (path === '/' || path === '') return 'dashboard';
  return path.replace(/^\//, '').replace(/\//g, '-');
}

/**
 * Top navigation bar with logo, grouped tabs, and search trigger.
 * @param {{ onSearchClick: () => void }} props
 */
export function TopNav({ onSearchClick }) {
  const currentRoute = route.value;

  return html`
    <header class="topbar" data-testid="top-nav">
      <div class="topbar-left">
        <div class="logo">
          <div class="logo-icon">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          </div>
          AIPerf
        </div>
        <nav class="nav" aria-label="Main navigation">
          ${NAV_GROUPS.map((group, gi) => html`
            ${gi > 0 && html`<span class="nav-sep" />`}
            ${group.items.map((item) => html`
              <button
                key=${item.path}
                class=${'nav-tab' + (isActive(item.path, currentRoute) ? ' active' : '')}
                onclick=${() => navigate(item.path)}
                aria-current=${isActive(item.path, currentRoute) ? 'page' : undefined}
                data-testid=${'nav-link-' + routeSlug(item.path)}
              >
                ${item.label}
              </button>
            `)}
          `)}
          <span class="nav-sep" />
          <a
            class="nav-tab"
            href=${PLOTS_LINK.path}
            target="_blank"
            rel="noopener"
          >
            ${PLOTS_LINK.label}
            <span class="nav-external">\u2197</span>
          </a>
        </nav>
      </div>
      <div class="topbar-right">
        <button
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

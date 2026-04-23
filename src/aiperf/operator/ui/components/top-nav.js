import { html } from 'htm/preact';
import { route, navigate } from '../lib/router.js';

/**
 * CONSOLE nav items. Labels match the rack aesthetic (CONSOLE / RUNS /
 * BOARD / COMPARE / LOG), but routes and `data-testid` slugs preserve the
 * existing SPA paths so e2e tests still resolve ``nav-link-dashboard``,
 * ``nav-link-jobs``, etc.
 */
const NAV_ITEMS = [
  { path: '/', label: 'CONSOLE', icon: 'ph-waveform' },
  { path: '/jobs', label: 'RUNS', icon: 'ph-list-numbers' },
  { path: '/leaderboard', label: 'BOARD', icon: 'ph-chart-bar' },
  { path: '/compare', label: 'COMPARE', icon: 'ph-scales' },
  { path: '/history', label: 'LOG', icon: 'ph-clock-counter-clockwise' },
];

function isActive(itemPath, currentRoute) {
  if (itemPath === '/') return currentRoute === '/' || currentRoute === '';
  return currentRoute.startsWith(itemPath);
}

function routeSlug(path) {
  if (path === '/' || path === '') return 'dashboard';
  return path.replace(/^\//, '').replace(/\//g, '-');
}

/**
 * Top navigation bar — rack header with pulsing amber pilot light,
 * segmented CONSOLE/RUNS/BOARD/COMPARE/LOG nav, and Ctrl+K search trigger.
 *
 * @param {{ onSearchClick: () => void }} props
 */
export function TopNav({ onSearchClick }) {
  const currentRoute = route.value;

  return html`
    <header class="topbar" data-testid="top-nav">
      <div class="topbar-left">
        <div class="logo">
          <div class="logo-icon" aria-hidden="true"></div>
          AIPERF · CONSOLE
        </div>
        <nav class="nav" aria-label="Main navigation">
          ${NAV_ITEMS.map((item) => html`
            <button
              key=${item.path}
              class=${'nav-tab' + (isActive(item.path, currentRoute) ? ' active' : '')}
              onclick=${() => navigate(item.path)}
              aria-current=${isActive(item.path, currentRoute) ? 'page' : undefined}
              data-testid=${'nav-link-' + routeSlug(item.path)}
            >
              <i class=${'ph ' + item.icon} aria-hidden="true"></i>
              ${item.label}
            </button>
          `)}
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
          <i class="ph ph-magnifying-glass" aria-hidden="true"></i>
          Search
          <kbd>⌘K</kbd>
        </button>
      </div>
    </header>
  `;
}

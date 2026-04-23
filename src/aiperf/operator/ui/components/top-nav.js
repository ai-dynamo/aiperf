import { html } from 'htm/preact';
import { route, navigate } from '../lib/router.js';

const NAV_GROUPS = [
  {
    items: [
      { path: '/', label: 'Dashboard', icon: 'ph-squares-four' },
      { path: '/jobs', label: 'Jobs', icon: 'ph-list-bullets' },
    ],
  },
  {
    items: [
      { path: '/leaderboard', label: 'Leaderboard', icon: 'ph-chart-bar' },
      { path: '/compare', label: 'Compare', icon: 'ph-scales' },
      { path: '/history', label: 'History', icon: 'ph-clock-counter-clockwise' },
    ],
  },
];

const PLOTS_LINK = {
  path: '/dashboard/',
  label: 'Plots',
  icon: 'ph-chart-line-up',
  external: true,
};

const NAV_LINK_STYLE =
  'display: inline-flex; gap: var(--space-2); align-items: center';

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
                style=${NAV_LINK_STYLE}
                onclick=${() => navigate(item.path)}
                aria-current=${isActive(item.path, currentRoute) ? 'page' : undefined}
                data-testid=${'nav-link-' + routeSlug(item.path)}
              >
                <i class=${'ph ' + item.icon} aria-hidden="true"></i>
                ${item.label}
              </button>
            `)}
          `)}
          <span class="nav-sep" />
          <a
            class="nav-tab"
            style=${NAV_LINK_STYLE}
            href=${PLOTS_LINK.path}
            target="_blank"
            rel="noopener"
          >
            <i class=${'ph ' + PLOTS_LINK.icon} aria-hidden="true"></i>
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
          <i class="ph ph-magnifying-glass" aria-hidden="true"></i>
          Search
          <kbd>Ctrl+K</kbd>
        </button>
      </div>
    </header>
  `;
}

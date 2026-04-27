import { html } from 'htm/preact';
import { route, navigate, matchRoute } from '../lib/router.js';

/**
 * Route-aware breadcrumb bar.
 * Renders path segments based on the current hash route.
 */
export function Breadcrumb() {
  const currentRoute = route.value;

  const segments = buildSegments(currentRoute);

  if (segments.length <= 1) {
    return html`
      <div class="breadcrumb" data-testid="breadcrumb">
        <span class="breadcrumb-current">${segments[0]?.label ?? 'Dashboard'}</span>
      </div>
    `;
  }

  return html`
    <div class="breadcrumb" data-testid="breadcrumb">
      ${segments.map((seg, i) => {
        const isLast = i === segments.length - 1;
        return html`
          ${i > 0 && html`<span class="breadcrumb-sep">/</span>`}
          ${isLast
            ? html`<span class="breadcrumb-current">${seg.label}</span>`
            : html`<a href="#${seg.path}" onclick=${(e) => { e.preventDefault(); navigate(seg.path); }}>${seg.label}</a>`
          }
        `;
      })}
    </div>
  `;
}

function buildSegments(currentRoute) {
  if (currentRoute === '/' || currentRoute === '') {
    return [{ label: 'Dashboard', path: '/' }];
  }

  const jobRunMatch = matchRoute('/jobs/:ns/:name/runs/:epoch', currentRoute);
  if (jobRunMatch) {
    const detailPath = `/jobs/${jobRunMatch.ns}/${jobRunMatch.name}`;
    return [
      { label: 'Jobs', path: '/jobs' },
      { label: decodeURIComponent(jobRunMatch.ns), path: '/jobs' },
      { label: decodeURIComponent(jobRunMatch.name), path: detailPath },
      { label: 'runs', path: detailPath },
      { label: decodeURIComponent(jobRunMatch.epoch), path: currentRoute },
    ];
  }

  const sweepRunMatch = matchRoute('/sweeps/:ns/:name/runs/:epoch', currentRoute);
  if (sweepRunMatch) {
    const detailPath = `/sweeps/${sweepRunMatch.ns}/${sweepRunMatch.name}`;
    return [
      { label: 'Sweeps', path: '/sweeps' },
      { label: decodeURIComponent(sweepRunMatch.ns), path: '/sweeps' },
      { label: decodeURIComponent(sweepRunMatch.name), path: detailPath },
      { label: 'runs', path: detailPath },
      { label: decodeURIComponent(sweepRunMatch.epoch), path: currentRoute },
    ];
  }

  const jobMatch = matchRoute('/jobs/:ns/:name', currentRoute);
  if (jobMatch) {
    return [
      { label: 'Jobs', path: '/jobs' },
      { label: decodeURIComponent(jobMatch.ns), path: '/jobs' },
      { label: decodeURIComponent(jobMatch.name), path: currentRoute },
    ];
  }

  const sweepMatch = matchRoute('/sweeps/:ns/:name', currentRoute);
  if (sweepMatch) {
    return [
      { label: 'Sweeps', path: '/sweeps' },
      { label: decodeURIComponent(sweepMatch.ns), path: '/sweeps' },
      { label: decodeURIComponent(sweepMatch.name), path: currentRoute },
    ];
  }

  // Simple pages
  const PAGE_LABELS = {
    '/jobs': 'Jobs',
    '/sweeps': 'Sweeps',
    '/leaderboard': 'Leaderboard',
    '/compare': 'Compare',
    '/history': 'History',
  };

  const label = PAGE_LABELS[currentRoute];
  if (label) {
    return [{ label, path: currentRoute }];
  }

  return [{ label: currentRoute, path: currentRoute }];
}

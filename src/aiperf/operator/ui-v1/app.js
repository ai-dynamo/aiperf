import { html, render } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { route, matchRoute } from './lib/router.js';
import { globalError } from './lib/state.js';
import { TopNav } from './components/top-nav.js';
import { Breadcrumb } from './components/breadcrumb.js';
import { CommandPalette } from './components/command-palette.js';
import { Dashboard } from './pages/dashboard.js';
import { Jobs } from './pages/jobs.js';
import { JobDetail } from './pages/job-detail.js';
import { Leaderboard } from './pages/leaderboard.js';
import { Compare } from './pages/compare.js';
import { History } from './pages/history.js';

function App() {
  const [showPalette, setShowPalette] = useState(false);
  const currentRoute = route.value;
  const error = globalError.value;

  // Ctrl+K to open command palette
  useEffect(() => {
    function handleKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setShowPalette((v) => !v);
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  // Route matching
  let page;
  const jobDetailMatch = matchRoute('/jobs/:ns/:name', currentRoute);

  if (currentRoute === '/' || currentRoute === '') {
    page = html`<${Dashboard} />`;
  } else if (currentRoute === '/jobs') {
    page = html`<${Jobs} />`;
  } else if (jobDetailMatch) {
    page = html`<${JobDetail} namespace=${jobDetailMatch.ns} name=${jobDetailMatch.name} />`;
  } else if (currentRoute === '/leaderboard') {
    page = html`<${Leaderboard} />`;
  } else if (currentRoute === '/compare') {
    page = html`<${Compare} />`;
  } else if (currentRoute === '/history') {
    page = html`<${History} />`;
  } else {
    page = html`<div class="page-stub"><h2>Not Found</h2><p class="text-dim">${currentRoute}</p></div>`;
  }

  return html`
    <div class="app">
      <${TopNav} onSearchClick=${() => setShowPalette(true)} />
      <${Breadcrumb} />
      ${error && html`
        <div class="error-banner">
          <strong>Error:</strong> ${error}
        </div>
      `}
      <div class="content">${page}</div>
      ${showPalette && html`<${CommandPalette} onClose=${() => setShowPalette(false)} />`}
    </div>
  `;
}

render(html`<${App} />`, document.getElementById('app'));

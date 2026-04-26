import { html } from 'htm/preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate, route, matchRoute } from '../lib/router.js';
import { getLastNamespace } from '../lib/ns-prefs.js';

/**
 * Resolve the current namespace for namespace-scoped palette actions.
 *
 * Falls back to the sticky last-used namespace from ``ns-prefs``; if that
 * is empty too (cold first-load), returns null so callers can route to
 * the picker (``/``).
 */
function paletteNamespace() {
  return getLastNamespace() ?? null;
}

/**
 * Resolve the namespace embedded in the current route, or null when on
 * a cross-namespace tier (``/``, ``/compare``, ``/log``).
 *
 * Ordering uses the route ns (not the sticky preference) because the
 * partition the user sees should match the page they are on right now.
 */
function currentRouteNamespace() {
  const r = route.value;
  const m =
    matchRoute('/ns/:ns', r) ??
    matchRoute('/ns/:ns/launch', r) ??
    matchRoute('/ns/:ns/archive', r) ??
    matchRoute('/ns/:ns/run/:name', r) ??
    matchRoute('/ns/:ns/run/:name/runs/:epoch', r);
  return m?.ns ?? null;
}

const PAGES = [
  { label: 'Home', path: '/', hint: 'All runs' },
  { label: 'Launch', pathFn: () => {
      const ns = paletteNamespace();
      return ns ? `/ns/${encodeURIComponent(ns)}/launch` : '/';
    }, hint: '⌘N — new run' },
  { label: 'Archive', pathFn: () => {
      const ns = paletteNamespace();
      return ns ? `/ns/${encodeURIComponent(ns)}/archive` : '/';
    }, hint: 'Past runs' },
  { label: 'Compare', path: '/compare', hint: 'Analysis lab' },
  { label: 'Log', path: '/log', hint: 'Historical run log' },
];

/**
 * Simple fuzzy match: returns true if all chars of query appear in order in text.
 * @param {string} text
 * @param {string} query
 * @returns {boolean}
 */
function fuzzyMatch(text, query) {
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  let ti = 0;
  for (let qi = 0; qi < q.length; qi++) {
    while (ti < t.length && t[ti] !== q[qi]) ti++;
    if (ti >= t.length) return false;
    ti++;
  }
  return true;
}

/**
 * Command palette modal. Triggered by Ctrl+K.
 * @param {{ onClose: () => void }} props
 */
export function CommandPalette({ onClose }) {
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef(null);

  // Build items: pages + job entries. Jobs carry their own kind/ns/name so the
  // render can stamp namespace-prefixed data-testids and partition by route ns.
  const pageItems = PAGES.map((p) => ({
    kind: 'page',
    label: p.label,
    sub: p.hint ?? 'Page',
    action: () => navigate(p.pathFn ? p.pathFn() : p.path),
  }));
  const jobItems = jobs.value.map((j) => {
    // /api/v1/jobs returns flat AIPerfJobInfo records (K8sCamelModel),
    // not raw CR objects — so namespace/name live at the top level.
    const ns = j.namespace ?? 'default';
    const name = j.name ?? '';
    return {
      kind: 'job',
      namespace: ns,
      name,
      label: name,
      sub: ns,
      action: () => navigate(`/ns/${encodeURIComponent(ns)}/run/${encodeURIComponent(name)}`),
    };
  });
  const allItems = [...pageItems, ...jobItems];

  const filtered = query
    ? allItems.filter((item) => fuzzyMatch(item.label, query) || fuzzyMatch(item.sub, query))
    : allItems;

  // Partition job rows so the current-route namespace surfaces first.
  // PAGES stay where they are (top); cross-namespace tiers (no route ns)
  // skip the partition and render the existing flat order.
  const currentNs = currentRouteNamespace();
  const filteredPages = filtered.filter((it) => it.kind === 'page');
  const filteredJobs = filtered.filter((it) => it.kind === 'job');
  const currentNsJobs = currentNs
    ? filteredJobs.filter((it) => (it.namespace || 'default') === currentNs)
    : [];
  const otherNsJobs = currentNs
    ? filteredJobs.filter((it) => (it.namespace || 'default') !== currentNs)
    : filteredJobs;
  const showDivider = currentNs && currentNsJobs.length > 0 && otherNsJobs.length > 0;

  // Flat ordered list for keyboard navigation (Enter/cursor index).
  const ordered = currentNs
    ? [...filteredPages, ...currentNsJobs, ...otherNsJobs]
    : [...filteredPages, ...otherNsJobs];

  // Reset cursor when filter changes
  useEffect(() => {
    setCursor(0);
  }, [query]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Global ESC handler — inner onkeydown only fires while focus is inside the
  // palette div; a stray click elsewhere would otherwise strand the modal.
  useEffect(() => {
    function onGlobalKey(e) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', onGlobalKey);
    return () => document.removeEventListener('keydown', onGlobalKey);
  }, [onClose]);

  function handleKeyDown(e) {
    if (e.key === 'Escape') {
      onClose();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setCursor((c) => Math.min(c + 1, ordered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (e.key === 'Enter') {
      const item = ordered[cursor];
      if (item) {
        item.action();
        onClose();
      }
    }
  }

  function selectItem(item) {
    item.action();
    onClose();
  }

  function renderRow(item, flatIndex) {
    const testId =
      item.kind === 'job'
        ? `cmdp-job-${item.namespace}-${item.name}`
        : `cmdp-page-${item.label.toLowerCase()}`;
    return html`
      <li
        key=${item.kind + ':' + item.label + ':' + (item.sub ?? '')}
        class=${'cmdp-row' + (flatIndex === cursor ? ' cmdp-row--active' : '')}
        role="option"
        aria-selected=${flatIndex === cursor}
        data-testid=${testId}
        onmouseenter=${() => setCursor(flatIndex)}
        onclick=${() => selectItem(item)}
      >
        <span class="cmdp-row-label">${item.label}</span>
        <span class="cmdp-row-kind">${item.sub}</span>
      </li>
    `;
  }

  // Build rendered children with a divider between current-ns and other-ns
  // job blocks, keeping the flat-index alignment with `ordered`.
  const rendered = [];
  let idx = 0;
  for (const it of filteredPages) {
    rendered.push(renderRow(it, idx));
    idx++;
  }
  for (const it of currentNsJobs) {
    rendered.push(renderRow(it, idx));
    idx++;
  }
  if (showDivider) {
    rendered.push(html`
      <li class="cmdp-divider" data-testid="cmdp-divider" role="separator" aria-disabled="true">
        other namespaces
      </li>
    `);
  }
  for (const it of otherNsJobs) {
    rendered.push(renderRow(it, idx));
    idx++;
  }

  return html`
    <div
      class="cmdp-overlay"
      onclick=${onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div
        class="cmdp"
        onclick=${(e) => e.stopPropagation()}
        onkeydown=${handleKeyDown}
        data-testid="command-palette"
      >
        <input
          ref=${inputRef}
          type="text"
          class="cmdp-input"
          placeholder="Search runs, namespaces, or commands…"
          value=${query}
          oninput=${(e) => setQuery(e.target.value)}
          data-testid="command-palette-input"
        />
        <ul class="cmdp-list" role="listbox">
          ${ordered.length === 0
            ? html`<li class="cmdp-empty">No matches</li>`
            : rendered}
        </ul>
      </div>
    </div>
  `;
}

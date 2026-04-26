import { html } from 'htm/preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';

const PAGES = [
  { label: 'Dashboard', path: '/' },
  { label: 'Jobs', path: '/jobs' },
  { label: 'Leaderboard', path: '/leaderboard' },
  { label: 'Compare', path: '/compare' },
  { label: 'History', path: '/history' },
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

  // Build items: pages + job entries
  const allItems = [
    ...PAGES.map((p) => ({ label: p.label, sub: 'Page', action: () => navigate(p.path) })),
    ...jobs.value.map((j) => {
      // /api/v1/jobs returns flat AIPerfJobInfo records (K8sCamelModel),
      // not raw CR objects — so namespace/name live at the top level.
      const ns = j.namespace ?? 'default';
      const name = j.name ?? '';
      return {
        label: name,
        sub: ns,
        action: () => navigate(`/jobs/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`),
      };
    }),
  ];

  const filtered = query
    ? allItems.filter((item) => fuzzyMatch(item.label, query) || fuzzyMatch(item.sub, query))
    : allItems;

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
      setCursor((c) => Math.min(c + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (e.key === 'Enter') {
      const item = filtered[cursor];
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

  return html`
    <div
      class="command-palette-backdrop"
      onclick=${onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div
        class="command-palette"
        onclick=${(e) => e.stopPropagation()}
        onkeydown=${handleKeyDown}
        data-testid="command-palette"
      >
        <div class="command-palette-search">
          <svg class="command-palette-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"/>
            <line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          <input
            ref=${inputRef}
            type="text"
            class="command-palette-input"
            placeholder="Search pages and jobs..."
            value=${query}
            oninput=${(e) => setQuery(e.target.value)}
            data-testid="command-palette-input"
          />
          <kbd class="command-palette-esc">Esc</kbd>
        </div>
        <ul class="command-palette-list" role="listbox">
          ${filtered.length === 0 && html`
            <li class="command-palette-empty">No results for "${query}"</li>
          `}
          ${filtered.map(
            (item, i) => html`
              <li
                key=${item.label + item.sub}
                class=${'command-palette-item' + (i === cursor ? ' command-palette-item--active' : '')}
                role="option"
                aria-selected=${i === cursor}
                onmouseenter=${() => setCursor(i)}
                onclick=${() => selectItem(item)}
              >
                <span class="command-palette-item-label">${item.label}</span>
                <span class="command-palette-item-sub">${item.sub}</span>
              </li>
            `,
          )}
        </ul>
      </div>
    </div>
  `;
}

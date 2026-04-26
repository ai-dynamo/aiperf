import { html } from 'htm/preact';
import { useState, useEffect, useRef } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';

const PAGES = [
  { label: 'Home', path: '/', hint: 'All runs' },
  { label: 'Launch', path: '/launch', hint: '⌘N — new run' },
  { label: 'Archive', path: '/archive', hint: 'Past runs' },
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

  // Build items: pages + job entries
  const allItems = [
    ...PAGES.map((p) => ({ label: p.label, sub: p.hint ?? 'Page', action: () => navigate(p.path) })),
    ...jobs.value.map((j) => {
      // /api/v1/jobs returns flat AIPerfJobInfo records (K8sCamelModel),
      // not raw CR objects — so namespace/name live at the top level.
      const ns = j.namespace ?? 'default';
      const name = j.name ?? '';
      return {
        label: name,
        sub: ns,
        action: () => navigate(`/run/${encodeURIComponent(ns)}/${encodeURIComponent(name)}`),
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
        <div class="cmdp-list">
          ${filtered.length === 0
            ? html`<div class="cmdp-empty">No matches</div>`
            : filtered.map(
              (item, i) => html`
                <div
                  key=${item.label + item.sub}
                  class=${'cmdp-row' + (i === cursor ? ' cmdp-row--active' : '')}
                  role="option"
                  aria-selected=${i === cursor}
                  onmouseenter=${() => setCursor(i)}
                  onclick=${() => selectItem(item)}
                >
                  <span class="cmdp-row-label">${item.label}</span>
                  <span class="cmdp-row-kind">${item.sub}</span>
                </div>
              `,
            )}
        </div>
      </div>
    </div>
  `;
}

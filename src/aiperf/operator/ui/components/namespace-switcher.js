// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NAMESPACE SWITCHER — compact dropdown rendered from the breadcrumb pill.
 *
 * Same data source as the picker (group ``jobs.value`` by namespace),
 * but compact: name + a single phase-chip row + last-activity. Selecting
 * an item navigates to the namespace overview (``/ns/<chosen>``); the
 * "View all namespaces" footer item navigates to ``/``.
 */

import { html } from 'htm/preact';
import { useMemo, useState } from 'preact/hooks';
import { jobs } from '../lib/state.js';
import { navigate } from '../lib/router.js';
import { setLastNamespace } from '../lib/ns-prefs.js';

export function NamespaceSwitcher({ currentNs, onClose }) {
  const [query, setQuery] = useState('');
  const list = jobs.value ?? [];

  const items = useMemo(() => {
    const groups = new Map();
    for (const j of list) {
      const ns = j.namespace || 'default';
      if (!groups.has(ns)) groups.set(ns, { name: ns, running: 0, total: 0 });
      const g = groups.get(ns);
      g.total += 1;
      const p = (j.phase ?? '').toLowerCase();
      if (p === 'running' || p === 'initializing' || p === 'pending') g.running += 1;
    }
    return Array.from(groups.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [list]);

  const filtered = query
    ? items.filter(i => i.name.toLowerCase().includes(query.toLowerCase()))
    : items;

  function pick(name) {
    setLastNamespace(name);
    onClose?.();
    navigate('/ns/' + encodeURIComponent(name));
  }

  return html`
    <div class="ns-switcher-dropdown" data-testid="ns-switcher-dropdown">
      <input
        class="ns-switcher-search"
        data-testid="ns-switcher-search"
        autofocus
        placeholder="filter namespaces…"
        value=${query}
        oninput=${(e) => setQuery(e.target.value)}
      />
      <div class="ns-switcher-list">
        ${filtered.map(i => html`
          <button
            class=${'ns-switcher-item' + (i.name === currentNs ? ' ns-switcher-item--current' : '')}
            data-testid=${'ns-switcher-item-' + i.name}
            onclick=${() => pick(i.name)}
          >
            <span class="ns-switcher-name">${i.name}</span>
            <span class="ns-switcher-meta">${i.running} / ${i.total}</span>
          </button>
        `)}
      </div>
      <button
        class="ns-switcher-view-all"
        data-testid="ns-switcher-view-all"
        onclick=${() => { onClose?.(); navigate('/'); }}
      >View all namespaces →</button>
    </div>
  `;
}

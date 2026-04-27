import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';

/**
 * Reusable epoch dropdown + "viewing N of M" banner.
 *
 * Props:
 *   epochs:   [{ epoch, isLatest, mtimeEpoch, fileCount }]
 *   current:  string|undefined  — the epoch the user is viewing (undefined === latest)
 *   onPick:   (epoch:string|undefined) => void  — undefined when user picks the latest pseudo-row
 */
export function EpochSelector({ epochs, current, onPick }) {
  if (!epochs || epochs.length === 0) {
    return html`<div data-testid="epoch-selector" class="text-dim" style="font-size:11px">
      No persisted epochs.
    </div>`;
  }

  const latest = epochs.find(e => e.isLatest);
  const sortedDesc = [...epochs].sort((a, b) => b.epoch.localeCompare(a.epoch));
  const isCurrentLatest = !current || (latest && current === latest.epoch);

  return html`
    <div data-testid="epoch-selector" style="display:flex;gap:var(--space-2);align-items:center">
      <label class="text-dim" style="font-size:11px">Epoch:</label>
      <select
        value=${current ?? '__latest__'}
        onchange=${e => {
          const v = e.target.value;
          onPick(v === '__latest__' ? undefined : v);
        }}
        style=${`padding:var(--space-1) var(--space-2);background:${palette.mantle};
                 border:1px solid ${palette.surface0};border-radius:var(--radius-sm);
                 color:${palette.text};font-size:var(--font-size-sm)`}
      >
        <option value="__latest__">latest${latest ? ` (${latest.epoch})` : ''}</option>
        ${sortedDesc.map(e => html`
          <option key=${e.epoch} value=${e.epoch}>
            ${e.epoch}${e.isLatest ? ' · latest' : ''}
          </option>
        `)}
      </select>
      ${!isCurrentLatest && html`
        <span data-testid="epoch-banner-not-latest" class="text-dim" style="font-size:11px">
          viewing ${current} of ${epochs.length} ·
          <a href="#" onclick=${ev => { ev.preventDefault(); onPick(undefined); }}>jump to latest</a>
        </span>
      `}
    </div>
  `;
}

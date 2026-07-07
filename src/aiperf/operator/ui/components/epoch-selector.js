// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { palette } from '../lib/theme.js';
import { EpochPill } from './pills.js';

/**
 * Reusable epoch pill + dropdown. Always shows the active epoch as a pill
 * (even on latest, so the user can never wonder which epoch they are on).
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
  const sortedDesc = [...epochs].sort((a, b) =>
    (b?.epoch ?? '').localeCompare(a?.epoch ?? '')
  );
  const isCurrentLatest = !current || (latest && current === latest.epoch);
  const activeEpoch = current ?? latest?.epoch;
  // With only one epoch there's nothing to switch to — render the pill
  // alone so users don't get a dropdown caret that opens a one-row menu.
  const hasChoice = epochs.length > 1;

  if (!hasChoice) {
    return html`
      <div data-testid="epoch-selector" style="display:flex;gap:var(--space-2);align-items:center;flex-wrap:wrap">
        <${EpochPill} epoch=${activeEpoch} isLatest=${isCurrentLatest} testId="epoch-selector-pill" />
      </div>
    `;
  }

  return html`
    <div data-testid="epoch-selector" style="display:flex;gap:var(--space-2);align-items:center;flex-wrap:wrap">
      <span style="position:relative;display:inline-flex;align-items:center">
        <${EpochPill} epoch=${activeEpoch} isLatest=${isCurrentLatest} testId="epoch-selector-pill" />
        <select
          aria-label="Select epoch"
          value=${current ?? '__latest__'}
          onchange=${e => {
            const v = e.target.value;
            onPick(v === '__latest__' ? undefined : v);
          }}
          style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer;font-size:11px"
        >
          <option value="__latest__">latest${latest ? ` (${latest.epoch})` : ''}</option>
          ${sortedDesc.map(e => html`
            <option key=${e.epoch} value=${e.epoch}>
              ${e.epoch}${e.isLatest ? ' · latest' : ''}
            </option>
          `)}
        </select>
      </span>
      ${!isCurrentLatest && html`
        <span data-testid="epoch-banner-not-latest" class="text-dim" style="font-size:11px">
          ${epochs.length} total ·
          <button
            type="button"
            onclick=${() => onPick(undefined)}
            style=${`background:none;border:none;padding:0;color:${palette.blue};text-decoration:underline;cursor:pointer;font-size:11px;font-family:inherit`}
          >jump to latest</button>
        </span>
      `}
    </div>
  `;
}

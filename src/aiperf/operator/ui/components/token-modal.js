// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { palette } from '../lib/theme.js';

/**
 * Modal overlay that collects the operator bearer token from the user and
 * stores it in sessionStorage. Rendered by pages that call mutating API
 * routes when no token is present.
 *
 * @param {{ onConfirm: (token: string) => void, onCancel: () => void }} props
 */
export function TokenModal({ onConfirm, onCancel }) {
  const [token, setToken] = useState('');
  const inputRef = useRef(null);
  const dialogRef = useRef(null);

  // The overlay covers the page, so a keyboard user who cannot see a way out
  // is trapped: Escape has to dismiss it. `autofocus` is not enough on a node
  // that preact inserts after first paint, so move focus explicitly.
  useEffect(() => {
    inputRef.current?.focus();
    const onKeyDown = (e) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onCancel]);

  function submit(e) {
    e.preventDefault();
    const t = token.trim();
    if (t) onConfirm(t);
  }

  function trapFocus(e) {
    if (e.key !== 'Tab') return;
    const tabbable = [...dialogRef.current.querySelectorAll(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [href]:not([tabindex="-1"]), [tabindex]:not([tabindex="-1"])',
    )].filter((element) => element.offsetParent !== null);
    if (tabbable.length === 0) return;
    const first = tabbable[0];
    const last = tabbable[tabbable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  const overlayStyle = [
    'position:fixed; inset:0; z-index:1000',
    'display:flex; align-items:center; justify-content:center',
    `background:${palette.base}cc`,
  ].join('; ');

  const cardStyle = [
    `background:var(--bg-tile); border:1px solid ${palette.surface1}`,
    'border-radius:var(--radius-md); padding:var(--space-6)',
    'width:420px; max-width:92vw',
  ].join('; ');

  const inputStyle = [
    'width:100%; box-sizing:border-box',
    'font-family:var(--font-mono); font-size:var(--font-size-sm)',
    `padding:var(--space-2) var(--space-3); background:var(--bg)`,
    `border:1px solid ${palette.surface1}; border-radius:var(--radius-sm)`,
    `color:${palette.text}; margin-bottom:var(--space-4)`,
    'outline:none',
  ].join('; ');

  return html`
    <div style=${overlayStyle} onclick=${(e) => { if (e.target === e.currentTarget) onCancel(); }}>
      <div ref=${dialogRef} style=${cardStyle} role="dialog" aria-modal="true" aria-label="API token required" onkeydown=${trapFocus}>
        <div style=${`font-weight:600; margin-bottom:var(--space-1); color:${palette.text}; font-size:var(--font-size-base)`}>
          API token required
        </div>
        <div style=${`font-size:var(--font-size-sm); color:${palette.subtext1}; margin-bottom:var(--space-4); line-height:1.5`}>
          Enter the value of${' '}
          <code style=${`font-family:var(--font-mono); color:${palette.text}`}>AIPERF_OPERATOR_MUTATING_ROUTES_TOKEN</code>${' '}
          from the operator. The token is kept in
          sessionStorage for this tab only and cleared on close.
        </div>
        <form onsubmit=${submit}>
          <input
            ref=${inputRef}
            type="password"
            style=${inputStyle}
            placeholder="Bearer token"
            value=${token}
            oninput=${(e) => setToken(e.target.value)}
            autofocus
            data-testid="token-modal-input"
          />
          <div style="display:flex; gap:var(--space-2); justify-content:flex-end">
            <button
              type="button"
              class="btn btn--ghost"
              onclick=${onCancel}
              data-testid="token-modal-cancel"
            >Cancel</button>
            <button
              type="submit"
              class="btn btn--primary"
              disabled=${!token.trim()}
              data-testid="token-modal-confirm"
            >Authenticate</button>
          </div>
        </form>
      </div>
    </div>
  `;
}

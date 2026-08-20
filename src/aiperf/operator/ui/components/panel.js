// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical chrome for any titled section on the job-detail page.
 *
 * One of:
 *   - Controlled: pass ``open`` + ``onToggle``.
 *   - Uncontrolled collapsible: pass ``collapsible=true`` and optional ``defaultOpen``.
 *   - Static: pass neither.
 *
 * Visual: 1px border (tone-tinted), 6px radius, 6px 10px header, 8px 10px body.
 * Header has uppercase green title (font-size-xs / accent), optional pill badge,
 * optional collapse arrow on the far right.
 */

import { html } from 'htm/preact';
import { useState } from 'preact/hooks';

export function Panel({
  title,
  badge,
  badgeTone,            // 'neutral' | 'warn' | 'bad'
  tone,                 // 'neutral' | 'good' | 'warn' | 'bad' — border tint
  collapsible = false,
  defaultOpen = true,
  open: controlledOpen,
  onToggle,
  children,
  testId,
}) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(defaultOpen);
  const isControlled = controlledOpen !== undefined;
  const open = isControlled ? controlledOpen : uncontrolledOpen;

  const handleToggle = () => {
    if (!collapsible) return;
    if (isControlled) onToggle && onToggle(!open);
    else setUncontrolledOpen((v) => !v);
  };

  const toneClass = tone ? ` panel--tone-${tone}` : '';
  const badgeToneClass = badgeTone ? ` panel-badge--${badgeTone}` : '';

  return html`
    <div class=${'panel' + toneClass} data-testid=${testId}>
      <div class=${'panel-h' + (collapsible ? ' panel-h--clickable' : '')}
           onClick=${handleToggle}
           role=${collapsible ? 'button' : undefined}
           tabindex=${collapsible ? 0 : undefined}>
        <span class="panel-h-title">${title}</span>
        ${badge != null && html`<span class=${'panel-badge' + badgeToneClass}>${badge}</span>`}
        ${collapsible && html`<span class="panel-h-arrow" aria-hidden="true">${open ? '▾' : '▸'}</span>`}
      </div>
      ${open && html`<div class="panel-b">${children}</div>`}
    </div>
  `;
}

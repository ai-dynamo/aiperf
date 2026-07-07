// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';

const SIZE_PRESETS = { xs: 10, sm: 12, md: 16, lg: 20, xl: 28 };

// Polymorphic size: callers historically passed either pixel numbers
// (size={14}) or string presets (size="sm"). The string form was silently
// broken until now — it interpolated as `width: smpx` and the browser fell
// back to default sizing. Centralising resolution here keeps both callsite
// styles working without touching any caller.
function resolveSize(size) {
  if (size == null) return 16;
  if (typeof size === 'number') return size;
  const preset = SIZE_PRESETS[size];
  return preset ?? 16;
}

/**
 * Animated CSS spinner used everywhere a page is fetching data.
 *
 * Pages used to show plain "Loading…" text, which left a static empty
 * panel during slow fetches and made users assume there was no data.
 * Always pair the spinner with a label so the user knows *what* is
 * loading.
 *
 * @param {object} props
 * @param {number|('xs'|'sm'|'md'|'lg'|'xl')} [props.size] - Pixel number
 *   (e.g. ``14``) or named preset: ``xs``=10, ``sm``=12, ``md``=16,
 *   ``lg``=20, ``xl``=28. Defaults to 16 (``md``).
 * @param {number} [props.thickness] - Border thickness in pixels.
 * @param {string} [props.color] - CSS color for the spinning arc.
 */
export function Spinner({ size, thickness = 2, color }) {
  const px = resolveSize(size);
  const stroke = color ?? 'var(--mauve)';
  const style =
    'display: inline-block;'
    + ' box-sizing: border-box;'
    + ` width: ${px}px;`
    + ` height: ${px}px;`
    + ` border: ${thickness}px solid ${stroke};`
    + ' border-top-color: transparent;'
    + ' border-radius: 50%;'
    + ' animation: ui-spinner-rotate 0.8s linear infinite;'
    + ' vertical-align: middle;';
  return html`<span class="ui-spinner" style=${style} aria-label="loading" role="status"></span>`;
}

/**
 * Centered spinner + label, sized for full-card or full-page placeholders.
 * Use ``inline=true`` for the small in-row "Loading…" affordance inside
 * scroll panels.
 *
 * @param {object} props
 * @param {string} [props.label] - Text shown next to the spinner.
 * @param {string} [props.padding] - Override CSS padding shorthand.
 * @param {number|('xs'|'sm'|'md'|'lg'|'xl')} [props.size] - Forwarded to
 *   ``Spinner``; accepts pixel number or named preset (see ``Spinner``).
 *   Defaults to 18.
 * @param {boolean} [props.inline] - Left-align with smaller padding.
 * @param {string} [props.testid] - ``data-testid`` for harness selectors.
 */
export function LoadingPanel({
  label = 'Loading…',
  padding,
  size = 18,
  inline = false,
  testid,
}) {
  const pad = padding ?? (inline ? 'var(--space-3)' : 'var(--space-6) var(--space-4)');
  const justify = inline ? 'flex-start' : 'center';
  return html`
    <div
      data-testid=${testid}
      style=${'display: flex; align-items: center; justify-content: ' + justify
        + '; gap: var(--space-2); padding: ' + pad + '; color: var(--subtext0)'}
    >
      <${Spinner} size=${size} />
      <span class="text-dim" style="font-size: var(--font-size-sm)">${label}</span>
    </div>
  `;
}

/**
 * Skeleton bar — alternative loading affordance for table rows / list items
 * where a centered spinner would jump the layout. Pulses opacity rather
 * than rotating, matching the existing ``.status-dot.live`` pulse rhythm.
 */
export function SkeletonBar({ width = '100%', height = 14, radius = 4 }) {
  const w = typeof width === 'number' ? width + 'px' : width;
  const style =
    'display: block;'
    + ` width: ${w};`
    + ` height: ${height}px;`
    + ` border-radius: ${radius}px;`
    + ' background: var(--surface0);'
    + ' animation: ui-skeleton-pulse 1.4s ease-in-out infinite;';
  return html`<span class="ui-skeleton" style=${style}></span>`;
}

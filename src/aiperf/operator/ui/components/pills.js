// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';

function metaStyle(clickable) {
  return clickable ? 'cursor:pointer;' : '';
}

/**
 * Namespace metadata. Optional onClick filters/navigates to that namespace.
 *
 * Props:
 *   ns:      string
 *   onClick: (ns:string) => void | undefined
 *   testId:  string | undefined
 */
export function NsPill({ ns, onClick, testId }) {
  if (!ns) return null;
  const clickable = typeof onClick === 'function';
  const handler = clickable
    ? (e) => { e.stopPropagation(); onClick(ns); }
    : undefined;
  const title = clickable ? `Filter by namespace: ${ns}` : `Namespace: ${ns}`;
  return html`
    <span
      class=${'meta-pill' + (clickable ? ' meta-pill--clickable' : '')}
      style=${metaStyle(clickable)}
      title=${title}
      data-testid=${testId ?? 'ns-pill'}
      onclick=${handler}
    >
      <span class="meta-pill__prefix">ns</span>${ns}
    </span>
  `;
}

/**
 * Static epoch metadata. For an interactive selector use EpochSelector.
 *
 * Props:
 *   epoch:    string
 *   isLatest: boolean
 *   testId:   string | undefined
 */
export function EpochPill({ epoch, isLatest, testId }) {
  if (!epoch) return null;
  return html`
    <span
      class="meta-pill"
      style=${metaStyle(false)}
      title=${`Epoch: ${epoch}${isLatest ? ' (latest)' : ''}`}
      data-testid=${testId ?? 'epoch-pill'}
    >
      <span class="meta-pill__prefix">ep</span>${epoch}
      ${isLatest && html`<span class="meta-pill__suffix"> · latest</span>`}
    </span>
  `;
}

/**
 * Model metadata. Optional onClick for click-to-filter.
 *
 * Props:
 *   model:   string
 *   onClick: (model:string) => void | undefined
 *   testId:  string | undefined
 */
export function ModelPill({ model, onClick, testId }) {
  if (!model) return null;
  const clickable = typeof onClick === 'function';
  const handler = clickable
    ? (e) => { e.stopPropagation(); onClick(model); }
    : undefined;
  return html`
    <span
      class=${'meta-pill meta-pill--model' + (clickable ? ' meta-pill--clickable' : '')}
      style=${clickable ? 'cursor:pointer;' : ''}
      title=${clickable ? `Filter by model: ${model}` : `Model: ${model}`}
      data-testid=${testId ?? 'model-pill'}
      onclick=${handler}
    >
      ${model}
    </span>
  `;
}

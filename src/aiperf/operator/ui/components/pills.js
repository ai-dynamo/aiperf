// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { palette, modelColor } from '../lib/theme.js';

const NS_COLOR = palette.teal;
const EPOCH_COLOR = palette.indigo;

function pillStyle(color, clickable) {
  return (
    'background:' + color + '22;' +
    'color:' + color + ';' +
    'border-color:' + color + '55;' +
    (clickable ? 'cursor:pointer;' : '')
  );
}

/**
 * Namespace pill. Optional onClick filters/navigates to that namespace.
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
      style=${pillStyle(NS_COLOR, clickable)}
      title=${title}
      data-testid=${testId ?? 'ns-pill'}
      onclick=${handler}
    >
      <span class="meta-pill__prefix">ns</span>${ns}
    </span>
  `;
}

/**
 * Epoch pill. Static (non-interactive) display of an epoch value.
 * For an interactive selector use the EpochSelector component which embeds this.
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
      style=${pillStyle(EPOCH_COLOR, false)}
      title=${`Epoch: ${epoch}${isLatest ? ' (latest)' : ''}`}
      data-testid=${testId ?? 'epoch-pill'}
    >
      <span class="meta-pill__prefix">ep</span>${epoch}
      ${isLatest && html`<span class="meta-pill__suffix"> · latest</span>`}
    </span>
  `;
}

/**
 * Model pill. Colored dot (stable hash of model name) plus the model name
 * on a neutral-tinted chip. Optional onClick for click-to-filter.
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
  const dotColor = modelColor(model);
  return html`
    <span
      class=${'meta-pill meta-pill--model' + (clickable ? ' meta-pill--clickable' : '')}
      style=${clickable ? 'cursor:pointer;' : ''}
      title=${clickable ? `Filter by model: ${model}` : `Model: ${model}`}
      data-testid=${testId ?? 'model-pill'}
      onclick=${handler}
    >
      <span class="meta-pill__dot" style=${'background:' + dotColor}></span>${model}
    </span>
  `;
}

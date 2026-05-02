// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';
import { PodHeatmap } from './pod-heatmap.js';
import { fmtInt } from '../lib/format.js';

/**
 * Pods-status strip with embedded heatmap. Replaces the slim portion of the
 * old PodsBar component; the full pod table moves into DiagnosticsPanel.PodsTab.
 *
 * Click anywhere in the bar (or any tile) → calls `onExpand` so the parent
 * can navigate to ?diag=pods.
 */
export function PodsStrip({ pods, onExpand }) {
  const list = Array.isArray(pods) ? pods : [];
  const ready = list.filter((p) => {
    const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
    return ph === 'running';
  }).length;
  const failed = list.filter((p) => {
    const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
    const reason = (p?.reason ?? '').toLowerCase();
    return ph === 'failed' || reason.includes('crashloop');
  }).length;
  const pending = list.filter((p) => {
    const ph = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
    return ph === 'pending';
  }).length;

  const metaParts = [];
  if (failed) metaParts.push(`${fmtInt(failed)} crashloop`);
  if (pending) metaParts.push(`${fmtInt(pending)} pending`);
  if (metaParts.length === 0) metaParts.push('all healthy');
  metaParts.push('click to expand');

  return html`
    <${Strip} label=${`pods ${fmtInt(ready)}/${fmtInt(list.length)}`}
              testId="strip-pods"
              onBarClick=${onExpand}
              meta=${metaParts.join(' · ')}>
      <${PodHeatmap} pods=${list} onPodClick=${onExpand} />
    <//>
  `;
}

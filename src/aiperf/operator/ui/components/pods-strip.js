// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { Strip } from './strip.js';
import { PodHeatmap } from './pod-heatmap.js';
import { fmtInt } from '../lib/format.js';

function podPhase(pod) {
  return (pod?.phase ?? pod?.status?.phase ?? '').toLowerCase();
}

function podWaitingReasons(pod) {
  const statuses = pod?.status?.containerStatuses ?? pod?.containerStatuses ?? [];
  return statuses
    .map((status) => status?.state?.waiting?.reason ?? status?.waiting?.reason ?? '')
    .filter(Boolean);
}

export function hasPodCrashLoop(pod) {
  const reason = pod?.reason ?? pod?.status?.reason ?? '';
  return [reason, ...podWaitingReasons(pod)].some((value) => /crashloop/i.test(value));
}

/**
 * Pods-status strip with embedded heatmap. Replaces the slim portion of the
 * old PodsBar component; the full pod table moves into DiagnosticsPanel.PodsTab.
 *
 * Click anywhere in the bar (or any tile) → calls `onExpand` so the parent
 * can navigate to ?diag=pods.
 */
export function PodsStrip({ pods, onExpand }) {
  const list = Array.isArray(pods) ? pods : [];
  const ready = list.filter((p) => p.ready).length;
  const failed = list.filter((p) => {
    const ph = podPhase(p);
    return ph === 'failed' || hasPodCrashLoop(p);
  }).length;
  const pending = list.filter((p) => podPhase(p) === 'pending').length;

  const metaParts = [];
  if (list.length === 0) metaParts.push('no pods');
  else if (failed) metaParts.push(`${fmtInt(failed)} crashloop`);
  if (pending) metaParts.push(`${fmtInt(pending)} pending`);
  if (list.length > 0 && metaParts.length === 0) metaParts.push(`${'all healthy'}`);
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

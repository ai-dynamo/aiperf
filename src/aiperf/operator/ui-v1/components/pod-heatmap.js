// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compact per-pod heatmap. One 6×6px tile per pod, flex-wrapped to fill
 * the bar slot.  Designed to scale to ≥1000 pods at modest row count.
 *
 * Pod state colors:
 *   - Running   → green
 *   - Pending   → muted-green
 *   - Succeeded → blue
 *   - Failed / CrashLoopBackOff → red
 *   - Unknown   → grey
 */

import { html } from 'htm/preact';

function classifyPod(p) {
  const phase = (p?.phase ?? p?.status?.phase ?? '').toLowerCase();
  const reason = (p?.reason ?? '').toLowerCase();
  if (phase === 'running') return 'running';
  if (phase === 'pending') return 'pending';
  if (phase === 'succeeded') return 'succeeded';
  if (phase === 'failed' || reason.includes('crashloop') || reason === 'erorr') return 'failed';
  return 'unknown';
}

export function PodHeatmap({ pods, onPodClick, testId }) {
  const list = Array.isArray(pods) ? pods : [];
  return html`
    <div class="pod-heatmap" data-testid=${testId} role="img"
         aria-label=${`${list.length} pod tiles`}>
      ${list.map((p, i) => {
        const cls = classifyPod(p);
        const name = p?.name ?? p?.metadata?.name ?? `pod-${i}`;
        const node = p?.node ?? p?.spec?.nodeName ?? '';
        const tooltip = `${name}${node ? ` · ${node}` : ''} · ${cls}`;
        return html`
          <span class=${'pod-heatmap-tile pod-heatmap-tile--' + cls}
                title=${tooltip}
                onClick=${onPodClick ? () => onPodClick(p) : undefined}
                key=${name + '-' + i}></span>
        `;
      })}
    </div>
  `;
}

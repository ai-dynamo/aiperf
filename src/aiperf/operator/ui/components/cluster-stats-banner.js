// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';

const POLL_MS = 30_000;

/**
 * Pick a tone for the utilization figure based on the same thresholds
 * gpu-report.sh uses (<50 ok, <80 warn, ≥80 bad).
 */
function utilTone(pct) {
  if (pct < 50) return 'ok';
  if (pct < 80) return 'warn';
  return 'bad';
}

// ─────────────────────────────────────────────────────────────────────────
// Material-style outlined SVG icons. Inline so we don't pull in a font
// or icon library; sized via the wrapper .cluster-banner__icon.
// All are 24×24 viewBox, 1.8 stroke-width, rounded joins — matches the
// Material Symbols Outlined style without the font dependency.
// ─────────────────────────────────────────────────────────────────────────

function IconGpu() {
  return html`
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="2.5" x2="9" y2="5" />
      <line x1="15" y1="2.5" x2="15" y2="5" />
      <line x1="9" y1="19" x2="9" y2="21.5" />
      <line x1="15" y1="19" x2="15" y2="21.5" />
      <line x1="2.5" y1="9" x2="5" y2="9" />
      <line x1="2.5" y1="15" x2="5" y2="15" />
      <line x1="19" y1="9" x2="21.5" y2="9" />
      <line x1="19" y1="15" x2="21.5" y2="15" />
    </svg>
  `;
}

function IconSpeed() {
  // Half-circle gauge with a needle pointing top-right.
  return html`
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M3 16 a9 9 0 0 1 18 0" />
      <line x1="12" y1="16" x2="16.5" y2="9" />
      <circle cx="12" cy="16" r="1.4" fill="currentColor" stroke="none" />
      <line x1="12" y1="19.5" x2="12" y2="20.5" />
      <line x1="4.6" y1="13" x2="5.6" y2="13.4" />
      <line x1="19.4" y1="13" x2="18.4" y2="13.4" />
      <line x1="7.5" y1="8.5" x2="8.2" y2="9.2" />
    </svg>
  `;
}

function IconNodes() {
  // Three stacked server bays with status LEDs.
  return html`
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="4" width="18" height="5" rx="1.2" />
      <rect x="3" y="10" width="18" height="5" rx="1.2" />
      <rect x="3" y="16" width="18" height="4" rx="1.2" />
      <circle cx="6.5" cy="6.5" r="0.8" fill="currentColor" stroke="none" />
      <circle cx="6.5" cy="12.5" r="0.8" fill="currentColor" stroke="none" />
      <circle cx="6.5" cy="18" r="0.8" fill="currentColor" stroke="none" />
      <line x1="9.5" y1="6.5" x2="14" y2="6.5" />
      <line x1="9.5" y1="12.5" x2="14" y2="12.5" />
      <line x1="9.5" y1="18" x2="14" y2="18" />
    </svg>
  `;
}

function IconK8s() {
  // Hexagonal helm-like outline with internal radii (Kubernetes logo motif).
  return html`
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <polygon points="12,2.5 20.5,7 20.5,17 12,21.5 3.5,17 3.5,7" />
      <circle cx="12" cy="12" r="3.2" />
      <line x1="12" y1="2.5" x2="12" y2="8.8" />
      <line x1="12" y1="15.2" x2="12" y2="21.5" />
      <line x1="20.5" y1="7" x2="14.7" y2="10.4" />
      <line x1="3.5" y1="7" x2="9.3" y2="10.4" />
      <line x1="20.5" y1="17" x2="14.7" y2="13.6" />
      <line x1="3.5" y1="17" x2="9.3" y2="13.6" />
    </svg>
  `;
}

// ─────────────────────────────────────────────────────────────────────────

/**
 * Top-of-page banner showing cluster-wide GPU usage and Kubernetes
 * version. Polls /api/v1/cluster every 30s. Renders nothing until
 * the first response lands so it never flashes a "0 GPUs" placeholder.
 *
 * Schema: ../../routers/jobs_models.py::ClusterResponse
 */
export function ClusterStatsBanner() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      try {
        setInfo(await api.getCluster());
        setError(false);
      } catch (_e) {
        // Best-effort: leave the prior value in place so a brief 503
        // during operator restart doesn't clear the banner. Surface a
        // small "stale" hint after the first failure.
        if (!info) setError(true);
      }
    }, POLL_MS, ac.signal);
    return () => ac.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!info && !error) return null;

  if (!info && error) {
    return html`
      <div class="cluster-banner cluster-banner--degraded" role="status" data-testid="cluster-stats">
        <div class="cluster-banner__tile">
          <div class="cluster-banner__icon"><${IconK8s} /></div>
          <div class="cluster-banner__body">
            <div class="cluster-banner__label">Cluster</div>
            <div class="cluster-banner__value cluster-banner__value--bad">unavailable</div>
            <div class="cluster-banner__sub">/api/v1/cluster failed — see operator logs</div>
          </div>
        </div>
      </div>
    `;
  }

  const {
    nodes = 0,
    gpus = 0,
    gpus_used: used = 0,
    gpus_free: free = 0,
    utilization_percent: util = 0,
    gpu_nodes: gpuNodes = 0,
    nodes_free: nFree = 0,
    nodes_partial: nPartial = 0,
    nodes_full: nFull = 0,
    kubernetes_version: k8sVersion = 'unknown',
    cluster_name: clusterName = null,
  } = info;

  // JSON nulls bypass destructuring defaults; coerce to 0 / safe display
  // so a partially-degraded /api/v1/cluster response doesn't crash on
  // ``.toFixed`` or render literal "null".
  const safeNodes = nodes ?? 0;
  const safeGpus = gpus ?? 0;
  const safeUsed = used ?? 0;
  const safeFree = free ?? 0;
  const safeUtil = util ?? 0;
  const safeGpuNodes = gpuNodes ?? 0;
  const safeNFree = nFree ?? 0;
  const safeNPartial = nPartial ?? 0;
  const safeNFull = nFull ?? 0;
  const safeK8sVersion = k8sVersion || 'unknown';
  const safeClusterName = (clusterName && String(clusterName).trim()) || null;

  const hasGpus = safeGpus > 0;
  const tone = utilTone(safeUtil);

  return html`
    <div class="cluster-banner" role="status" data-testid="cluster-stats">
      ${hasGpus && html`
        <div class="cluster-banner__tile" data-testid="banner-gpus" title="GPU capacity across all nodes">
          <div class="cluster-banner__icon cluster-banner__icon--ok"><${IconGpu} /></div>
          <div class="cluster-banner__body">
            <div class="cluster-banner__label">GPUs</div>
            <div class="cluster-banner__value">
              <span class=${'cluster-banner__num cluster-banner__num--' + tone}>${safeUsed}</span>
              <span class="cluster-banner__num-sep">/</span>
              <span class="cluster-banner__num cluster-banner__num--total">${safeGpus}</span>
            </div>
            <div class="cluster-banner__sub">
              <span class="cluster-banner__chip cluster-banner__chip--ok">${safeFree} free</span>
              <span class="cluster-banner__chip cluster-banner__chip--warn">${safeUsed} used</span>
            </div>
          </div>
        </div>

        <div class="cluster-banner__tile" data-testid="banner-util" title="100 × used / total GPUs">
          <div class=${'cluster-banner__icon cluster-banner__icon--' + tone}><${IconSpeed} /></div>
          <div class="cluster-banner__body">
            <div class="cluster-banner__label">Utilization</div>
            <div class="cluster-banner__value">
              <span class=${'cluster-banner__num cluster-banner__num--' + tone}>${safeUtil.toFixed(1)}<span class="cluster-banner__unit">%</span></span>
            </div>
            <div class="cluster-banner__bar" aria-hidden="true">
              <div class=${'cluster-banner__bar-fill cluster-banner__bar-fill--' + tone} style=${'width: ' + Math.min(100, Math.max(0, safeUtil)) + '%'}></div>
            </div>
          </div>
        </div>

        <div class="cluster-banner__tile" data-testid="banner-nodes" title="GPU-bearing nodes by allocation state">
          <div class="cluster-banner__icon cluster-banner__icon--neutral"><${IconNodes} /></div>
          <div class="cluster-banner__body">
            <div class="cluster-banner__label">GPU Nodes</div>
            <div class="cluster-banner__value">
              <span class="cluster-banner__num cluster-banner__num--total">${safeGpuNodes}</span>
            </div>
            <div class="cluster-banner__sub">
              <span class="cluster-banner__dot cluster-banner__dot--ok"></span>
              <span>${safeNFree} free</span>
              <span class="cluster-banner__dot cluster-banner__dot--warn"></span>
              <span>${safeNPartial} partial</span>
              <span class="cluster-banner__dot cluster-banner__dot--bad"></span>
              <span>${safeNFull} full</span>
            </div>
          </div>
        </div>
      `}

      ${!hasGpus && html`
        <div class="cluster-banner__tile" data-testid="banner-nodes" title="Total cluster nodes">
          <div class="cluster-banner__icon cluster-banner__icon--neutral"><${IconNodes} /></div>
          <div class="cluster-banner__body">
            <div class="cluster-banner__label">Nodes</div>
            <div class="cluster-banner__value">
              <span class="cluster-banner__num cluster-banner__num--total">${safeNodes > 0 ? safeNodes : '—'}</span>
            </div>
            <div class="cluster-banner__sub">no GPU-bearing nodes detected</div>
          </div>
        </div>
      `}

      <div class="cluster-banner__tile" data-testid="banner-k8s" title=${safeClusterName ? `Cluster: ${safeClusterName} (Kubernetes ${safeK8sVersion})` : 'Kubernetes server version'}>
        <div class="cluster-banner__icon cluster-banner__icon--accent"><${IconK8s} /></div>
        <div class="cluster-banner__body">
          <div class="cluster-banner__label">${safeClusterName ? 'Cluster' : 'Kubernetes'}</div>
          <div class="cluster-banner__value">
            <span class="cluster-banner__num cluster-banner__num--total cluster-banner__num--mono">${safeClusterName || safeK8sVersion}</span>
          </div>
          <div class="cluster-banner__sub">
            ${safeClusterName
              ? html`<span>${safeNodes} node${safeNodes === 1 ? '' : 's'} · k8s ${safeK8sVersion}</span>`
              : html`<span>${safeNodes} node${safeNodes === 1 ? '' : 's'} total</span>`}
          </div>
        </div>
      </div>
    </div>
  `;
}

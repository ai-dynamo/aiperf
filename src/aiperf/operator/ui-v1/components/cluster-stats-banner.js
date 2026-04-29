import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api, poll } from '../lib/api.js';

const POLL_MS = 30_000;

/**
 * Pick a CSS class for the utilization figure based on the same
 * thresholds gpu-report.sh uses (<50 green, <80 amber, ≥80 red).
 */
function utilClass(pct) {
  if (pct < 50) return 'cluster-stat-num cluster-stat-num--ok';
  if (pct < 80) return 'cluster-stat-num cluster-stat-num--warn';
  return 'cluster-stat-num cluster-stat-num--bad';
}

/**
 * Top-of-page banner showing cluster-wide GPU usage and Kubernetes
 * version. Polls /api/v1/cluster every 30 s. Renders nothing until
 * the first response lands so it never flashes a "0 GPUs" placeholder.
 *
 * Schema: ../../routers/jobs_models.py::ClusterResponse
 */
export function ClusterStatsBanner() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    const ac = new AbortController();
    poll(async () => {
      try {
        setInfo(await api.getCluster());
      } catch (_e) {
        // Best-effort: leave the prior value in place so a brief 503
        // during operator restart doesn't clear the banner.
      }
    }, POLL_MS, ac.signal);
    return () => ac.abort();
  }, []);

  if (!info) return null;

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
  } = info;

  // If the cluster has no GPUs (or pod-listing failed), show a slim
  // version-only banner rather than a row of zeros.
  const hasGpus = gpus > 0;

  return html`
    <div class="cluster-stats" role="status" data-testid="cluster-stats">
      ${hasGpus && html`
        <div class="cluster-stat-group" title="GPU capacity across all nodes">
          <span class="cluster-stat-label">GPUs</span>
          <span class="cluster-stat-num cluster-stat-num--warn">${used}</span>
          <span class="cluster-stat-sep">used</span>
          <span class="cluster-stat-num cluster-stat-num--ok">${free}</span>
          <span class="cluster-stat-sep">free</span>
          <span class="cluster-stat-num cluster-stat-num--total">${gpus}</span>
          <span class="cluster-stat-sep">total</span>
        </div>
        <div class="cluster-stat-group" title="100 × used / total">
          <span class="cluster-stat-label">Util</span>
          <span class=${utilClass(util)}>${util.toFixed(1)}%</span>
        </div>
        <div class="cluster-stat-group" title="GPU-bearing nodes by allocation state">
          <span class="cluster-stat-label">Nodes</span>
          <span class="cluster-stat-num cluster-stat-num--ok">${nFree}</span>
          <span class="cluster-stat-sep">free</span>
          <span class="cluster-stat-num cluster-stat-num--warn">${nPartial}</span>
          <span class="cluster-stat-sep">partial</span>
          <span class="cluster-stat-num cluster-stat-num--bad">${nFull}</span>
          <span class="cluster-stat-sep">full</span>
          <span class="cluster-stat-sep">/ ${gpuNodes} GPU</span>
        </div>
      `}
      ${!hasGpus && html`
        <div class="cluster-stat-group" title="Total cluster nodes">
          <span class="cluster-stat-label">Nodes</span>
          <span class="cluster-stat-num cluster-stat-num--total">${nodes}</span>
        </div>
      `}
      <div class="cluster-stat-group cluster-stat-group--right" title="Kubernetes server version">
        <span class="cluster-stat-label">k8s</span>
        <span class="cluster-stat-num cluster-stat-num--total">${k8sVersion}</span>
      </div>
    </div>
  `;
}

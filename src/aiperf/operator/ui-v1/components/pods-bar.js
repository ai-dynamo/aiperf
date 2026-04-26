import { html } from 'htm/preact';

/**
 * Dot color for a pod based on phase and ready state.
 * @param {{ phase: string, ready: boolean }} pod
 * @returns {string} CSS class
 */
function podDotClass(pod) {
  const phase = (pod.phase ?? '').toLowerCase();
  if (phase === 'failed' || phase === 'error') return 'pod-dot--failed';
  if (pod.ready) return 'pod-dot--ready';
  if (phase === 'running') return 'pod-dot--not-ready';
  return 'pod-dot--pending';
}

/**
 * Truncate a pod name for display, keeping the suffix.
 * @param {string} name
 * @param {number} maxLen
 * @returns {string}
 */
function truncatePodName(name, maxLen = 20) {
  if (name.length <= maxLen) return name;
  return '...' + name.slice(-(maxLen - 3));
}

/**
 * Horizontal pod status bar.
 * @param {{ pods: Array<{name: string, phase: string, ready: boolean, restarts: number}> }} props
 */
export function PodsBar({ pods }) {
  if (!pods || pods.length === 0) {
    return html`<div class="pods-bar pods-bar--empty">No pods</div>`;
  }

  const readyCount = pods.filter((p) => p.ready).length;
  const totalRestarts = pods.reduce((sum, p) => sum + (p.restarts ?? 0), 0);

  return html`
    <div class="pods-bar">
      <div class="pods-bar-dots">
        ${pods.map(
          (pod) => html`
            <span
              key=${pod.name}
              class=${'pod-dot ' + podDotClass(pod)}
              title=${pod.name + ' (' + (pod.phase ?? 'unknown') + ')'}
            />
          `,
        )}
      </div>
      <div class="pods-bar-names">
        ${pods.map(
          (pod) => html`
            <span
              key=${pod.name}
              class="pods-bar-name"
              title=${pod.name}
            >
              ${truncatePodName(pod.name)}
            </span>
          `,
        )}
      </div>
      <div class="pods-bar-summary">
        <span class="pods-bar-ready">${readyCount}/${pods.length} ready</span>
        ${totalRestarts > 0 && html`
          <span class="pods-bar-restarts">${totalRestarts} restart${totalRestarts !== 1 ? 's' : ''}</span>
        `}
      </div>
    </div>
  `;
}

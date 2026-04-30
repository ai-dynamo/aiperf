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
 * Defensive cap on per-pod rendering. Sweep jobs at very high concurrency can
 * spawn 200+ pods; at that scale individual dots collapse to pixel-wide
 * artifacts and hover stops being meaningful. Cap visible dots/names and
 * surface the overflow as an aggregate chip; the summary still reflects all
 * pods so the ready/restarts counts stay correct.
 */
const MAX_VISIBLE_PODS = 100;

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

  const overflowCount = Math.max(0, pods.length - MAX_VISIBLE_PODS);
  const visiblePods = overflowCount > 0 ? pods.slice(0, MAX_VISIBLE_PODS) : pods;

  return html`
    <div class="pods-bar">
      <div class="pods-bar-dots">
        ${visiblePods.map(
          (pod) => html`
            <span
              key=${pod.name}
              class=${'pod-dot ' + podDotClass(pod)}
              title=${pod.name + ' (' + (pod.phase ?? 'unknown') + ')'}
            />
          `,
        )}
        ${overflowCount > 0 && html`
          <span
            class="pod-dot-overflow"
            title=${'+' + overflowCount + ' more pods (showing first ' + MAX_VISIBLE_PODS + ')'}
            style="display:inline-flex;align-items:center;padding:0 6px;font-size:11px;color:var(--text-dim,#888);border:1px dashed currentColor;border-radius:8px;margin-left:4px"
          >
            +${overflowCount}
          </span>
        `}
      </div>
      <div class="pods-bar-names">
        ${visiblePods.map(
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
        ${overflowCount > 0 && html`
          <span class="pods-bar-name" style="opacity:0.7;font-style:italic">
            +${overflowCount} more
          </span>
        `}
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

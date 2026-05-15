import { html } from 'htm/preact';

const CONDITION_LABELS = {
  ConfigValid: 'Config',
  EndpointReachable: 'Endpoint',
  PreflightPassed: 'Preflight',
  ResourcesCreated: 'Resources',
  WorkersReady: 'Workers',
  BenchmarkRunning: 'Running',
  ResultsAvailable: 'Results',
};

/**
 * Determine badge CSS class based on condition status/reason.
 * @param {object} condition - K8s condition object
 * @returns {string} CSS class suffix
 */
function conditionClass(condition) {
  const status = (condition.status ?? '').toLowerCase();
  const reason = (condition.reason ?? '').toLowerCase();

  if (status === 'true') return 'condition-badge--true';
  if (reason.includes('progress') || reason.includes('waiting')) {
    return 'condition-badge--progress';
  }
  if (status === 'false' && reason.includes('failed')) {
    return 'condition-badge--false';
  }
  return 'condition-badge--unknown';
}

/**
 * Defensive cap on rendered condition badges. K8s conditions for AIPerfJob
 * top out at ~10 in normal operation; a malformed status block has no upper
 * bound, so cap rendering to keep DOM bounded and prevent runaway layouts.
 */
const MAX_VISIBLE_CONDITIONS = 50;

/**
 * Row of condition status badges.
 * @param {{ conditions: Array<{type: string, status: string, reason?: string, message?: string}> }} props
 */
export function Conditions({ conditions }) {
  if (!conditions || conditions.length === 0) {
    return html`<div class="conditions conditions--empty">No conditions</div>`;
  }

  const overflow = Math.max(0, conditions.length - MAX_VISIBLE_CONDITIONS);
  const visible = overflow > 0 ? conditions.slice(0, MAX_VISIBLE_CONDITIONS) : conditions;

  return html`
    <div
      class="conditions"
      role="list"
      aria-label="Conditions"
      style="display:flex;flex-wrap:wrap;gap:var(--space-1,4px);align-items:center"
    >
      ${visible.map((cond) => {
        const label = CONDITION_LABELS[cond.type] ?? cond.type;
        const cls = conditionClass(cond);
        const title = cond.message
          ? `${cond.type}: ${cond.message}`
          : cond.type;

        return html`
          <span
            key=${cond.type}
            class=${'condition-badge ' + cls}
            title=${title}
            role="listitem"
            style="word-break:break-word;max-width:100%"
          >
            ${label}
          </span>
        `;
      })}
      ${overflow > 0 && html`
        <span
          class="condition-badge condition-badge--unknown"
          role="listitem"
          title=${'+' + overflow + ' more conditions hidden (showing first ' + MAX_VISIBLE_CONDITIONS + ')'}
          style="word-break:break-word;max-width:100%"
        >
          +${overflow} more
        </span>
      `}
    </div>
  `;
}

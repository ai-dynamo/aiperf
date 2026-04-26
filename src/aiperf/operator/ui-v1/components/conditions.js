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
 * Row of condition status badges.
 * @param {{ conditions: Array<{type: string, status: string, reason?: string, message?: string}> }} props
 */
export function Conditions({ conditions }) {
  if (!conditions || conditions.length === 0) {
    return html`<div class="conditions conditions--empty">No conditions</div>`;
  }

  return html`
    <div class="conditions" role="list" aria-label="Conditions">
      ${conditions.map((cond) => {
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
          >
            ${label}
          </span>
        `;
      })}
    </div>
  `;
}

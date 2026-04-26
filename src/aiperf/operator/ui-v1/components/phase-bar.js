import { html } from 'htm/preact';
import { fmtInt } from '../lib/format.js';

/**
 * Phase progress bar row.
 * @param {{ phases: Array<{name: string, completed: number, total: number}> }} props
 */
export function PhaseBar({ phases }) {
  if (!phases || phases.length === 0) {
    return html`<div class="phase-bar phase-bar--empty">No phases</div>`;
  }

  return html`
    <div class="phase-bar">
      ${phases.map((phase) => {
        const pct =
          phase.total > 0 ? Math.round((phase.completed / phase.total) * 100) : 0;
        const done = phase.completed >= phase.total && phase.total > 0;
        const active = !done && phase.completed > 0;
        const statusClass = done
          ? 'phase-bar-item--done'
          : active
          ? 'phase-bar-item--active'
          : 'phase-bar-item--pending';

        return html`
          <div key=${phase.name} class=${'phase-bar-item ' + statusClass}>
            <div class="phase-bar-header">
              <span class="phase-bar-name">${phase.name}</span>
              <span class="phase-bar-fraction">
                ${fmtInt(phase.completed)}/${fmtInt(phase.total)}
              </span>
            </div>
            <div class="phase-bar-track">
              <div
                class="phase-bar-fill"
                style=${'width: ' + pct + '%'}
                role="progressbar"
                aria-valuenow=${pct}
                aria-valuemin="0"
                aria-valuemax="100"
              />
            </div>
          </div>
        `;
      })}
    </div>
  `;
}

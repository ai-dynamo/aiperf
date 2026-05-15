import { html } from 'htm/preact';
import { useEffect } from 'preact/hooks';
import { DiagnosticsPanel } from './diagnostics-panel.js';

/**
 * Slide-in drawer wrapping ``DiagnosticsPanel``. The panel itself is
 * unchanged — it manages its own ``?diag=`` URL state and pops a
 * ``popstate`` event the parent (``pages/job-detail.js``) listens to.
 * The drawer only adds chrome: positioning, a backdrop, ``Esc`` /
 * outside-click close, and width transitions.
 *
 * Inline styles only — class names ``diagnostics-drawer*`` are
 * reserved for the Wave 2 CSS pass which will refine breakpoints
 * (mobile full-screen via ``@media``). Today ``width: 420px;
 * max-width: 100vw`` keeps narrow viewports clamped to screen width.
 *
 * @example
 *   <DiagnosticsDrawer
 *     open=${diagnosticsOpen}
 *     onClose=${() => setDiagnosticsOpen(false)}
 *     ns=${namespace}
 *     name=${name}
 *     conditions=${conditions}
 *     pods=${pods}
 *     mode=${mode}
 *     archived=${archived}
 *     eventCount=${eventCount}
 *     logSeverityCounts=${logSeverityCounts}
 *     conditionWarnCount=${conditionWarnCount}
 *     podCrashCount=${podCrashCount}
 *   />
 *
 * @param {object} props
 * @param {boolean} props.open - When false the drawer (and the wrapped
 *   panel) is unmounted entirely; mounting drives the panel's URL
 *   writeback.
 * @param {() => void} props.onClose - Invoked on Esc or backdrop click.
 *   Every other prop is forwarded verbatim to ``DiagnosticsPanel``.
 */
export function DiagnosticsDrawer({ open, onClose, ...panelProps }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return html`
    <div
      class="diagnostics-drawer__backdrop"
      data-testid="diagnostics-drawer-backdrop"
      onClick=${onClose}
      style=${BACKDROP_STYLE}
    ></div>
    <aside
      class="diagnostics-drawer"
      data-testid="diagnostics-drawer"
      role="dialog"
      aria-label="Diagnostics"
      style=${ASIDE_STYLE}
      onClick=${(e) => e.stopPropagation()}
    >
      <header class="diagnostics-drawer__head" style=${HEAD_STYLE}>
        <span class="diagnostics-drawer__title" style=${TITLE_STYLE}>Diagnostics</span>
        <button
          class="diagnostics-drawer__close"
          aria-label="Close diagnostics"
          data-testid="diagnostics-drawer-close"
          onClick=${onClose}
          style=${CLOSE_STYLE}
        >×</button>
      </header>
      <div class="diagnostics-drawer__body" style=${BODY_STYLE}>
        <${DiagnosticsPanel} ...${panelProps} />
      </div>
    </aside>
  `;
}

const BACKDROP_STYLE =
  'position: fixed;'
  + ' inset: 0;'
  + ' background: rgba(0,0,0,0.45);'
  + ' z-index: 100;';

const ASIDE_STYLE =
  'position: fixed;'
  + ' top: 0; right: 0; bottom: 0;'
  + ' width: 420px;'
  + ' max-width: 100vw;'
  + ' background: var(--bg-card);'
  + ' border-left: 1px solid var(--border);'
  + ' z-index: 101;'
  + ' display: flex;'
  + ' flex-direction: column;'
  + ' box-shadow: -8px 0 24px rgba(0,0,0,0.35);';

const HEAD_STYLE =
  'display: flex;'
  + ' align-items: center;'
  + ' justify-content: space-between;'
  + ' padding: 12px 14px;'
  + ' border-bottom: 1px solid var(--border);'
  + ' flex-shrink: 0;';

const TITLE_STYLE = 'font-weight: 600; font-size: 13px;';

const CLOSE_STYLE =
  'background: transparent;'
  + ' color: var(--muted);'
  + ' border: none;'
  + ' font-size: 22px;'
  + ' line-height: 1;'
  + ' cursor: pointer;'
  + ' padding: 0 6px;';

const BODY_STYLE = 'flex: 1; overflow-y: auto;';

import { html } from 'htm/preact';

/**
 * Typed key/value pair used in the job-detail identity strip and the
 * right rail. Renders the key in tiny uppercase Inter and the value in
 * JetBrains Mono. Optional ``accent`` makes the value carry the
 * NVIDIA-green accent (only used for the ``phase`` pair today).
 *
 * Inline styles only — class names ``kv``, ``kv__k``, ``kv__v``,
 * ``kv--accent`` are reserved for the Wave 2 CSS rewrite but have no
 * rules attached today, so the inline ``style=…`` attributes must
 * achieve the visual outcome on their own.
 *
 * @example
 *   <KV k="phase" v="profiling" accent />
 *   <KV k="ns" v="acme-bench" />
 *
 * @param {object} props
 * @param {string} props.k - Short uppercase-shaped key (e.g. "phase", "ns").
 * @param {string|number} props.v - Value rendered in monospace.
 * @param {boolean} [props.accent] - Tints value with --accent at weight 600.
 * @param {string} [props.testId] - data-testid override (defaults to `kv-${k}`).
 */
export function KV({ k, v, accent, testId }) {
  return html`
    <span
      class=${'kv' + (accent ? ' kv--accent' : '')}
      data-testid=${testId ?? `kv-${k}`}
      style=${KV_STYLE}
    >
      <span class="kv__k" style=${KV_KEY_STYLE}>${k}</span>
      <span class="kv__v" style=${accent ? KV_VALUE_ACCENT_STYLE : KV_VALUE_STYLE}>${v}</span>
    </span>
  `;
}

/**
 * Dim "·" separator placed between adjacent ``<KV>`` pairs in the
 * identity strip. Pulled out into its own component because the strip
 * uses it repeatedly and Wave 2's CSS pass will style ``.kv-sep``.
 */
export function KvSep() {
  return html`<span class="kv-sep" style=${KV_SEP_STYLE}>·</span>`;
}

const KV_STYLE = 'display: inline-flex; align-items: baseline; gap: 6px; font-size: 11.5px;';
const KV_KEY_STYLE =
  'color: var(--dim);'
  + ' font-size: 9.5px;'
  + ' letter-spacing: 0.08em;'
  + ' text-transform: uppercase;'
  + ' font-weight: 500;';
const KV_VALUE_STYLE = 'font-family: var(--font-mono); color: var(--sub);';
const KV_VALUE_ACCENT_STYLE =
  'font-family: var(--font-mono); color: var(--accent); font-weight: 600;';
const KV_SEP_STYLE = 'color: var(--hairline);';

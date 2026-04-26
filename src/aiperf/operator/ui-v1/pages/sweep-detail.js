import { html } from 'htm/preact';

export function SweepDetail({ namespace, name }) {
  return html`<div data-testid="page-sweep-detail">${namespace}/${name} (stub)</div>`;
}

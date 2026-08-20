// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Launch — create a new AIPerfJob from the UI.
 *
 * Pick a starting template (or paste your own YAML), edit in the textarea,
 * and copy it for `aiperf kube apply` or `kubectl apply`. Browser-side job
 * creation stays disabled because the static SPA has no safe bearer-token
 * delivery path for protected mutating routes.
 *
 * YAML is parsed by the locally-vendored, standards-compliant parser, then
 * checked before the manifest can reach a protected mutating API route.
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { loadAll } from 'js-yaml';
import { api, isTokenRequiredError, setSessionToken } from '../lib/api.js';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';
import { Spinner } from '../components/spinner.js';
import { TokenModal } from '../components/token-modal.js';
import { YamlEditor } from '../components/yaml-editor.js';

/* ───────────────────────── templates ───────────────────────── */

function dateStamp() {
  return new Date().toISOString().slice(0, 10).replace(/-/g, '');
}

function buildTemplates() {
  const stamp = dateStamp();
  return [
    {
      id: 'llama3-70b-throughput',
      name: 'Llama 3 · 70B throughput sweep',
      desc: 'Stress a single TRT-LLM endpoint with high concurrency; ideal starting point for a capacity sweep.',
      yaml: `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: llama3-70b-throughput-${stamp}
  namespace: default
spec:
  benchmark:
    models:
      - meta-llama/Llama-3-70B
    endpoint:
      urls:
        - "http://trtllm.default.svc:8000"
      type: chat
      streaming: true
      path: /v1/chat/completions
    datasets:
      main:
        type: synthetic
        entries: 8000
        isl: { mean: 1024, stddev: 0 }
        osl: { mean: 256, stddev: 0 }
    warmup:
      type: concurrency
      concurrency: 64
      requests: 256
    profiling:
      type: concurrency
      concurrency: 256
      requests: 8000
    slos:
      request_latency: 500
      time_to_first_token: 300
      inter_token_latency: 30
  podTemplate: {}
`,
    },
    {
      id: 'mistral-burst',
      name: 'Mistral 7B · burst test',
      desc: 'Short, bursty load suitable for smoke-testing a freshly-deployed endpoint.',
      yaml: `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: mistral-7b-smoke-${stamp}
  namespace: default
spec:
  benchmark:
    models:
      - mistralai/Mistral-7B-Instruct
    endpoint:
      urls:
        - "http://vllm.default.svc:8000"
      type: chat
      streaming: true
    datasets:
      main:
        type: synthetic
        entries: 1000
        isl: { mean: 256, stddev: 0 }
        osl: { mean: 128, stddev: 0 }
    warmup:
      type: concurrency
      concurrency: 16
      requests: 32
    profiling:
      type: concurrency
      concurrency: 128
      requests: 1000
  podTemplate: {}
`,
    },
    {
      id: 'minimal',
      name: 'Minimal skeleton',
      desc: 'Bare-bones AIPerfJob. Fill in your own models, endpoint, and dataset.',
      yaml: `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: my-benchmark
  namespace: default
spec:
  benchmark:
    models:
      - <model-name>
    endpoint:
      urls:
        - "http://<endpoint-host>:8000"
      type: chat
      streaming: true
    datasets:
      main:
        type: synthetic
        entries: 1000
        isl: { mean: 256, stddev: 0 }
        osl: { mean: 128, stddev: 0 }
    profiling:
      type: concurrency
      concurrency: 32
      requests: 1000
  podTemplate: {}
`,
    },
  ];
}

/* ───────────────────────── tiny YAML parser ───────────────────────── */

const DANGEROUS_YAML_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function assertSafeYamlKey(key, lineNo = null) {
  if (!DANGEROUS_YAML_KEYS.has(key)) return;
  const location = lineNo === null ? 'Manifest' : `Line ${lineNo + 1}`;
  throw new Error(`${location}: key '${key}' is not allowed in launch YAML.`);
}

function sanitizeParsedYaml(value, path = 'manifest') {
  if (Array.isArray(value)) {
    return value.map((item, index) => sanitizeParsedYaml(item, `${path}[${index}]`));
  }
  if (!value || typeof value !== 'object') return value;
  if (Object.getPrototypeOf(value) !== Object.prototype) {
    throw new Error(`${path}: only plain YAML mappings and sequences are allowed.`);
  }

  const out = {};
  for (const [key, child] of Object.entries(value)) {
    assertSafeYamlKey(key);
    out[key] = sanitizeParsedYaml(child, `${path}.${key}`);
  }
  return out;
}

function parseYaml(text) {
  const documents = loadAll(text);
  if (documents.length !== 1) {
    throw new Error('Launch YAML must contain exactly one document.');
  }
  return sanitizeParsedYaml(documents[0]);
}

function validateManifest(manifest) {
  if (!manifest || Object.keys(manifest).length === 0) {
    throw new Error('Manifest is empty; paste an AIPerfJob YAML manifest.');
  }
  if (manifest.kind !== 'AIPerfJob' && manifest.kind !== 'AIPerfSweep') {
    throw new Error(`kind must be AIPerfJob or AIPerfSweep, got ${manifest.kind ?? 'missing'}.`);
  }

  const name = manifest?.metadata?.name;
  if (typeof name !== 'string' || name.trim() === '') {
    throw new Error('metadata.name is required.');
  }
  if (!/^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$/.test(name) || name.length > 253) {
    throw new Error('metadata.name must be a valid Kubernetes DNS subdomain.');
  }

  const namespace = manifest?.metadata?.namespace;
  if (typeof namespace !== 'string' || namespace.trim() === '') {
    throw new Error('metadata.namespace is required.');
  }
  if (!/^[a-z0-9]([a-z0-9-]*[a-z0-9])?$/.test(namespace) || namespace.length > 63) {
    throw new Error('metadata.namespace must be a valid Kubernetes namespace name.');
  }
}

function parseLaunchManifest(text) {
  const manifest = parseYaml(text);
  validateManifest(manifest);
  return manifest;
}

// Peek at current YAML to derive a live, non-editable view of the target
// namespace / name / kind without committing to a full POST. Swallow parse
// errors here — the dedicated parse-error banner handles user-visible feedback.
function peekManifest(text) {
  try {
    const m = parseLaunchManifest(text);
    return {
      namespace: m.metadata.namespace,
      name: m.metadata.name,
      kind: m.kind,
      parseError: null,
    };
  } catch (e) {
    return { namespace: null, name: null, kind: null, parseError: e.message };
  }
}

/* ────────────────────────────── view ─────────────────────────── */

export function Launch() {
  // Templates are date-stamped at mount time so each visit gets a fresh suffix.
  const [templates] = useState(() => buildTemplates());
  const [templateId, setTemplateId] = useState(templates[0].id);
  const [yaml, setYaml] = useState(() => templates[0].yaml);
  const [state, setState] = useState({ kind: 'idle' });
  const [prefillFrom, setPrefillFrom] = useState(null);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const pendingActionRef = useRef(null);

  // Consume a sessionStorage handoff from a future Re-launch button. One-shot:
  // we clear it immediately so refreshing /launch doesn't keep re-prefilling.
  // The handoff payload shape: { yaml, sourceNs, sourceName, at }.
  useEffect(() => {
    let raw;
    try { raw = sessionStorage.getItem('aiperf.launch.prefill'); }
    catch (_e) { return; }
    if (!raw) return;
    try { sessionStorage.removeItem('aiperf.launch.prefill'); } catch (_e) { /* ignore */ }
    let payload;
    try { payload = JSON.parse(raw); } catch (_e) { return; }
    if (!payload || typeof payload.yaml !== 'string') return;
    if (!payload.at || Date.now() - payload.at > 60000) return;
    setYaml(payload.yaml);
    setTemplateId(null);
    setPrefillFrom({ ns: payload.sourceNs ?? '?', name: payload.sourceName ?? '?' });
  }, []);

  function pickTemplate(id) {
    const t = templates.find((tt) => tt.id === id);
    if (!t) return;
    setTemplateId(id);
    setYaml(t.yaml);
    setState({ kind: 'idle' });
    setPrefillFrom(null);
  }

  const peek = peekManifest(yaml);
  const canSubmit = state.kind !== 'submitting'
    && state.kind !== 'ok'
    && !peek.parseError;
  const submitGuardRef = useRef({ canSubmit, yaml });
  submitGuardRef.current = { canSubmit, yaml };

  function promptToken(retry) {
    pendingActionRef.current = retry;
    setShowTokenModal(true);
  }

  function onTokenConfirm(token) {
    setSessionToken(token);
    setShowTokenModal(false);
    const retry = pendingActionRef.current;
    pendingActionRef.current = null;
    if (retry) retry();
  }

  function onTokenCancel() {
    pendingActionRef.current = null;
    setShowTokenModal(false);
  }

  async function launch() {
    const guard = submitGuardRef.current;
    if (!guard.canSubmit) return;
    const yaml = guard.yaml;

    let manifest;
    try {
      manifest = parseLaunchManifest(yaml);
    } catch (e) {
      setState({ kind: 'err', msg: e.message, stage: 'parse' });
      return;
    }
    submitGuardRef.current = { canSubmit: false, yaml };
    setState({ kind: 'submitting' });
    try {
      const r = manifest.kind === 'AIPerfSweep'
        ? await api.createSweep(manifest)
        : await api.createJob(manifest);
      setState({ kind: 'ok', namespace: r.namespace, name: r.name, sweepKind: manifest.kind === 'AIPerfSweep' });
    } catch (e) {
      if (isTokenRequiredError(e)) {
        submitGuardRef.current = { canSubmit: true, yaml };
        setState({ kind: 'idle' });
        promptToken(() => launch());
        return;
      }
      let msg = e.message;
      let status = null;
      const m = /^API (\d+):\s*/.exec(msg);
      if (m) status = parseInt(m[1], 10);
      try {
        const body = JSON.parse(msg.replace(/^API \d+:\s*/, ''));
        msg = body?.detail ?? body?.message ?? msg;
      } catch (_) { /* leave as-is */ }
      setState({ kind: 'err', msg, stage: 'submit', status });
    }
  }

  function copyYaml() {
    navigator.clipboard?.writeText(yaml).catch(() => {});
  }

  function onYamlKeydown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (state.kind !== 'submitting') launch();
    }
  }

  function viewRun() {
    if (state.kind !== 'ok') return;
    if (state.sweepKind) {
      navigate(`/sweeps/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`);
    } else {
      navigate(`/jobs/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`);
    }
  }

  const activeTemplate = templates.find((t) => t.id === templateId);

  // Style helpers — keep palette colors in inline styles to match other ui-v1
  // pages that don't lean on dedicated stylesheet classes.
  const templateTabBase = 'padding: var(--space-1) 0; border: 0; border-bottom: 2px solid; border-radius: 0; background: transparent; font-size: var(--font-size-sm); cursor: pointer; font-family: inherit;';
  const templateTabIdle = ` color: ${palette.subtext1}; border-color: transparent;`;
  const templateTabActive = ` color: ${palette.text}; border-color: ${palette.accent};`;

  const targetRowStyle = `display: block; width: 100%; box-sizing: border-box; margin-bottom: var(--space-3); padding: var(--space-2) var(--space-3); background: var(--bg-tile); border: 1px solid ${palette.surface0}; border-radius: var(--radius-md); color: ${palette.subtext1}; font-family: var(--font-mono); font-size: var(--font-size-sm);`;



  return html`
    <div class="launch-page" data-testid="page-launch">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Prepare a new run</span>
      </div>

      <div class="card">
        <div class="card-title" style="margin-bottom: var(--space-3)">Template</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--space-4); margin-bottom: var(--space-4)" aria-label="Templates">
          ${templates.map((t) => html`
            <button type="button"
              key=${t.id}
              style=${templateTabBase + (t.id === templateId ? templateTabActive : templateTabIdle)}
              onclick=${() => pickTemplate(t.id)}
              data-testid=${'launch-template-' + t.id}
              aria-pressed=${t.id === templateId}
              title=${t.desc}
            >${t.name}</button>
          `)}
        </div>

        ${prefillFrom && html`
          <div
            data-testid="launch-prefill-notice"
            style=${`margin-bottom: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.surface1}; background: transparent; color: ${palette.subtext1}; font-size: var(--font-size-sm)`}
          >
            Pre-filled from run: <strong style=${`color: ${palette.text}; font-family: var(--font-mono)`}>${prefillFrom.ns}/${prefillFrom.name}</strong>
          </div>
        `}

        <div
          data-testid="launch-target"
          style=${targetRowStyle}
        >${`${peek.namespace ?? '—'} / ${peek.name ?? '—'}  ·  kind: ${peek.kind ?? '—'}${activeTemplate ? `  ·  template: ${activeTemplate.name}` : ''}`}</div>

        <${YamlEditor}
          value=${yaml}
          onInput=${(e) => { setYaml(e.target.value); if (state.kind !== 'submitting') setState({ kind: 'idle' }); }}
          onKeydown=${onYamlKeydown}
          testid="launch-editor"
        />

        ${state.kind === 'ok' && html`
          <div
            data-testid="launch-success"
            style=${`margin-top: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.green}; color: ${palette.subtext1}; display: flex; align-items: center; gap: var(--space-3);`}
          >
            <span>Created <strong style=${`color: ${palette.text}; font-family: var(--font-mono)`}>${state.namespace}/${state.name}</strong></span>
            <a
              class="btn btn--ghost"
              data-testid="launch-view-run"
              href=${state.sweepKind
                ? `#/sweeps/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`
                : `#/jobs/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`}
              onclick=${(e) => { e.preventDefault(); viewRun(); }}
            >View run</a>
          </div>
        `}

        ${state.kind !== 'ok' && peek.parseError && html`
          <div
            data-testid="launch-parse-err"
            style=${`margin-top: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.peach}; color: ${palette.peach}; font-family: var(--font-mono); font-size: var(--font-size-sm);`}
          >YAML · ${peek.parseError}</div>
        `}

        ${state.kind === 'err' && html`
          <div
            data-testid="launch-err"
            style=${`margin-top: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.red}; color: ${palette.red}; font-size: var(--font-size-sm);`}
          >
            <strong style=${`color: ${palette.red}; margin-right: var(--space-2)`}>Error</strong>
            ${state.stage === 'parse'
              ? `YAML: ${state.msg}`
              : (state.status ? `HTTP ${state.status}: ${state.msg}` : state.msg)}
          </div>
        `}

        <div style="display: flex; gap: var(--space-2); justify-content: flex-end; margin-top: var(--space-4)">
          <button type="button"
            class="btn btn--ghost"
            onclick=${copyYaml}
            data-testid="launch-copy"
            title="Copy YAML to clipboard"
          >Copy</button>
          <button type="button"
            class="btn btn--primary"
            disabled=${!canSubmit}
            onclick=${launch}
            data-testid="launch-submit"
            title="Create the AIPerfJob"
          >${state.kind === 'submitting'
              ? html`<span style="display: inline-flex; align-items: center; gap: var(--space-2)"><${Spinner} size=${12} thickness=${1.5} color="var(--bg)" />Launching…</span>`
              : state.kind === 'ok' ? 'Launched' : 'Launch'}</button>
        </div>
      </div>
    </div>

    ${showTokenModal && html`<${TokenModal} onConfirm=${onTokenConfirm} onCancel=${onTokenCancel} />`}
  `;
}

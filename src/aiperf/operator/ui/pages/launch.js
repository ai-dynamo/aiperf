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
 * Hand-rolled YAML parser is used to keep the UI build-step-free; it handles
 * the AIPerfJob shape (mappings, sequences, scalars, inline flow maps for
 * ``isl``/``osl``).
 */

import { html } from 'htm/preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { api, DASHBOARD_MUTATIONS_DISABLED_MESSAGE, DASHBOARD_MUTATIONS_ENABLED } from '../lib/api.js';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';
import { Spinner } from '../components/spinner.js';

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

function setYamlKey(target, key, value, lineNo) {
  assertSafeYamlKey(key, lineNo);
  target[key] = value;
}

function sanitizeParsedYaml(value, path = 'manifest') {
  if (Array.isArray(value)) {
    return value.map((item, index) => sanitizeParsedYaml(item, `${path}[${index}]`));
  }
  if (!value || typeof value !== 'object') return value;
  if (Object.getPrototypeOf(value) !== Object.prototype) {
    throw new Error(`${path}: object prototype was modified by YAML keys.`);
  }

  const out = {};
  for (const [key, child] of Object.entries(value)) {
    assertSafeYamlKey(key);
    out[key] = sanitizeParsedYaml(child, `${path}.${key}`);
  }
  return out;
}

/** Very small YAML-ish parser sufficient for AIPerfJob manifests. Real YAML
 *  would need a library; for the expected shape (mappings, sequences, scalars,
 *  inline flow maps for isl/osl) this hand-rolled version is enough and keeps
 *  the UI build-step-free. If parsing fails, we raise a structured error with
 *  the offending line so the user can fix it.
 */
function parseYaml(text) {
  const lines = text.split('\n');
  const root = {};
  // Stack of (container, indent) frames. Sequences push arrays; mappings push objects.
  const stack = [{ val: root, indent: -1, key: null }];

  const stripComment = (s) => {
    let inStr = null;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (inStr) { if (c === inStr) inStr = null; continue; }
      if (c === '"' || c === "'") { inStr = c; continue; }
      if (c === '#') return s.slice(0, i);
    }
    return s;
  };

  const parseScalar = (s) => {
    s = s.trim();
    if (s === '' || s === '~' || s.toLowerCase() === 'null') return null;
    if (s === 'true')  return true;
    if (s === 'false') return false;
    if (/^-?\d+$/.test(s)) return parseInt(s, 10);
    if (/^-?\d+\.\d+$/.test(s)) return parseFloat(s);
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
      return s.slice(1, -1);
    }
    if (s.startsWith('{') && s.endsWith('}')) {
      // Inline flow map: { k1: v1, k2: v2 }
      const inner = s.slice(1, -1).trim();
      const out = {};
      if (inner) {
        for (const pair of inner.split(',')) {
          const ci = pair.indexOf(':');
          if (ci < 0) continue;
          const k = pair.slice(0, ci).trim();
          const v = pair.slice(ci + 1).trim();
          setYamlKey(out, k, parseScalar(v), null);
        }
      }
      return out;
    }
    if (s.startsWith('[') && s.endsWith(']')) {
      const inner = s.slice(1, -1).trim();
      return inner ? inner.split(',').map(p => parseScalar(p.trim())) : [];
    }
    return s;
  };

  for (let lineNo = 0; lineNo < lines.length; lineNo++) {
    let raw = stripComment(lines[lineNo]).replace(/\s+$/, '');
    if (raw.trim() === '') continue;

    const indent = raw.match(/^ */)[0].length;
    const trimmed = raw.slice(indent);

    // Pop frames whose indent is ≥ current
    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) stack.pop();
    const top = stack[stack.length - 1];

    if (trimmed.startsWith('- ') || trimmed === '-') {
      // Sequence item. Parent must be a list; auto-promote if needed.
      if (!Array.isArray(top.val)) {
        throw new Error(`Line ${lineNo + 1}: unexpected '-' — parent is not a sequence.`);
      }
      const rest = trimmed.slice(2).trimStart();
      if (rest === '') {
        const child = {};
        top.val.push(child);
        stack.push({ val: child, indent });
      } else if (
        rest.includes(':') &&
        !rest.match(/^["'[{]/) &&
        !rest.match(/^[a-zA-Z][a-zA-Z0-9+.\-]*:\/\//)
      ) {
        const child = {};
        top.val.push(child);
        const [k, ...vparts] = rest.split(':');
        const itemKey = k.trim();
        const v = vparts.join(':').trim();
        const itemValue = v === '' ? {} : parseScalar(v);
        setYamlKey(child, itemKey, itemValue, lineNo);
        if (v === '') stack.push({ val: itemValue, indent: indent + 2 });
        else stack.push({ val: child, indent });
      } else {
        top.val.push(parseScalar(rest));
      }
      continue;
    }

    const ci = trimmed.indexOf(':');
    if (ci < 0) {
      throw new Error(`Line ${lineNo + 1}: expected 'key: value' or sequence item.`);
    }
    const key = trimmed.slice(0, ci).trim();
    const rest = trimmed.slice(ci + 1).trim();

    if (rest === '') {
      // Peek next non-empty line to decide object vs sequence.
      let child = {};
      for (let la = lineNo + 1; la < lines.length; la++) {
        const laRaw = stripComment(lines[la]).replace(/\s+$/, '');
        if (laRaw.trim() === '') continue;
        const laIndent = laRaw.match(/^ */)[0].length;
        if (laIndent <= indent) break;
        if (laRaw.slice(laIndent).startsWith('- ') || laRaw.slice(laIndent) === '-') child = [];
        break;
      }
      if (Array.isArray(top.val)) {
        throw new Error(`Line ${lineNo + 1}: key '${key}' at sequence position (should be inside an object).`);
      }
      setYamlKey(top.val, key, child, lineNo);
      stack.push({ val: child, indent });
    } else {
      if (Array.isArray(top.val)) {
        throw new Error(`Line ${lineNo + 1}: key '${key}' at sequence position.`);
      }
      setYamlKey(top.val, key, parseScalar(rest), lineNo);
    }
  }

  return sanitizeParsedYaml(root);
}

function validateManifest(manifest) {
  if (!manifest || Object.keys(manifest).length === 0) {
    throw new Error('Manifest is empty; paste an AIPerfJob YAML manifest.');
  }
  if (manifest.kind !== 'AIPerfJob') {
    throw new Error(`kind must be AIPerfJob, got ${manifest.kind ?? 'missing'}.`);
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
  const canSubmit = DASHBOARD_MUTATIONS_ENABLED
    && state.kind !== 'submitting'
    && state.kind !== 'ok'
    && !peek.parseError;
  const submitGuardRef = useRef({ canSubmit, yaml });
  submitGuardRef.current = { canSubmit, yaml };

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
      const r = await api.createJob(manifest);
      setState({ kind: 'ok', namespace: r.namespace, name: r.name });
    } catch (e) {
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
    navigate(`/jobs/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`);
  }

  const activeTemplate = templates.find((t) => t.id === templateId);

  // Style helpers — keep palette colors in inline styles to match other ui-v1
  // pages that don't lean on dedicated stylesheet classes.
  const pillBase = 'padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid; font-size: var(--font-size-sm); cursor: pointer; font-family: inherit;';
  const pillIdle = ` background: transparent; color: ${palette.subtext1}; border-color: ${palette.surface0};`;
  const pillActive = ` background: ${palette.accentDim}; color: ${palette.text}; border-color: ${palette.accent};`;

  const targetRowStyle = `display: block; width: 100%; box-sizing: border-box; margin-bottom: var(--space-3); padding: var(--space-2) var(--space-3); background: var(--bg-tile); border: 1px solid ${palette.surface0}; border-radius: var(--radius-md); color: ${palette.subtext1}; font-family: var(--font-mono); font-size: var(--font-size-sm);`;

  const textareaStyle = `font-family: var(--font-mono); font-size: var(--font-size-sm); width: 100%; box-sizing: border-box; min-height: 480px; background: var(--bg-tile); border: 1px solid ${palette.surface0}; border-radius: var(--radius-md); padding: var(--space-3); color: ${palette.text}; resize: vertical; line-height: 1.5;`;

  return html`
    <div class="launch-page" data-testid="page-launch">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Prepare a new run</span>
        <span class="text-dim" style="margin-left: var(--space-3); font-size: var(--font-size-xs); font-family: var(--font-mono)">read-only dashboard</span>
      </div>

      <div class="card">
        <div class="card-title" style="margin-bottom: var(--space-3)">Template</div>
        <div style="display: flex; flex-wrap: wrap; gap: var(--space-2); margin-bottom: var(--space-4)" aria-label="Templates">
          ${templates.map((t) => html`
            <button type="button"
              key=${t.id}
              style=${pillBase + (t.id === templateId ? pillActive : pillIdle)}
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
            style=${`margin-bottom: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.mauve}; background: ${palette.surface0}; color: ${palette.subtext1}; font-size: var(--font-size-sm)`}
          >
            Pre-filled from run: <strong style=${`color: ${palette.text}; font-family: var(--font-mono)`}>${prefillFrom.ns}/${prefillFrom.name}</strong>
          </div>
        `}

        <div
          data-testid="launch-target"
          style=${targetRowStyle}
        >${`${peek.namespace ?? '—'} / ${peek.name ?? '—'}  ·  kind: ${peek.kind ?? '—'}${activeTemplate ? `  ·  template: ${activeTemplate.name}` : ''}`}</div>

        ${!DASHBOARD_MUTATIONS_ENABLED && html`
          <div
            data-testid="launch-readonly-notice"
            style=${`margin-bottom: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.yellow}; color: ${palette.yellow}; background: ${palette.yellow}11; font-size: var(--font-size-sm);`}
          >${DASHBOARD_MUTATIONS_DISABLED_MESSAGE}</div>
        `}

        <textarea
          style=${textareaStyle}
          value=${yaml}
          oninput=${(e) => { setYaml(e.target.value); if (state.kind !== 'submitting') setState({ kind: 'idle' }); }}
          onkeydown=${onYamlKeydown}
          spellcheck="false"
          wrap="off"
          data-testid="launch-editor"
        ></textarea>

        ${state.kind === 'ok' && html`
          <div
            data-testid="launch-success"
            style=${`margin-top: var(--space-3); padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid ${palette.green}; color: ${palette.subtext1}; display: flex; align-items: center; gap: var(--space-3);`}
          >
            <span>Created <strong style=${`color: ${palette.text}; font-family: var(--font-mono)`}>${state.namespace}/${state.name}</strong></span>
            <a
              class="btn btn--ghost"
              data-testid="launch-view-run"
              href=${`#/jobs/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`}
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
            title=${!DASHBOARD_MUTATIONS_ENABLED ? DASHBOARD_MUTATIONS_DISABLED_MESSAGE : 'Create the AIPerfJob'}
          >${state.kind === 'submitting'
              ? html`<span style="display: inline-flex; align-items: center; gap: var(--space-2)"><${Spinner} size=${12} thickness=${1.5} color="var(--bg)" />Launching…</span>`
              : !DASHBOARD_MUTATIONS_ENABLED ? 'Launch disabled' : state.kind === 'ok' ? 'Launched' : 'Launch'}</button>
        </div>
      </div>
    </div>
  `;
}

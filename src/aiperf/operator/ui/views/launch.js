// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LAUNCH — create a new AIPerfJob from the UI.
 *
 * Left column: template picker. Pick a starting point, see it populate the
 * editor. Right column: raw YAML editor (monospace textarea — nothing fancier
 * since we want to stay build-step-free; no Monaco / no CodeMirror).
 *
 * SUBMIT path: parse the YAML → POST /api/v1/jobs → navigate to the new run's
 * workbench page. Errors are surfaced inline (validation, name collision,
 * RBAC).
 */

import { html } from 'htm/preact';
import { useEffect, useState } from 'preact/hooks';
import { api } from '../lib/api.js';
import { navigate } from '../lib/router.js';

/* ───────────────────────── templates ───────────────────────── */

const TEMPLATES = [
  {
    id: 'llama3-70b-throughput',
    name: 'Llama 3 · 70B throughput sweep',
    desc: 'Stress a single TRT-LLM endpoint with high concurrency; ideal starting point for a capacity sweep.',
    yaml: `apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: llama3-70b-throughput-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}
  namespace: default
spec:
  benchmark:
    models:
    - meta-llama/Llama-3-70B
    endpoint:
      urls:
      - http://trtllm.default.svc:8000
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
  name: mistral-7b-smoke-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}
  namespace: default
spec:
  benchmark:
    models:
    - mistralai/Mistral-7B-Instruct
    endpoint:
      urls:
      - http://vllm.default.svc:8000
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
      - http://<endpoint-host>:8000
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

/* ───────────────────────── tiny YAML parser ───────────────────────── */

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
          out[k] = parseScalar(v);
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
      } else if (rest.includes(':') && !rest.match(/^["'[{]/)) {
        const child = {};
        top.val.push(child);
        const [k, ...vparts] = rest.split(':');
        const v = vparts.join(':').trim();
        child[k.trim()] = v === '' ? {} : parseScalar(v);
        if (v === '') stack.push({ val: child[k.trim()], indent: indent + 2 });
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
      top.val[key] = child;
      stack.push({ val: child, indent });
    } else {
      if (Array.isArray(top.val)) {
        throw new Error(`Line ${lineNo + 1}: key '${key}' at sequence position.`);
      }
      top.val[key] = parseScalar(rest);
    }
  }

  return root;
}

/* ────────────────────────────── view ─────────────────────────── */

// Peek at current YAML to derive a live, non-editable view of the target
// namespace / name / kind without committing to a full POST. Swallow parse
// errors here — the dedicated parse-error banner handles user-visible feedback.
function peekManifest(text) {
  try {
    const m = parseYaml(text);
    return {
      namespace: m?.metadata?.namespace ?? null,
      name: m?.metadata?.name ?? null,
      kind: m?.kind ?? null,
      parseError: null,
    };
  } catch (e) {
    return { namespace: null, name: null, kind: null, parseError: e.message };
  }
}

export function Launch() {
  const [templateId, setTemplateId] = useState(TEMPLATES[0].id);
  const [yaml, setYaml] = useState(TEMPLATES[0].yaml);
  const [state, setState] = useState({ kind: 'idle' });
  const [prefillFrom, setPrefillFrom] = useState(null);

  // Consume a sessionStorage handoff from Run's RE-LAUNCH button. One-shot:
  // we clear it immediately so refreshing /launch doesn't keep re-prefilling.
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
    const t = TEMPLATES.find(t => t.id === id);
    if (!t) return;
    setTemplateId(id);
    setYaml(t.yaml);
    setState({ kind: 'idle' });
  }

  async function launch() {
    let manifest;
    try {
      manifest = parseYaml(yaml);
    } catch (e) {
      setState({ kind: 'err', msg: e.message, stage: 'parse' });
      return;
    }
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
    navigate(`/run/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`);
  }

  const activeTemplate = TEMPLATES.find(t => t.id === templateId);
  const peek = peekManifest(yaml);
  const canSubmit = state.kind !== 'submitting' && state.kind !== 'ok' && !peek.parseError;

  return html`
    <div class="v-launch" data-testid="page-launch">
      <header class="v-head">
        <div class="v-head-title">
          <span class="v-head-caret">▸</span>
          <h1>Launch a new run</h1>
        </div>
        <div class="v-head-meta">POST /api/v1/jobs</div>
      </header>

      <div class="launch-grid">
        <aside class="launch-templates" aria-label="Templates">
          <div class="launch-section-head">TEMPLATES</div>
          ${TEMPLATES.map(t => html`
            <button
              key=${t.id}
              class=${'launch-template' + (t.id === templateId ? ' is-active' : '')}
              onclick=${() => pickTemplate(t.id)}
              data-testid=${'launch-template-' + t.id}
              aria-pressed=${t.id === templateId}
            >
              <div class="launch-template-name">${t.name}</div>
              <div class="launch-template-desc">${t.desc}</div>
            </button>
          `)}
        </aside>

        <section class="launch-editor">
          ${prefillFrom && html`
            <div class="launch-prefill-notice" data-testid="launch-prefill-notice">
              <i class="ph ph-arrow-counter-clockwise"></i>
              Pre-filled from run:
              <span class="launch-prefill-ref">${prefillFrom.ns}/${prefillFrom.name}</span>
            </div>
          `}
          <header class="launch-editor-head">
            <div class="launch-editor-title">
              <span class="launch-editor-caret">▸</span>
              MANIFEST · ${activeTemplate?.name ?? 'custom'}
            </div>
            <button class="launch-copy" onclick=${copyYaml} title="Copy YAML to clipboard">
              <i class="ph ph-copy"></i>
              Copy
            </button>
          </header>

          <div
            class="launch-editor-head"
            style="border-top: 1px solid var(--edge-2); border-bottom: 1px solid var(--edge-2); gap: var(--s-3);"
            data-testid="launch-target"
          >
            <div class="launch-editor-title" style="gap: 10px; font-size: 10px; letter-spacing: 0.22em;">
              <span style="color: var(--paper-faint);">TARGET</span>
              ${peek.namespace
                ? html`<span class="run-header-ns-name" style="cursor: default;">${peek.namespace}</span>`
                : html`<span style="color: var(--red); font-family: var(--f-mono); font-size: 11px; letter-spacing: 0.18em;">NO NAMESPACE</span>`}
              <span class="run-header-ns-sep">/</span>
              ${peek.name
                ? html`<span style="font-family: var(--f-mono); font-size: 11px; color: var(--paper); letter-spacing: 0.04em; text-transform: none;">${peek.name}</span>`
                : html`<span style="color: var(--red); font-family: var(--f-mono); font-size: 11px; letter-spacing: 0.18em;">NO NAME</span>`}
            </div>
            <div class="launch-hint" style="padding: 0;">
              ${peek.kind ? `kind: ${peek.kind}` : 'kind: —'}
            </div>
          </div>

          <textarea
            class="launch-yaml"
            value=${yaml}
            oninput=${e => { setYaml(e.target.value); if (state.kind !== 'submitting') setState({ kind: 'idle' }); }}
            onkeydown=${onYamlKeydown}
            spellcheck="false"
            wrap="off"
            data-testid="launch-yaml"
          ></textarea>

          ${state.kind === 'ok' && html`
            <div
              class="run-header-phase run-header-phase--passed"
              style="display: flex; gap: var(--s-3); align-items: center; justify-content: space-between;
                     padding: var(--s-3) var(--s-4); font-size: 11px; letter-spacing: 0.2em;"
              data-testid="launch-success"
            >
              <div style="display: inline-flex; align-items: center; gap: var(--s-3); flex-wrap: wrap;">
                <strong style="letter-spacing: 0.28em;">CR CREATED</strong>
                <span class="run-header-ns-sep">·</span>
                <span class="run-header-ns-name" style="cursor: default;">${state.namespace}</span>
                <span class="run-header-ns-sep">/</span>
                <span style="font-family: var(--f-mono); font-size: 11px; color: var(--paper); letter-spacing: 0.04em; text-transform: none;">${state.name}</span>
              </div>
              <button
                class="launch-submit"
                onclick=${viewRun}
                data-testid="launch-view-run"
                style="padding: 6px 14px; font-size: 11px;"
              >
                <i class="ph ph-arrow-right"></i>
                VIEW RUN
              </button>
            </div>
          `}

          ${state.kind !== 'ok' && peek.parseError && html`
            <div class="v-analysis-err" style="margin: 0;" data-testid="launch-parse-err">
              YAML · ${peek.parseError}
            </div>
          `}

          ${state.kind === 'err' && html`
            <div class="run-results-err run-results--err" data-testid="launch-err"
                 style="display: flex; gap: var(--s-3); align-items: flex-start; border-top: 1px solid rgba(255, 89, 100, 0.3); padding: var(--s-3) var(--s-4);">
              <strong style="color: var(--red); font-weight: 700; letter-spacing: 0.24em; text-transform: uppercase; flex-shrink: 0; font-size: 11px;">
                ${state.stage === 'parse' ? 'YAML ERROR' : (state.status ? `HTTP ${state.status}` : 'LAUNCH FAILED')}
              </strong>
              <span style="color: var(--paper); letter-spacing: 0; text-transform: none; white-space: pre-wrap; word-break: break-word;">${state.msg}</span>
            </div>
          `}

          <footer class="launch-actions">
            <div class="launch-hint">
              <kbd>⌘Enter</kbd> submit · the operator creates the CR; kopf schedules workers.
            </div>
            <button
              class="launch-submit"
              disabled=${!canSubmit}
              onclick=${launch}
              data-testid="launch-submit"
            >
              <i class="ph ph-rocket-launch"></i>
              ${state.kind === 'submitting' ? 'LAUNCHING…' : state.kind === 'ok' ? 'LAUNCHED' : 'LAUNCH'}
            </button>
          </footer>
        </section>
      </div>
    </div>
  `;
}

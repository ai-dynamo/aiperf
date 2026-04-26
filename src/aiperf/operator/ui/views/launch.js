// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Launch — create a new AIPerfJob from the UI.
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

  // Consume a sessionStorage handoff from Run's Re-launch button. One-shot:
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
          <h1>Launch a new run</h1>
        </div>
        <div class="v-head-meta">POST /api/v1/jobs</div>
      </header>

      <div class="launch-templates" aria-label="Templates">
        ${TEMPLATES.map(t => html`
          <button
            key=${t.id}
            class=${'launch-template' + (t.id === templateId ? ' launch-template--active' : '')}
            onclick=${() => pickTemplate(t.id)}
            data-testid=${'launch-template-' + t.id}
            aria-pressed=${t.id === templateId}
            title=${t.desc}
          >${t.name}</button>
        `)}
      </div>

      ${prefillFrom && html`
        <div class="launch-prefill-notice" data-testid="launch-prefill-notice">
          <i class="ph ph-arrow-counter-clockwise"></i>
          Pre-filled from run: <strong>${prefillFrom.ns}/${prefillFrom.name}</strong>
        </div>
      `}

      <input
        class="launch-target"
        data-testid="launch-target"
        readonly
        value=${`${peek.namespace ?? '—'} / ${peek.name ?? '—'}  ·  kind: ${peek.kind ?? '—'}${activeTemplate ? `  ·  template: ${activeTemplate.name}` : ''}`}
      />

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
        <div class="launch-success" data-testid="launch-success">
          Created <strong>${state.namespace}/${state.name}</strong>
          <a
            class="btn btn--ghost"
            data-testid="launch-view-run"
            href=${`/run/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.name)}`}
            onclick=${(e) => { e.preventDefault(); viewRun(); }}
          >View run</a>
        </div>
      `}

      ${state.kind !== 'ok' && peek.parseError && html`
        <div class="v-analysis-err" data-testid="launch-parse-err">
          YAML · ${peek.parseError}
        </div>
      `}

      ${state.kind === 'err' && html`
        <div class="bench-error-flash" data-testid="launch-err">
          <strong>Error</strong>
          ${state.stage === 'parse' ? `YAML: ${state.msg}` : (state.status ? `HTTP ${state.status}: ${state.msg}` : state.msg)}
        </div>
      `}

      <div class="launch-actions">
        <button
          class="btn btn--ghost"
          onclick=${copyYaml}
          title="Copy YAML to clipboard"
        >Copy</button>
        <button
          class="btn btn--primary"
          disabled=${!canSubmit}
          onclick=${launch}
          data-testid="launch-submit"
        >${state.kind === 'submitting' ? 'Launching…' : state.kind === 'ok' ? 'Launched' : 'Launch'}</button>
      </div>
    </div>
  `;
}

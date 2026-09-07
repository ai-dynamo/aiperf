// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * YamlEditor — syntax-highlighted YAML editor built from a textarea/pre pair.
 *
 * `highlightYaml(text)` returns an array of `{cls, text}` tokens.
 * `cls` is a CSS class suffix ("key", "string", "num", ...) or null for
 * unstyled plain text. The `YamlEditor` component renders these as Preact
 * vnodes inside a `<pre>` overlay stacked on a transparent `<textarea>`,
 * keeping native caret and selection behaviour while adding per-token colour.
 */

import { html } from 'htm/preact';
import { useCallback, useRef } from 'preact/hooks';

/* ─────────────────────────── tokeniser ─────────────────────────── */

function keyColonIdx(s) {
  if (!s) return -1;
  if (s[0] === '"' || s[0] === "'" || s[0] === '{' || s[0] === '[') return -1;
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(s)) return -1;
  const m = s.match(/^([^:{"\[\]']*?)(:(?:\s|$))/);
  return m ? m[1].length : -1;
}

function inlineCommentIdx(s) {
  let q = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === q) q = null; continue; }
    if (c === '"' || c === "'") { q = c; continue; }
    if (c === '#' && (i === 0 || s[i - 1] === ' ')) return i;
  }
  return -1;
}

function pushValue(s, out) {
  s = s.trimEnd();
  if (!s) return;

  const ci = inlineCommentIdx(s);
  let commentStr = '';
  if (ci > 0 && s[ci - 1] === ' ') {
    commentStr = s.slice(ci);
    s = s.slice(0, ci).trimEnd();
    if (!s && commentStr) {
      out.push({ cls: null, text: ' ' });
      out.push({ cls: 'comment', text: commentStr });
      return;
    }
  }

  if ((s[0] === '"' || s[0] === "'") && s[s.length - 1] === s[0]) {
    out.push({ cls: 'string', text: s });
  } else if (s[0] === '{' || s[0] === '[') {
    out.push({ cls: 'flow', text: s });
  } else if (s[0] === '&' || s[0] === '*') {
    out.push({ cls: 'meta', text: s });
  } else if (s === 'true' || s === 'false') {
    out.push({ cls: 'bool', text: s });
  } else if (s === 'null' || s === '~') {
    out.push({ cls: 'null', text: s });
  } else if (/^[+\-]?(\d+\.?\d*|\.\d+)([eE][+\-]?\d+)?$/.test(s)) {
    out.push({ cls: 'num', text: s });
  } else {
    out.push({ cls: null, text: s });
  }

  if (commentStr) {
    out.push({ cls: null, text: ' ' });
    out.push({ cls: 'comment', text: commentStr });
  }
}

function tokeniseLine(line) {
  const out = [];
  const indent = line.match(/^(\s*)/)[1];
  const body = line.slice(indent.length);

  if (!body) {
    if (indent) out.push({ cls: null, text: indent });
    return out;
  }

  if (body[0] === '#') {
    if (indent) out.push({ cls: null, text: indent });
    out.push({ cls: 'comment', text: body });
    return out;
  }

  if (body === '---' || body === '...') {
    out.push({ cls: 'meta', text: line });
    return out;
  }

  if (indent) out.push({ cls: null, text: indent });

  if (body === '-' || body.startsWith('- ')) {
    const dash = body.startsWith('- ') ? '- ' : '-';
    out.push({ cls: 'dash', text: dash });
    const rest = body.slice(dash.length);
    if (rest) {
      const ki = keyColonIdx(rest);
      if (ki !== -1) {
        out.push({ cls: 'key', text: rest.slice(0, ki) });
        const afterColon = rest.slice(ki);
        const colonEnd = afterColon.match(/^:\s*/)[0].length;
        out.push({ cls: 'punct', text: afterColon.slice(0, colonEnd) });
        const val = afterColon.slice(colonEnd);
        if (val) pushValue(val, out);
      } else {
        pushValue(rest, out);
      }
    }
    return out;
  }

  const ki = keyColonIdx(body);
  if (ki !== -1) {
    out.push({ cls: 'key', text: body.slice(0, ki) });
    const afterColon = body.slice(ki);
    const colonEnd = afterColon.match(/^:\s*/)[0].length;
    out.push({ cls: 'punct', text: afterColon.slice(0, colonEnd) });
    const val = afterColon.slice(colonEnd);
    if (val) pushValue(val, out);
    return out;
  }

  pushValue(body, out);
  return out;
}

/**
 * Tokenise YAML text into an array of {cls, text} tokens.
 *
 * `cls` is one of: "key", "string", "num", "bool", "null", "comment",
 * "dash", "meta", "punct", "flow", or null for plain text.
 * Newlines between lines are emitted as plain-text tokens; a trailing
 * newline is appended to prevent scroll crop on the last line.
 */
export function highlightYaml(text) {
  const lines = text.split('\n');
  const out = [];
  lines.forEach((line, i) => {
    if (i > 0) out.push({ cls: null, text: '\n' });
    for (const tok of tokeniseLine(line)) out.push(tok);
  });
  out.push({ cls: null, text: '\n' });
  return out;
}

/* ───────────────────────── component ───────────────────────── */

/**
 * YamlEditor — controlled YAML editor with syntax highlighting.
 *
 * Props:
 *   value     controlled YAML string
 *   onInput   called with the native InputEvent when text changes
 *   onKeydown called with KeyboardEvent (Tab handling, etc.)
 *   testid    forwarded as data-testid on the hidden textarea
 */
export function YamlEditor({ value, onInput, onKeydown, testid }) {
  const taRef = useRef(null);
  const preRef = useRef(null);

  const syncScroll = useCallback(() => {
    const ta = taRef.current;
    const pre = preRef.current;
    if (!ta || !pre) return;
    pre.scrollTop = ta.scrollTop;
    pre.scrollLeft = ta.scrollLeft;
  }, []);

  const tokens = highlightYaml(value);

  return html`
    <div class="yaml-editor">
      <pre ref=${preRef} class="yaml-editor-hl" aria-hidden="true">${
        tokens.map(({ cls, text }, i) =>
          cls ? html`<span key=${i} class=${'yl-' + cls}>${text}</span>` : text
        )
      }</pre>
      <textarea
        ref=${taRef}
        class="yaml-editor-ta"
        value=${value}
        oninput=${onInput}
        onkeydown=${onKeydown}
        onscroll=${syncScroll}
        spellcheck="false"
        autocorrect="off"
        autocapitalize="off"
        wrap="off"
        data-testid=${testid}
      ></textarea>
    </div>
  `;
}

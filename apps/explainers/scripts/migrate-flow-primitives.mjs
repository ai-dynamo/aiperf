#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Mechanical Flow deck migrator: core.panel / core.header / motion.signal.
 * Only rewrites objects whose *own* top-level capability is core.rect.
 *
 * Usage: node apps/explainers/scripts/migrate-flow-primitives.mjs [deck-id...]
 */

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

const decksDir = new URL("../decks-flow/", import.meta.url);

function matchBraced(source, openIndex) {
  if (source[openIndex] !== "{") {
    return undefined;
  }
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = openIndex; i < source.length; i++) {
    const ch = source[i];
    if (inString) {
      if (escape) {
        escape = false;
        continue;
      }
      if (ch === "\\") {
        escape = true;
        continue;
      }
      if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        return {
          start: openIndex,
          end: i + 1,
          text: source.slice(openIndex, i + 1),
        };
      }
    }
  }
  return undefined;
}

/** Find `{` that opens the object containing `index` (must be inside that object). */
function enclosingObjectStart(source, index) {
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = index; i >= 0; i--) {
    const ch = source[i];
    if (inString) {
      if (escape) {
        escape = false;
        continue;
      }
      // walking backward: previous char being \\ is hard; approximate
      if (ch === '"') {
        // count preceding backslashes
        let bs = 0;
        let j = i - 1;
        while (j >= 0 && source[j] === "\\") {
          bs += 1;
          j -= 1;
        }
        if (bs % 2 === 0) {
          inString = false;
        }
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "}") {
      depth += 1;
      continue;
    }
    if (ch === "{") {
      if (depth === 0) {
        return i;
      }
      depth -= 1;
    }
  }
  return undefined;
}

function topLevelEntries(block) {
  if (block[0] !== "{" || block[block.length - 1] !== "}") {
    return [];
  }
  const body = block.slice(1, -1);
  const entries = [];
  let i = 0;
  let inString = false;
  let escape = false;
  while (i < body.length) {
    while (i < body.length && /[\s,]/.test(body[i])) {
      i += 1;
    }
    if (i >= body.length) {
      break;
    }
    const keyMatch = /^([A-Za-z_][A-Za-z0-9_]*)\s*:/.exec(body.slice(i));
    if (!keyMatch) {
      i += 1;
      continue;
    }
    const key = keyMatch[1];
    i += keyMatch[0].length;
    while (i < body.length && /\s/.test(body[i])) {
      i += 1;
    }
    if (i >= body.length) {
      break;
    }
    const start = i;
    const ch = body[i];
    if (ch === "{") {
      const nested = matchBraced(body, i);
      if (!nested) {
        break;
      }
      entries.push({ key, value: nested.text, kind: "object" });
      i = nested.end;
      continue;
    }
    if (ch === "[") {
      let depth = 0;
      let s = false;
      let e = false;
      let j = i;
      for (; j < body.length; j++) {
        const c = body[j];
        if (s) {
          if (e) {
            e = false;
            continue;
          }
          if (c === "\\") {
            e = true;
            continue;
          }
          if (c === '"') {
            s = false;
          }
          continue;
        }
        if (c === '"') {
          s = true;
          continue;
        }
        if (c === "[") {
          depth += 1;
        } else if (c === "]") {
          depth -= 1;
          if (depth === 0) {
            j += 1;
            break;
          }
        }
      }
      entries.push({ key, value: body.slice(i, j), kind: "array" });
      i = j;
      continue;
    }
    if (ch === '"') {
      let j = i + 1;
      let e = false;
      for (; j < body.length; j++) {
        const c = body[j];
        if (e) {
          e = false;
          continue;
        }
        if (c === "\\") {
          e = true;
          continue;
        }
        if (c === '"') {
          j += 1;
          break;
        }
      }
      entries.push({
        key,
        value: body.slice(i, j),
        kind: "string",
        unquoted: JSON.parse(body.slice(i, j)),
      });
      i = j;
      continue;
    }
    // number / bare ident / @theme...
    let j = i;
    while (j < body.length && !/[\s,]/.test(body[j]) && body[j] !== "\n") {
      j += 1;
    }
    // include trailing theme path tokens with dots
    entries.push({ key, value: body.slice(i, j).trim(), kind: "atom" });
    i = j;
    void inString;
    void escape;
    void start;
  }
  return entries;
}

function entryMap(block) {
  const map = new Map();
  for (const entry of topLevelEntries(block)) {
    if (!map.has(entry.key)) {
      map.set(entry.key, entry);
    }
  }
  return map;
}

function arrayObjects(arrayValue) {
  if (!arrayValue.startsWith("[") || !arrayValue.endsWith("]")) {
    return [];
  }
  const body = arrayValue.slice(1, -1);
  const objects = [];
  let i = 0;
  while (i < body.length) {
    if (body[i] === "{") {
      const matched = matchBraced(body, i);
      if (!matched) {
        break;
      }
      objects.push(matched.text);
      i = matched.end;
      continue;
    }
    i += 1;
  }
  return objects;
}

function indentOf(source, index) {
  let start = index;
  while (start > 0 && source[start - 1] !== "\n") {
    start -= 1;
  }
  const line = source.slice(start, index);
  const m = line.match(/^(\s*)/);
  return m ? m[1] : "        ";
}

function migrateMotionSignal(source) {
  let out = source.replace(
    /(id:\s*"[^"]*motion[-_]?sig[^"]*"\s*\n\s*capability:\s*)"core\.path"/g,
    '$1"motion.signal"',
  );
  out = out.replace(
    /(capability:\s*)"core\.path"([\s\S]{0,240}?accessibility:\s*\{\s*label:\s*"motion signal")/g,
    '$1"motion.signal"$2',
  );
  return out;
}

function migratePanelsAndHeaders(source) {
  const replacements = [];
  const re = /capability:\s*"core\.rect"/g;
  let match;
  while ((match = re.exec(source)) !== null) {
    const objStart = enclosingObjectStart(source, match.index);
    if (objStart === undefined) {
      continue;
    }
    const matched = matchBraced(source, objStart);
    if (!matched) {
      continue;
    }
    const fields = entryMap(matched.text);
    const capability = fields.get("capability");
    if (!capability || capability.unquoted !== "core.rect") {
      continue;
    }
    const children = fields.get("children");
    if (!children || children.kind !== "array") {
      continue;
    }
    const kids = arrayObjects(children.value);
    if (kids.length !== 2) {
      continue;
    }
    const kid0 = entryMap(kids[0]);
    const kid1 = entryMap(kids[1]);
    if (
      kid0.get("capability")?.unquoted !== "core.text" ||
      kid1.get("capability")?.unquoted !== "core.text"
    ) {
      continue;
    }
    const title = kid0.get("text")?.unquoted;
    const detail = kid1.get("text")?.unquoted;
    const id = fields.get("id")?.unquoted;
    const layout = fields.get("layout")?.value;
    const style = fields.get("style")?.value;
    if (
      title === undefined ||
      detail === undefined ||
      !id ||
      !layout ||
      !style
    ) {
      continue;
    }
    const indent = indentOf(source, objStart);
    const inner = `${indent}  `;
    const isHeader =
      id === "header" ||
      id.endsWith("-header") ||
      id.startsWith("header") ||
      (/\bwidth:\s*664\b/.test(layout) && /\bheight:\s*44\b/.test(layout)) ||
      (kid0.get("id")?.unquoted === "label" &&
        (kid1.get("id")?.unquoted ?? "").includes("caption"));
    const newCap = isHeader ? "core.header" : "core.panel";
    const secondKey = isHeader ? "caption" : "detail";
    const escapeFlowString = (value) =>
      value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
    const replacement = `{
${inner}id: "${id}"
${inner}capability: "${newCap}"
${inner}title: "${escapeFlowString(title)}"
${inner}${secondKey}: "${escapeFlowString(detail)}"
${inner}layout: ${layout}
${inner}style: ${style}
${indent}}`;
    replacements.push({ start: matched.start, end: matched.end, replacement });
  }

  // Apply from end so earlier offsets stay valid.
  replacements.sort((a, b) => b.start - a.start);
  // Deduplicate overlapping (same object matched twice)
  const used = [];
  let out = source;
  let count = 0;
  for (const rep of replacements) {
    if (used.some((u) => !(rep.end <= u.start || rep.start >= u.end))) {
      continue;
    }
    out = out.slice(0, rep.start) + rep.replacement + out.slice(rep.end);
    used.push(rep);
    count += 1;
  }
  return { source: out, replacements: count };
}

async function migrateFile(deckId) {
  const path = join(decksDir.pathname, `${deckId}.flow`);
  const before = await readFile(path, "utf8");
  const panelResult = migratePanelsAndHeaders(before);
  let after = migrateMotionSignal(panelResult.source);
  if (after === before) {
    console.log(`${deckId}: no mechanical changes`);
    return;
  }
  // Safety: never shrink below 50% of original size
  if (after.length < before.length * 0.5) {
    console.error(
      `${deckId}: ABORT — output shrank too much (${before.length}→${after.length})`,
    );
    process.exitCode = 1;
    return;
  }
  await writeFile(path, after);
  console.log(
    `${deckId}: panel/header=${panelResult.replacements} lines ${before.split("\n").length}→${after.split("\n").length} motion.signal=${(after.match(/capability: "motion\.signal"/g) || []).length}`,
  );
}

const args = process.argv.slice(2);
const decks =
  args.length > 0
    ? args
    : [
        "slurm-velo",
        "rust-architecture",
        "rust-architecture-atlas",
        "cellular-internals",
        "cellular-algorithms",
      ];

for (const deck of decks) {
  await migrateFile(deck);
}

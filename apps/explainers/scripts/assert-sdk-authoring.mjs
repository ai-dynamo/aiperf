#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Strict SDK-authoring gate — CLI scan.
 *
 * Walks `decks-flow/*.flow`, extracts every embedded `@scene` body, and reports
 * prohibited authoring signatures with source ranges:
 *
 *   - package-form `@scene { roots: … }` scenes (must migrate to native SDK
 *     component authoring),
 *   - raw bespoke primitives authored outside an explicit `freeform { … }`
 *     block (panels, headers, chrome, edges, fans, layout, motion, pulse).
 *
 * The scan mirrors the source-oriented detection in
 * `src/flow/compiler/validate-sdk-authoring.ts`. Enforcement is phased: the scan
 * reports by default (exit 0) so the pre-migration corpus stays usable, and
 * `--strict` fails (exit 1) once decks have migrated and the corpus must stay
 * clean.
 *
 * Usage:
 *   node apps/explainers/scripts/assert-sdk-authoring.mjs
 *   node apps/explainers/scripts/assert-sdk-authoring.mjs --deck dynosim
 *   node apps/explainers/scripts/assert-sdk-authoring.mjs --strict
 *   node apps/explainers/scripts/assert-sdk-authoring.mjs --json
 *   npm run assert:sdk-authoring
 */

import { readdir, readFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXPLAINERS_ROOT = resolve(__dirname, "..");
const DECKS_DIR = join(EXPLAINERS_ROOT, "decks-flow");

/** Prohibited native primitive keyword → { signature, replacement }. */
const PRIMITIVE_SIGNATURES = {
  rect: { signature: "raw rect + text panel signature", replacement: "sdk.card / sdk.panel" },
  panel: { signature: "raw panel chrome", replacement: "sdk.panel" },
  header: { signature: "raw header chrome", replacement: "sdk.header" },
  note: { signature: "raw note chrome", replacement: "sdk.note" },
  chip: { signature: "raw chip chrome", replacement: "sdk.chip" },
  callout: { signature: "raw callout chrome", replacement: "sdk.callout" },
  bracket: { signature: "raw bracket chrome", replacement: "sdk.bracket" },
  divider: { signature: "raw divider chrome", replacement: "sdk.divider" },
  connector: { signature: "raw connector edge", replacement: "sdk.edge" },
  route: { signature: "raw route edge", replacement: "sdk.edge" },
  arrow: { signature: "raw arrow edge", replacement: "sdk.edge" },
  elbow: { signature: "raw elbow edge", replacement: "sdk.edge" },
  "fan-out": { signature: "manual fan-out tree", replacement: "sdk.fanOut" },
  "fan-in": { signature: "manual fan-in tree", replacement: "sdk.fanIn" },
  stack: { signature: "manual stack placement", replacement: "sdk.stack" },
  grid: { signature: "manual grid placement", replacement: "sdk.grid" },
  rail: { signature: "manual rail placement", replacement: "sdk.rail" },
  lane: { signature: "manual lane placement", replacement: "sdk.lane" },
  band: { signature: "manual band placement", replacement: "sdk.band" },
  swimlane: { signature: "manual swimlane placement", replacement: "sdk.swimlane" },
  stepper: { signature: "manual stepper placement", replacement: "sdk.stepper" },
  signal: { signature: "raw motion signal", replacement: "sdk.signal" },
};

const PRIMITIVE_KEYWORDS = Object.keys(PRIMITIVE_SIGNATURES);

function parseArgs(argv) {
  const options = { deck: null, strict: false, json: false, help: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--deck") {
      options.deck = argv[++i] ?? null;
    } else if (arg.startsWith("--deck=")) {
      options.deck = arg.slice("--deck=".length);
    } else if (arg === "--strict") {
      options.strict = true;
    } else if (arg === "--json") {
      options.json = true;
    } else if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else {
      console.error(`unknown argument: ${arg}`);
      process.exit(2);
    }
  }
  return options;
}

function printHelp() {
  console.log(`Strict SDK-authoring gate — scan explainer decks for prohibited bespoke composition.

Usage:
  node apps/explainers/scripts/assert-sdk-authoring.mjs [options]

Options:
  --deck <id>   Only scan one deck id (e.g. dynosim)
  --strict      Fail (exit 1) when any prohibited signature is found
  --json        Emit findings as JSON
  -h, --help    Show this help

Detected signatures:
  - package-form @scene { roots: … }   → native sdk.* / aiperf.* authoring
  - raw rect / panel / header / chrome → sdk.card / sdk.panel / sdk.header / …
  - raw connector / route / arrow      → sdk.edge / sdk.pipeline
  - manual fan-out / fan-in trees      → sdk.fanOut / sdk.fanIn
  - manual layout placement            → sdk.stack / sdk.grid / sdk.rail / …
  - raw motion signal / pulse overlay  → sdk.signal / sdk.flow / sdk.pulse

Unique illustration primitives are permitted only inside an explicit
freeform { … } block. Enforcement is phased: the scan reports by default and
--strict fails once the corpus has migrated and must stay clean.
`);
}

/** Line number (1-based) for a source offset. */
function lineAt(source, offset) {
  let line = 1;
  for (let i = 0; i < offset && i < source.length; i += 1) {
    if (source[i] === "\n") {
      line += 1;
    }
  }
  return line;
}

/**
 * Returns the index just past the `}` that closes the `{` at `openIndex`,
 * skipping braces inside double-quoted strings and Flow comments (`//`, `/*`).
 * Comment rules mirror `src/flow/language/tokens.ts` (LineComment / BlockComment).
 */
export function matchBrace(source, openIndex) {
  let depth = 0;
  let inString = false;
  for (let i = openIndex; i < source.length; i += 1) {
    const char = source[i];
    if (inString) {
      if (char === "\\") {
        i += 1;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }
    if (char === '"') {
      inString = true;
      continue;
    }
    // Line comment: // … to end of line (Flow lexer LineComment).
    if (char === "/" && source[i + 1] === "/") {
      i += 2;
      while (i < source.length && source[i] !== "\n" && source[i] !== "\r") {
        i += 1;
      }
      continue;
    }
    // Block comment: /* … */ (Flow lexer BlockComment; non-greedy).
    if (char === "/" && source[i + 1] === "*") {
      i += 2;
      while (i < source.length) {
        if (source[i] === "*" && source[i + 1] === "/") {
          i += 1; // loop will +1 past '/'
          break;
        }
        i += 1;
      }
      continue;
    }
    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return i + 1;
      }
    }
  }
  return -1;
}

/** Extracts every `@scene { … }` body with its owning label and offsets. */
export function extractScenes(source) {
  const scenes = [];
  const re = /(render|finalCard)\s*:\s*@scene\s*\{/g;
  let match;
  while ((match = re.exec(source)) !== null) {
    const openIndex = source.indexOf("{", match.index);
    if (openIndex === -1) {
      continue;
    }
    const end = matchBrace(source, openIndex);
    if (end === -1) {
      continue;
    }
    scenes.push({
      label: match[1],
      bodyStart: openIndex + 1,
      body: source.slice(openIndex + 1, end - 1),
      headerOffset: match.index,
    });
    re.lastIndex = end;
  }
  return scenes;
}

/** Replaces every `freeform { … }` block body with spaces (length-preserving). */
function blankFreeformBlocks(body) {
  const re = /freeform\b[^{]*\{/g;
  let out = body;
  let match;
  while ((match = re.exec(out)) !== null) {
    const openIndex = out.indexOf("{", match.index);
    if (openIndex === -1) {
      break;
    }
    const end = matchBrace(out, openIndex);
    if (end === -1) {
      break;
    }
    const blanked = out
      .slice(match.index, end)
      .replace(/[^\n]/g, " ");
    out = out.slice(0, match.index) + blanked + out.slice(end);
    re.lastIndex = end;
  }
  return out;
}

function detectSceneForm(body) {
  const trimmed = body.replace(/^\s+/, "");
  return /^(roots|timeline|camera)\s*:/.test(trimmed) ? "package" : "native";
}

/**
 * Detects prohibited signatures in one native scene body (freeform blocks
 * already blanked). Matches primitive keywords at statement start.
 */
function detectNativeSignatures(deck, label, sceneBody, sceneBodyStart, source) {
  const findings = [];
  const scanBody = blankFreeformBlocks(sceneBody);

  const keywordRe = new RegExp(
    `(^|[\\n{])\\s*(${PRIMITIVE_KEYWORDS.map((k) => k.replace("-", "\\-")).join("|")})\\b`,
    "g",
  );
  let match;
  while ((match = keywordRe.exec(scanBody)) !== null) {
    const keyword = match[2];
    const spec = PRIMITIVE_SIGNATURES[keyword];
    if (spec === undefined) {
      continue;
    }
    const offset = sceneBodyStart + match.index + match[0].indexOf(keyword);
    findings.push({
      deck,
      scene: label,
      code: "SDK_AUTHORING_RAW_PRIMITIVE",
      signature: spec.signature,
      replacement: spec.replacement,
      line: lineAt(source, offset),
    });
  }

  const pulseRe = /pulse\s*:\s*true/g;
  while ((match = pulseRe.exec(scanBody)) !== null) {
    const offset = sceneBodyStart + match.index;
    findings.push({
      deck,
      scene: label,
      code: "SDK_AUTHORING_RAW_PULSE",
      signature: "pulse-flag overlay",
      replacement: "sdk.pulse",
      line: lineAt(source, offset),
    });
  }

  return findings;
}

async function scanDeck(deckId, filePath) {
  const source = await readFile(filePath, "utf8");
  const scenes = extractScenes(source);
  const findings = [];
  for (const scene of scenes) {
    const label = `${scene.label}@line-${lineAt(source, scene.headerOffset)}`;
    if (detectSceneForm(scene.body) === "package") {
      findings.push({
        deck: deckId,
        scene: label,
        code: "SDK_AUTHORING_PACKAGE_FORM",
        signature: "package-form @scene { roots: … }",
        replacement: "native sdk.* / aiperf.* authoring",
        line: lineAt(source, scene.headerOffset),
      });
      continue;
    }
    findings.push(
      ...detectNativeSignatures(
        deckId,
        label,
        scene.body,
        scene.bodyStart,
        source,
      ),
    );
  }
  return { sceneCount: scenes.length, findings };
}

async function listDeckFiles(deckFilter) {
  const names = (await readdir(DECKS_DIR))
    .filter((name) => name.endsWith(".flow"))
    .sort();
  const files = [];
  for (const name of names) {
    const id = name.replace(/\.flow$/, "");
    if (deckFilter && id !== deckFilter) {
      continue;
    }
    files.push({ id, path: join(DECKS_DIR, name) });
  }
  if (deckFilter && files.length === 0) {
    throw new Error(`no deck "${deckFilter}.flow" in ${relative(EXPLAINERS_ROOT, DECKS_DIR)}`);
  }
  return files;
}

function formatFinding(f) {
  return `[${f.deck}] ${f.scene} L${f.line} ${f.code}: prohibited ${f.signature} → author with ${f.replacement}`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    process.exit(0);
  }

  const files = await listDeckFiles(options.deck);
  const allFindings = [];
  let totalScenes = 0;
  for (const file of files) {
    const { sceneCount, findings } = await scanDeck(file.id, file.path);
    totalScenes += sceneCount;
    allFindings.push(...findings);
  }

  if (options.json) {
    console.log(
      JSON.stringify(
        {
          decks: files.length,
          scenes: totalScenes,
          prohibited: allFindings.length,
          findings: allFindings,
        },
        null,
        2,
      ),
    );
  } else {
    for (const f of allFindings) {
      console.error(formatFinding(f));
    }
    console.error(
      `summary: ${allFindings.length} prohibited signature(s) across ${totalScenes} scene(s) in ${files.length} deck(s)`,
    );
    if (allFindings.length === 0) {
      console.log("OK: zero prohibited SDK-authoring signatures.");
    }
  }

  const failed = options.strict && allFindings.length > 0;
  process.exit(failed ? 1 : 0);
}

const isDirectRun =
  process.argv[1] != null &&
  resolve(fileURLToPath(import.meta.url)) === resolve(process.argv[1]);

if (isDirectRun) {
  main().catch((error) => {
    console.error(String(error?.stack ?? error));
    process.exit(2);
  });
}

#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Flow Verifier — IR playhead + Playwright full-play gate for explainer decks.
 *
 * There is no generated-package build step — `.flow` is the only source of
 * truth. The IR pass compiles `decks-flow/*.flow` in-memory (via
 * `scripts/compile-decks.ts` under `vite-node`), the same strict
 * `compileExplainerSource` + `validateExplainerSet` pipeline the live app
 * uses in `src/core/load-deck-flows.ts`.
 *
 * `npm run flow-verifier` / `npm run flow-verifier:ir` (IR-only, fast) is
 * the default gate. The Playwright full-deck walk is slow and lives behind
 * `npm run flow-verifier:extended` (or `--play-only` / omitting `--ir-only`
 * when invoking this script directly) — run it before landing changes that
 * touch timeline/scene rendering, not on every invocation.
 *
 * Usage:
 *   node apps/explainers/scripts/flow-verifier.mjs --ir-only
 *   node apps/explainers/scripts/flow-verifier.mjs --deck segment-pools
 *   node apps/explainers/scripts/flow-verifier.mjs
 *   node apps/explainers/scripts/flow-verifier.mjs --play-only --base-url http://127.0.0.1:5173
 *   node apps/explainers/scripts/flow-verifier.mjs --warn
 *   node apps/explainers/scripts/flow-verifier.mjs --verbose
 *   node apps/explainers/scripts/flow-verifier.mjs --play-only --preview
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { verifyAdvancedCurveRouting, verifyPackageIr } from "./flow-verifier/ir.mjs";
import { verifyPlayAll } from "./flow-verifier/play.mjs";

const execFileAsync = promisify(execFile);

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXPLAINERS_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const options = {
    deck: null,
    irOnly: false,
    playOnly: false,
    baseUrl: null,
    warn: false,
    verbose: false,
    strictDraw: false,
    preview: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--deck") {
      options.deck = argv[++i] ?? null;
    } else if (arg.startsWith("--deck=")) {
      options.deck = arg.slice("--deck=".length);
    } else if (arg === "--ir-only") {
      options.irOnly = true;
    } else if (arg === "--play-only") {
      options.playOnly = true;
    } else if (arg === "--base-url") {
      options.baseUrl = argv[++i] ?? null;
    } else if (arg.startsWith("--base-url=")) {
      options.baseUrl = arg.slice("--base-url=".length);
    } else if (arg === "--warn") {
      options.warn = true;
    } else if (arg === "--verbose") {
      options.verbose = true;
    } else if (arg === "--strict-draw") {
      options.strictDraw = true;
    } else if (arg === "--preview") {
      options.preview = true;
    } else if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else {
      console.error(`unknown argument: ${arg}`);
      process.exit(2);
    }
  }
  if (options.irOnly && options.playOnly) {
    console.error("use only one of --ir-only or --play-only");
    process.exit(2);
  }
  return options;
}

function printHelp() {
  console.log(`Flow Verifier — gate Scene IR + live playback for explainer decks.

Usage:
  node apps/explainers/scripts/flow-verifier.mjs [options]

Options:
  --deck <id>          Only verify one deck id (e.g. segment-pools)
  --ir-only            Skip Playwright full play (npm run flow-verifier / flow-verifier:ir)
  --play-only          Skip IR playhead pass
  --base-url <url>     Reuse an existing Vite server (skip spawn)
  --strict-draw        Emit IR warn for mid-draw arrow moments (SceneRenderer defers heads)
  --preview            Play against \`npm run build && npm run preview\` instead of \`vite dev\`
                        (proves production bundling; skip during active deck/compiler migration —
                        see .superpowers/sdd/playwright-hardening-report.md)
  --warn               Treat warnings as failures (exit non-zero)
  --verbose            Include informational resolver corrections
  -h, --help           Show this help

Default npm scripts:
  flow-verifier / flow-verifier:ir   IR-only (fast)
  flow-verifier:extended             IR + Playwright full-play (slow)

Playwright resolves from apps/explainers; install browsers with:
  cd apps/explainers && npx playwright install chromium
`);
}

async function loadPackages(deckFilter) {
  const { stdout } = await execFileAsync(
    "npx",
    ["vite-node", resolve(EXPLAINERS_ROOT, "scripts/compile-decks.ts")],
    { cwd: EXPLAINERS_ROOT, maxBuffer: 64 * 1024 * 1024 },
  );
  const bundle = JSON.parse(stdout);
  if (
    bundle == null ||
    !Array.isArray(bundle.packages) ||
    !Array.isArray(bundle.resolvedScenes)
  ) {
    throw new Error("compile-decks.ts returned an invalid verifier bundle");
  }
  const packages = bundle.packages;
  const filtered = deckFilter
    ? packages.filter((pkg) => pkg.id === deckFilter)
    : packages;
  if (deckFilter && filtered.length === 0) {
    throw new Error(`no compiled package for deck "${deckFilter}"`);
  }
  return {
    packages: filtered,
    resolvedScenes: deckFilter
      ? bundle.resolvedScenes.filter(({ deckId }) => deckId === deckFilter)
      : bundle.resolvedScenes,
  };
}

function formatFinding(f) {
  const location =
    typeof f.source === "string"
      ? ` ${f.source}:${f.line ?? 1}:${f.column ?? 1}`
      : "";
  return `[${f.severity}] ${f.deck} ${f.slide}${location} ${f.code}: ${f.message}`;
}

function summarize(findings) {
  const errors = findings.filter((f) => f.severity === "error");
  const warns = findings.filter((f) => f.severity === "warn");
  return { errors, warns };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    process.exit(0);
  }

  /** @type {import("./flow-verifier/ir.mjs").Finding[]} */
  const findings = [];

  if (!options.playOnly) {
    console.error("IR: verifying advanced curve routing matrix…");
    findings.push(...verifyAdvancedCurveRouting());
    const { packages, resolvedScenes } = await loadPackages(options.deck);
    console.error(`IR: verifying ${packages.length} package(s)…`);
    for (const pkg of packages) {
      findings.push(
        ...verifyPackageIr(pkg, {
          strictDraw: options.strictDraw,
          snapshots: resolvedScenes.filter(({ deckId }) => deckId === pkg.id),
        }),
      );
    }
  }

  if (!options.irOnly) {
    console.error("Play: full-deck Playwright walk…");
    findings.push(
      ...(await verifyPlayAll({
        deckRoute: options.deck,
        baseUrl: options.baseUrl,
        preview: options.preview,
      })),
    );
  }

  const { errors, warns } = summarize(findings);
  for (const f of findings) {
    if (f.severity === "info" && !options.verbose) continue;
    console.error(formatFinding(f));
  }
  console.error(
    `summary: ${errors.length} error(s), ${warns.length} warn(s)`,
  );

  const failed =
    errors.length > 0 || (options.warn && warns.length > 0);
  process.exit(failed ? 1 : 0);
}

main().catch((error) => {
  console.error(String(error?.stack ?? error));
  process.exit(2);
});

#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Flow Verifier — IR playhead + Playwright full-play gate for explainer decks.
 *
 * `npm run flow-verifier` (IR-only, fast) is the default gate. The
 * Playwright full-deck walk is slow and lives behind
 * `npm run flow-verifier:extended` (or `--play-only` / omitting `--ir-only`
 * when invoking this script directly) — run it before landing changes that
 * touch timeline/scene rendering, not on every invocation.
 *
 * Usage:
 *   node apps/explainers/scripts/flow-verifier.mjs --ir-only
 *   node apps/explainers/scripts/flow-verifier.mjs --deck segment-pools
 *   node apps/explainers/scripts/flow-verifier.mjs
 *   node apps/explainers/scripts/flow-verifier.mjs --play-only --base-url http://127.0.0.1:5173
 *   node apps/explainers/scripts/flow-verifier.mjs --from-flow
 *   node apps/explainers/scripts/flow-verifier.mjs --warn
 *   node apps/explainers/scripts/flow-verifier.mjs --play-only --preview
 */

import { readdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { verifyPackageIr } from "./flow-verifier/ir.mjs";
import { verifyPlayAll } from "./flow-verifier/play.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXPLAINERS_ROOT = resolve(__dirname, "..");
const PACKAGES_DIR = join(EXPLAINERS_ROOT, "src/decks-generated");

function parseArgs(argv) {
  const options = {
    deck: null,
    irOnly: false,
    playOnly: false,
    baseUrl: null,
    warn: false,
    fromFlow: false,
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
    } else if (arg === "--from-flow") {
      options.fromFlow = true;
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
  --ir-only            Skip Playwright full play
  --play-only          Skip IR playhead pass
  --base-url <url>     Reuse an existing Vite server (skip spawn)
  --from-flow          Rebuild decks-generated from decks-flow/*.flow first
  --strict-draw        Emit IR warn for mid-draw arrow moments (SceneRenderer defers heads)
  --preview            Play against \`npm run build && npm run preview\` instead of \`vite dev\`
                        (proves production bundling; skip during active deck/compiler migration —
                        see .superpowers/sdd/playwright-hardening-report.md)
  --warn               Treat warnings as failures (exit non-zero)
  -h, --help           Show this help

Playwright resolves from apps/explainers; install browsers with:
  cd apps/explainers && npx playwright install chromium
`);
}

function run(command, args, cwd) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: "inherit",
      shell: false,
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolveRun();
      else reject(new Error(`${command} ${args.join(" ")} exited ${code}`));
    });
  });
}

async function rebuildFromFlow() {
  await run("npm", ["run", "build:explainer-packages"], EXPLAINERS_ROOT);
}

async function loadPackages(deckFilter) {
  const names = (await readdir(PACKAGES_DIR))
    .filter((name) => name.endsWith(".package.json"))
    .sort();
  const packages = [];
  for (const name of names) {
    const id = name.replace(/\.package\.json$/, "");
    if (deckFilter && id !== deckFilter) continue;
    const raw = await readFile(join(PACKAGES_DIR, name), "utf8");
    packages.push(JSON.parse(raw));
  }
  if (deckFilter && packages.length === 0) {
    throw new Error(`no package for deck "${deckFilter}" in ${PACKAGES_DIR}`);
  }
  return packages;
}

function formatFinding(f) {
  return `[${f.severity}] ${f.deck} ${f.slide} ${f.code}: ${f.message}`;
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

  if (options.fromFlow) {
    console.error("rebuilding explainer packages from .flow…");
    await rebuildFromFlow();
  }

  /** @type {import("./flow-verifier/ir.mjs").Finding[]} */
  const findings = [];

  if (!options.playOnly) {
    const packages = await loadPackages(options.deck);
    console.error(`IR: verifying ${packages.length} package(s)…`);
    for (const pkg of packages) {
      findings.push(
        ...verifyPackageIr(pkg, { strictDraw: options.strictDraw }),
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

#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Assert that `src/decks-generated/*.package.json` holds all registered
 * explainer decks (packages-only registry inputs), each with slides that carry
 * non-empty narration, and that any slide with `render` present has a non-empty
 * `scene.timeline` (SceneRenderer animation contract).
 *
 * Part of the flow-backed explainers gate:
 *   one `.flow` per deck → DeckPackage → packages-only registry (no MentalModel).
 *
 * Usage:
 *   node scripts/assert-deck-packages.mjs
 *   npm run assert:deck-packages
 *   make assert-deck-packages
 *   make assert-explainer-packages
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const GENERATED = resolve(ROOT, "src/decks-generated");

/** Canonical package ids — must match deck-registry EXPECTED_DECK_ROUTES. */
const EXPECTED_DECK_IDS = [
  "rust-architecture",
  "rust-architecture-atlas",
  "segment-pools",
  "slurm-velo",
  "velo-deep-dive",
  "cellular-internals",
  "cellular-algorithms",
  "dynosim",
  "tstar-warmup",
];

function isFile(path) {
  try {
    return existsSync(path) && statSync(path).isFile();
  } catch {
    return false;
  }
}

function listPackageFiles() {
  if (!existsSync(GENERATED)) {
    return [];
  }
  return readdirSync(GENERATED)
    .filter((name) => name.endsWith(".package.json"))
    .map((name) => join(GENERATED, name))
    .filter(isFile)
    .sort();
}

function sceneTimeline(render) {
  if (render == null || typeof render !== "object") return null;
  const scene = render.scene;
  if (scene == null || typeof scene !== "object") return null;
  return scene.timeline;
}

function assertSceneTimeline(errors, label, render) {
  const timeline = sceneTimeline(render);
  if (!Array.isArray(timeline) || timeline.length === 0) {
    errors.push(`${label}: render present but scene.timeline is empty`);
  }
}

function validatePackage(filePath, errors) {
  const rel = relative(ROOT, filePath);
  let raw;
  try {
    raw = readFileSync(filePath, "utf8");
  } catch (err) {
    errors.push(`${rel}: failed to read (${err.message})`);
    return undefined;
  }

  let pkg;
  try {
    pkg = JSON.parse(raw);
  } catch (err) {
    errors.push(`${rel}: invalid JSON (${err.message})`);
    return undefined;
  }

  if (pkg == null || typeof pkg !== "object" || Array.isArray(pkg)) {
    errors.push(`${rel}: package root must be an object`);
    return undefined;
  }

  const id = typeof pkg.id === "string" ? pkg.id.trim() : "";
  if (!id) {
    errors.push(`${rel}: missing non-empty id`);
    return undefined;
  }

  if (!Array.isArray(pkg.slides) || pkg.slides.length === 0) {
    errors.push(`${id}: no slides`);
  } else {
    pkg.slides.forEach((slide, index) => {
      const slideLabel = `${id}: slide ${index + 1}`;
      if (slide == null || typeof slide !== "object") {
        errors.push(`${slideLabel}: not an object`);
        return;
      }
      const narration =
        typeof slide.narration === "string" ? slide.narration.trim() : "";
      if (!narration) {
        errors.push(`${slideLabel}: missing narration`);
      }
      if (slide.render != null) {
        assertSceneTimeline(errors, slideLabel, slide.render);
      }
    });
  }

  if (pkg.finalCard != null) {
    assertSceneTimeline(errors, `${id}: finalCard`, pkg.finalCard);
  }

  return id;
}

const errors = [];
const packageFiles = listPackageFiles();

if (packageFiles.length === 0) {
  errors.push(
    `no *.package.json under ${relative(ROOT, GENERATED)} (expected ${EXPECTED_DECK_IDS.length} decks)`,
  );
}

const foundIds = new Set();
for (const filePath of packageFiles) {
  const id = validatePackage(filePath, errors);
  if (id === undefined) continue;
  if (foundIds.has(id)) {
    errors.push(`duplicate package id: ${id}`);
  }
  foundIds.add(id);
}

if (foundIds.size !== EXPECTED_DECK_IDS.length) {
  errors.push(
    `expected ${EXPECTED_DECK_IDS.length} decks, found ${foundIds.size}`,
  );
}

for (const expectedId of EXPECTED_DECK_IDS) {
  if (!foundIds.has(expectedId)) {
    errors.push(`missing deck id: ${expectedId}`);
  }
}

for (const id of [...foundIds].sort()) {
  if (!EXPECTED_DECK_IDS.includes(id)) {
    errors.push(`unexpected deck id: ${id}`);
  }
}

if (errors.length > 0) {
  console.error("assert-deck-packages failed:");
  for (const error of errors) {
    console.error(`  - ${error}`);
  }
  process.exit(1);
}

console.log(
  `OK: ${EXPECTED_DECK_IDS.length} decks-generated packages have slides with non-empty narration and scene timelines when render is present.`,
);

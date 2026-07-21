#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// One-shot migration: rescale every authored geometry value in decks-flow/*.flow
// from the old 700x400 (7:4) implicit canvas to the new 1920x1080 (16:9) canvas,
// so existing decks fill the new default viewport instead of letterboxing.
//
// Scale factor is height-fit (400 -> 1080, factor 2.7); x gets a +15 offset to
// center the resulting 1890-wide content inside the 1920-wide canvas.

import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const SCALE = 2.7;
const X_OFFSET = 15;
const DECKS_DIR = join(import.meta.dirname, "..", "decks-flow");

// Keys authored in `key = number` form whose values are plain-offset spatial
// units (no x-offset applied): width, height, fontSize, strokeWidth, gap, padding.
const SCALE_ONLY_EQ_KEYS = ["width", "height", "fontSize", "strokeWidth", "gap", "padding"];
// Keys authored in `key: number` form (inside style={} object literals).
const SCALE_ONLY_COLON_KEYS = ["strokeWidth"];

function rescaleFile(path) {
  let text = readFileSync(path, "utf8");

  // x = N  -> scaled + centering offset
  text = text.replace(/\bx(\s*=\s*)(-?[0-9]+(?:\.[0-9]+)?)/g, (_m, sep, num) => {
    const scaled = Number(num) * SCALE + X_OFFSET;
    return `x${sep}${round(scaled)}`;
  });

  // y = N -> scaled, no offset
  text = text.replace(/\by(\s*=\s*)(-?[0-9]+(?:\.[0-9]+)?)/g, (_m, sep, num) => {
    return `y${sep}${round(Number(num) * SCALE)}`;
  });

  for (const key of SCALE_ONLY_EQ_KEYS) {
    const re = new RegExp(`\\b${key}(\\s*=\\s*)(-?[0-9]+(?:\\.[0-9]+)?)`, "g");
    text = text.replace(re, (_m, sep, num) => `${key}${sep}${round(Number(num) * SCALE)}`);
  }

  for (const key of SCALE_ONLY_COLON_KEYS) {
    const re = new RegExp(`\\b${key}(\\s*:\\s*)(-?[0-9]+(?:\\.[0-9]+)?)`, "g");
    text = text.replace(re, (_m, sep, num) => `${key}${sep}${round(Number(num) * SCALE)}`);
  }

  writeFileSync(path, text, "utf8");
}

function round(n) {
  return Math.round(n * 100) / 100;
}

const files = readdirSync(DECKS_DIR).filter((f) => f.endsWith(".flow"));
for (const file of files) {
  rescaleFile(join(DECKS_DIR, file));
  console.log(`rescaled ${file}`);
}
